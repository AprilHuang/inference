"""Stage 0 baseline: Qwen3-1.7B FP32 latency / memory benchmark.

Metrics
  TTFT  (time to first token)  : prefill forward over the prompt + pick first token
  TPOT  (time per output token): mean latency of each subsequent decode step
  Peak memory                  : torch.cuda.max_memory_allocated during a run
                                 (weights + KV cache + activations), also reports
                                 weights-only memory and max_memory_reserved.

Prefill and decode are driven manually (not model.generate) so each phase can be
timed precisely with CUDA events. Decoding is greedy and ignores EOS, so every
run produces exactly --output-len tokens.

Example
  python bench_fp32.py --prompt-lens 128 512 2048 --output-len 128 --runs 5
"""

import argparse
import json
import os
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--prompt-lens", type=int, nargs="+", default=[128, 512, 2048])
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--allow-tf32", action="store_true",
                   help="Allow TF32 matmuls (off by default to keep true FP32).")
    p.add_argument("--json", help="Optional path to dump results as JSON.")
    return p.parse_args()


def make_prompt(tok, prompt_len, batch_size, device):
    """Build a prompt of exactly prompt_len tokens from natural text."""
    text = ("The quick brown fox jumps over the lazy dog. "
            "Large language models generate text one token at a time. ")
    ids = tok(text, add_special_tokens=False).input_ids
    ids = (ids * (prompt_len // len(ids) + 1))[:prompt_len]
    return torch.tensor([ids] * batch_size, device=device)


@torch.inference_mode()
def run_once(model, input_ids, output_len):
    """Return (ttft_ms, list of per-step decode ms)."""
    cache = DynamicCache()
    start, first = torch.cuda.Event(True), torch.cuda.Event(True)
    step_events = [torch.cuda.Event(True) for _ in range(output_len - 1)]

    torch.cuda.synchronize()
    start.record()
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    first.record()

    for ev in step_events:
        out = model(input_ids=next_tok, past_key_values=cache, use_cache=True)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ev.record()
    torch.cuda.synchronize()

    ttft = start.elapsed_time(first)
    prev, steps = first, []
    for ev in step_events:
        steps.append(prev.elapsed_time(ev))
        prev = ev
    return ttft, steps


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32).to(device).eval()
    torch.cuda.synchronize()
    weights_gb = torch.cuda.memory_allocated() / 1e9
    n_params = sum(p.numel() for p in model.parameters())

    print(f"GPU            : {torch.cuda.get_device_name()}")
    print(f"torch / TF32   : {torch.__version__} / {args.allow_tf32}")
    print(f"model / dtype  : {args.model} / {next(model.parameters()).dtype}")
    print(f"params         : {n_params / 1e9:.3f} B")
    print(f"weights memory : {weights_gb:.2f} GB")
    print(f"batch / output : {args.batch_size} / {args.output_len} tokens, "
          f"warmup {args.warmup}, runs {args.runs}\n")

    header = (f"{'prompt':>7} {'TTFT ms':>10} {'TPOT ms':>9} {'tok/s':>8} "
              f"{'peak alloc GB':>14} {'peak rsvd GB':>13}")
    print(header)
    print("-" * len(header))

    results = []
    for plen in args.prompt_lens:
        input_ids = make_prompt(tok, plen, args.batch_size, device)
        for _ in range(args.warmup):
            run_once(model, input_ids, args.output_len)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ttfts, tpots = [], []
        for _ in range(args.runs):
            ttft, steps = run_once(model, input_ids, args.output_len)
            ttfts.append(ttft)
            tpots.append(statistics.mean(steps))
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_rsvd = torch.cuda.max_memory_reserved() / 1e9

        r = {
            "prompt_len": plen,
            "batch_size": args.batch_size,
            "output_len": args.output_len,
            "ttft_ms_median": statistics.median(ttfts),
            "tpot_ms_median": statistics.median(tpots),
            "ttft_ms_all": ttfts,
            "tpot_ms_all": tpots,
            "decode_tok_per_s": 1000 * args.batch_size / statistics.median(tpots),
            "peak_alloc_gb": peak_alloc,
            "peak_reserved_gb": peak_rsvd,
        }
        results.append(r)
        print(f"{plen:>7} {r['ttft_ms_median']:>10.2f} {r['tpot_ms_median']:>9.2f} "
              f"{r['decode_tok_per_s']:>8.1f} {peak_alloc:>14.2f} {peak_rsvd:>13.2f}")

    if args.json:
        meta = {"model": args.model, "dtype": "float32", "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__, "allow_tf32": args.allow_tf32,
                "weights_gb": weights_gb, "n_params": n_params}
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)
        print(f"\nsaved -> {args.json}")


if __name__ == "__main__":
    main()
