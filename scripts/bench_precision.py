# -*- coding: utf-8 -*-
"""对比不同精度的推理速度与数值稳定性（fp16 / bf16 / fp32）。

fp16 在上游被 `resolve_runtime_dtype` 硬性禁止，本脚本在运行时 monkeypatch 放行，
不修改任何文件——所以即使没打 fp16 补丁也能直接跑出基准。

示例：
    python scripts/bench_precision.py --precisions fp16,bf16 --steps 40,24
    python scripts/bench_precision.py --lora-dir ""          # 不加载 LoRA，测纯基座
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

import irodori_tts.inference_runtime as ir
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

_ORIG_RESOLVE = ir.resolve_runtime_dtype


def _allow_fp16(*, precision: str, device: torch.device) -> torch.dtype:
    if str(precision).strip().lower() in ("fp16", "float16", "half"):
        if device.type not in ("cuda", "xpu"):
            raise ValueError("fp16 requires CUDA or XPU")
        return torch.float16
    return _ORIG_RESOLVE(precision=precision, device=device)


ir.resolve_runtime_dtype = _allow_fp16

DEFAULT_TEXTS = [
    ("short", "こんにちは、今日もいい天気だね。"),
    (
        "medium",
        "おはようございます。今日はとてもいい天気ですね。"
        "こんな日は家にいるよりも、外に出て散歩でもしたほうが気持ちいいと思います。",
    ),
    (
        "long",
        "おはようございます。今日はとてもいい天気ですね。"
        "こんな日は家にいるよりも、外に出て散歩でもしたほうが気持ちいいと思います。"
        "それに、最近はずっと忙しかったので、たまにはゆっくり歩きながら話すのもいいんじゃないかな。"
        "どう思う？",
    ),
]


@dataclass
class Row:
    precision: str
    case: str
    steps: int
    sync: float
    audio: float
    rtf: float
    nan: int
    inf: int
    peak: float
    rms: float


def run(args: argparse.Namespace) -> list[Row]:
    rows: list[Row] = []
    texts = DEFAULT_TEXTS if not args.quick else DEFAULT_TEXTS[:2]
    for precision in [p.strip() for p in args.precisions.split(",") if p.strip()]:
        t0 = time.perf_counter()
        runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=args.checkpoint,
                model_device="cuda",
                codec_repo=args.codec,
                codec_device="cuda",
                model_precision=precision,
                codec_precision="fp32",
            )
        )
        load = time.perf_counter() - t0
        dtype = next(runtime.model.parameters()).dtype
        print(f"\n[{precision}] load={load:.1f}s dtype={dtype}")

        for name, text in texts:
            for steps in [int(s) for s in args.steps.split(",") if s.strip()]:
                req = SamplingRequest(
                    text=text,
                    ref_wav=args.ref_wav,
                    lora_adapter=args.lora_dir or None,
                    num_steps=steps,
                    seed=args.seed,
                )
                try:
                    t = time.perf_counter()
                    res = runtime.synthesize(req, log_fn=lambda m: None)
                    sync = time.perf_counter() - t
                except Exception as exc:  # noqa: BLE001
                    print(f"  {name:7s} steps={steps:2d} FAILED {type(exc).__name__}: {exc}")
                    continue

                audio = res.audio
                dur = audio.shape[-1] / res.sample_rate
                row = Row(
                    precision=precision,
                    case=name,
                    steps=steps,
                    sync=sync,
                    audio=dur,
                    rtf=sync / dur if dur else float("nan"),
                    nan=int(torch.isnan(audio).sum()),
                    inf=int(torch.isinf(audio).sum()),
                    peak=float(audio.abs().max()),
                    rms=float(audio.float().pow(2).mean().sqrt()),
                )
                rows.append(row)
                print(
                    f"  {name:7s} steps={steps:2d} {sync:6.2f}s audio={dur:6.2f}s "
                    f"rtf={row.rtf:4.2f} nan={row.nan} inf={row.inf} "
                    f"peak={row.peak:.3f} rms={row.rms:.4f}"
                )
        print(f"[{precision}] peak_vram={torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")

        del runtime
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return rows


def summarize(rows: list[Row]) -> None:
    if not rows:
        return
    print("\n=== speedup vs fp32 ===")
    base = {(r.case, r.steps): r for r in rows if r.precision == "fp32"}
    for row in rows:
        if row.precision == "fp32":
            continue
        ref = base.get((row.case, row.steps))
        if ref and row.sync > 0:
            print(
                f"  {row.case:7s} steps={row.steps:2d} "
                f"{row.precision}: {ref.sync / row.sync:.2f}x faster"
            )
    bad = [r for r in rows if r.nan or r.inf or r.peak > 1.0]
    print("\n=== 数值检查 ===")
    print("  全部正常" if not bad else f"  {len(bad)} 条异常: {[(r.precision, r.case) for r in bad]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/vol2/1000/Irodori-TTS/models/Irodori-TTS-v4.1-Small/model.safetensors",
    )
    parser.add_argument(
        "--codec", default="/vol2/1000/Irodori-TTS/models/dacvae/weights.pth"
    )
    parser.add_argument("--lora-dir", default="/vol2/1000/Irodori-TTS/models/LoRA/PJSK/宵崎奏/lora_adapter")
    parser.add_argument(
        "--ref-wav", default="/vol2/1000/Irodori-TTS/models/LoRA/PJSK/宵崎奏/emotion_refs/main.wav"
    )
    parser.add_argument("--precisions", default="fp16,bf16")
    parser.add_argument("--steps", default="40,24")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--quick", action="store_true", help="只跑短句和中句")
    args = parser.parse_args()

    rows = run(args)
    summarize(rows)


if __name__ == "__main__":
    main()
