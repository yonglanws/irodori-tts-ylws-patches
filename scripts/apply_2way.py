# -*- coding: utf-8 -*-
"""应用「2 路并发运行时池」补丁（在 apply_all.py 之后执行）。

改动内容：
  1. lib/inference_runtime.py
     - `_InferenceGuard`：每个 runtime 自己的锁 + 独立 CUDA stream
     - `_LazyContext`：把 LoRA 切换推迟到锁内（上游的 with 元组会提前执行）
     - `_LORA_LOAD_LOCK`：冷加载串行化，避免两路同时换 adapter 的显存双峰
     - `_sync_device` 改为只同步当前 stream，避免两路互相等待
  2. server/runtime.py
     - RuntimeManager 预加载 N 个 InferenceRuntime
     - 第 2..N 个 runtime 共享第一个的基座 / codec 参数存储（+约 105 MiB）
     - 轮询 + 优先空闲 runtime
  3. server/config.py
     - IRODORI_RUNTIME_POOL_SIZE（默认 1）
     - IRODORI_MAX_NUM_STEPS（默认 0 = 不限制）
  4. server/app.py
     - 按 IRODORI_MAX_NUM_STEPS 截断请求步数

用法：
    python scripts/apply_2way.py --check
    python scripts/apply_2way.py --lib-dir ... --server-dir ...
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEFAULT_LIB = "/vol2/1000/Irodori-TTS/Irodori-TTS"
DEFAULT_SERVER = "/vol2/1000/Irodori-TTS/Irodori-TTS-Server"

GUARD = '''class _InferenceGuard:
    """Per-runtime inference mutex plus a dedicated CUDA stream.

    Two InferenceRuntime instances in one process would otherwise share the
    legacy default CUDA stream, which serializes their kernels. Giving each
    runtime its own stream lets two personas generate at the same time; the
    lock still keeps a single runtime's own requests serialized (its LoRA
    adapter state is global).
    """

    def __init__(self, device: torch.device) -> None:
        self._lock = threading.Lock()
        self._stream: torch.cuda.Stream | None = None
        if device.type == "cuda":
            self._stream = torch.cuda.Stream(device=device)
        self._stream_ctx: Any = None

    def __enter__(self) -> "_InferenceGuard":
        self._lock.acquire()
        try:
            if self._stream is not None:
                self._stream_ctx = torch.cuda.stream(self._stream)
                self._stream_ctx.__enter__()
        except BaseException:
            self._lock.release()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        try:
            if self._stream is not None and self._stream_ctx is not None:
                # Finish this request before the caller (default stream)
                # concatenates and encodes the audio tensors.
                self._stream.synchronize()
                self._stream_ctx.__exit__(exc_type, exc, tb)
                self._stream_ctx = None
        finally:
            self._lock.release()
        return False


'''

LAZY = '''class _LazyContext:
    """Defer a context-manager factory until __enter__ runs.

    ``with (guard, self._prepare_lora_for_request(...))`` calls the adapter
    preparation while evaluating the with-items, i.e. *before* the guard is
    entered. Deferring it keeps the adapter switch inside the per-runtime
    guard so a concurrent request cannot switch the adapter under it.
    """

    def __init__(self, factory: Any) -> None:
        self._factory = factory
        self._ctx: Any = None

    def __enter__(self) -> Any:
        self._ctx = self._factory()
        return self._ctx.__enter__()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._ctx is None:
            return False
        return self._ctx.__exit__(exc_type, exc, tb)


_LORA_LOAD_LOCK = threading.Lock()


'''

RUNTIME_SOURCE = '''from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import torch

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    default_runtime_device,
    download_hf_checkpoint,
)

from .config import Settings

logger = logging.getLogger(__name__)


class RuntimeLoadTimeoutError(RuntimeError):
    pass


def _share_parameters(dst: torch.nn.Module, src: torch.nn.Module, label: str) -> None:
    """Point ``dst`` parameters at ``src`` storages.

    A second InferenceRuntime needs its own module tree (separate PEFT adapter
    state and inference guard) but not a second copy of the base weights.
    Sharing the storages keeps the extra runtime at roughly 0.1 GiB.
    """
    src_params = dict(src.named_parameters())
    shared = 0
    skipped = 0
    for name, param in dst.named_parameters():
        other = src_params.get(name)
        if other is None or other.shape != param.shape:
            skipped += 1
            continue
        param.data = other.data
        shared += 1
    logger.info("shared %s parameters: shared=%d skipped=%d", label, shared, skipped)


class RuntimeManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._pool_lock = threading.Lock()
        self._pool: list[InferenceRuntime] = []
        self._next = 0
        self._checkpoint_path: str | None = None

    def get(self) -> InferenceRuntime:
        if self._pool:
            return self._pick()

        timeout = float(self.settings.model_load_timeout)
        acquired = self._lock.acquire(timeout=max(timeout, 0.0))
        if not acquired:
            raise RuntimeLoadTimeoutError(
                f"Model is still loading. Retry after a moment. timeout={timeout:.1f}s"
            )

        try:
            if not self._pool:
                logger.info("loading runtime")
                t0 = time.perf_counter()
                self._checkpoint_path = self._resolve_checkpoint_path()
                logger.info("checkpoint resolved: %s", self._checkpoint_path)
                key = RuntimeKey(
                    checkpoint=self._checkpoint_path,
                    model_device=self._resolve_device(self.settings.model_device),
                    codec_repo=str(self.settings.codec_repo),
                    model_precision=str(self.settings.model_precision),
                    codec_device=self._resolve_device(self.settings.codec_device),
                    codec_precision=str(self.settings.codec_precision),
                    codec_deterministic_encode=bool(self.settings.codec_deterministic_encode),
                    codec_deterministic_decode=bool(self.settings.codec_deterministic_decode),
                    compile_model=bool(self.settings.compile_model),
                    compile_dynamic=bool(self.settings.compile_dynamic),
                )
                pool_size = max(1, int(self.settings.runtime_pool_size))
                logger.info("runtime pool size: %d", pool_size)

                primary = InferenceRuntime.from_key(key)
                _release_cuda_load_cache(primary)
                pool = [primary]
                for index in range(1, pool_size):
                    extra = InferenceRuntime.from_key(key)
                    _share_parameters(extra.model, primary.model, f"pool[{index}].model")
                    _share_parameters(extra.codec.model, primary.codec.model, f"pool[{index}].codec")
                    torch.cuda.empty_cache()
                    pool.append(extra)
                    logger.info("runtime pool[%d] ready", index)

                with self._pool_lock:
                    self._pool = pool
                    self._next = 0
                elapsed = time.perf_counter() - t0
                logger.info("runtime pool loaded in %.2fs (size=%d)", elapsed, len(pool))
            return self._pick()
        finally:
            self._lock.release()

    def _pick(self) -> InferenceRuntime:
        with self._pool_lock:
            runtime = self._pool[self._next % len(self._pool)]
            self._next += 1
        return runtime

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    @property
    def is_loaded(self) -> bool:
        return bool(self._pool)

    @property
    def is_loading(self) -> bool:
        return not self._pool and self._lock.locked()

    def _resolve_checkpoint_path(self) -> str:
        if self.settings.checkpoint is not None and str(self.settings.checkpoint).strip() != "":
            path = Path(str(self.settings.checkpoint)).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            return str(path)

        repo_id = str(self.settings.hf_checkpoint).strip()
        if repo_id == "":
            raise ValueError("Set IRODORI_CHECKPOINT or IRODORI_HF_CHECKPOINT.")
        logger.info("downloading checkpoint assets from hf://%s", repo_id)
        t0 = time.perf_counter()
        path = download_hf_checkpoint(repo_id)
        elapsed = time.perf_counter() - t0
        logger.info("checkpoint download/cache lookup completed in %.2fs", elapsed)
        return path

    @staticmethod
    def _resolve_device(value: str) -> str:
        raw = str(value).strip().lower()
        if raw in {"", "auto"}:
            return default_runtime_device()
        return str(value)


def _release_cuda_load_cache(runtime: InferenceRuntime) -> None:
    # Loading fp32 weights and then casting them to bf16 can leave large unused
    # allocations. Release them before inference,
    # rather than waiting for the post-synthesis cache release interval.
    devices = {
        getattr(runtime, "model_device", None),
        getattr(runtime, "codec_device", None),
    }
    for device in devices:
        if getattr(device, "type", None) != "cuda":
            continue
        with torch.cuda.device(device):
            before = torch.cuda.memory_reserved()
            torch.cuda.empty_cache()
            logger.info(
                "released CUDA load cache: device=%s reserved_before=%d "
                "reserved_after=%d allocated=%d",
                device,
                before,
                torch.cuda.memory_reserved(),
                torch.cuda.memory_allocated(),
            )
'''


def backup(path: Path) -> None:
    target = path.with_suffix(path.suffix + ".orig_2way")
    if not target.exists():
        shutil.copy2(path, target)


def patch(path: Path, old: str, new: str, marker: str, label: str, check: bool) -> str:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return f"  [already ] {label}"
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"[{label}] {path}: 锚点匹配 {count} 次（应为 1），请人工检查。")
    if not check:
        backup(path)
        path.write_text(text.replace(old, new), encoding="utf-8")
    return f"  [{'would' if check else 'applied':8s}] {label}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib-dir", default=DEFAULT_LIB)
    parser.add_argument("--server-dir", default=DEFAULT_SERVER)
    parser.add_argument("--check", action="store_true", help="只检查不写")
    args = parser.parse_args()

    lib_root = Path(args.lib_dir)
    server_root = Path(args.server_dir)
    lib_file = lib_root / "irodori_tts/inference_runtime.py"
    server_pkg = server_root / "src/irodori_openai_tts"
    config_file = server_pkg / "config.py"
    app_file = server_pkg / "app.py"
    runtime_file = server_pkg / "runtime.py"

    for path in (lib_file, config_file, app_file, runtime_file):
        if not path.is_file():
            raise SystemExit(f"文件不存在: {path}")

    results: list[str] = []

    results.append(
        patch(
            lib_file,
            "\ndef _measure_start(device: torch.device",
            "\n" + GUARD + "def _measure_start(device: torch.device",
            "_InferenceGuard",
            "lib: per-runtime inference guard",
            args.check,
        )
    )
    results.append(
        patch(
            lib_file,
            "        self._infer_lock = threading.Lock()\n",
            "        self._infer_ctx = _InferenceGuard(self.model_device)\n",
            "self._infer_ctx = _InferenceGuard",
            "lib: guard in __init__",
            args.check,
        )
    )
    results.append(
        patch(
            lib_file,
            "            self._infer_lock,\n",
            "            self._infer_ctx,\n",
            "            self._infer_ctx,\n",
            "lib: use guard in synthesize_batch",
            args.check,
        )
    )
    results.append(
        patch(
            config_file,
            "    max_concurrent_synthesis: int = 1\n",
            "    max_concurrent_synthesis: int = 1\n    runtime_pool_size: int = 1\n",
            "runtime_pool_size: int",
            "config: runtime_pool_size",
            args.check,
        )
    )

    runtime_text = runtime_file.read_text(encoding="utf-8")
    if "_share_parameters" in runtime_text and "runtime pool size" in runtime_text:
        results.append("  [already ] server: runtime pool")
    elif not args.check:
        backup(runtime_file)
        runtime_file.write_text(RUNTIME_SOURCE, encoding="utf-8")
        results.append("  [applied ] server: runtime pool")
    else:
        results.append("  [would   ] server: runtime pool（整文件替换）")

    results.append(
        patch(
            lib_file,
            "class _InferenceGuard:",
            LAZY + "class _InferenceGuard:",
            "_LORA_LOAD_LOCK",
            "lib: lazy LoRA context + load lock",
            args.check,
        )
    )
    results.append(
        patch(
            lib_file,
            "        self._stream_ctx: Any = None\n\n    def __enter__",
            "        self._stream_ctx: Any = None\n\n    @property\n    def busy(self) -> bool:\n        return self._lock.locked()\n\n    def __enter__",
            "def busy(self)",
            "lib: guard.busy",
            args.check,
        )
    )
    results.append(
        patch(
            lib_file,
            '    if device.type == "cuda":\n        torch.cuda.synchronize(device)\n',
            '    if device.type == "cuda":\n        # Sync only this thread\'s stream: a device-wide synchronize would wait\n        # for the other runtime\'s kernels and serialize concurrent personas.\n        torch.cuda.current_stream(device).synchronize()\n',
            "current_stream(device).synchronize()",
            "lib: per-stream sync",
            args.check,
        )
    )
    old_prep = '''        should_time = adapter_path is not None and str(adapter_path).strip() != ""
        t0 = _measure_start(self.model_device) if should_time else None
        try:
            return self._prepare_lora_for_request_inner(
                adapter_path,
                messages=messages,
                log_fn=log_fn,
            )
        finally:
            if t0 is not None:
                stage_sec = _measure_end(self.model_device, t0)
                stage_timings.append(("prepare_lora", stage_sec))
                log_fn(f"[runtime] prepare_lora: {stage_sec * 1000.0:.1f} ms")
'''
    new_prep = '''        def _run() -> Any:
            should_time = adapter_path is not None and str(adapter_path).strip() != ""
            t0 = _measure_start(self.model_device) if should_time else None
            try:
                with _LORA_LOAD_LOCK:
                    return self._prepare_lora_for_request_inner(
                        adapter_path,
                        messages=messages,
                        log_fn=log_fn,
                    )
            finally:
                if t0 is not None:
                    stage_sec = _measure_end(self.model_device, t0)
                    stage_timings.append(("prepare_lora", stage_sec))
                    log_fn(f"[runtime] prepare_lora: {stage_sec * 1000.0:.1f} ms")

        return _LazyContext(_run)
'''
    results.append(
        patch(
            lib_file,
            old_prep,
            new_prep,
            "return _LazyContext(_run)",
            "lib: lazy LoRA body",
            args.check,
        )
    )
    results.append(
        patch(
            config_file,
            "    runtime_pool_size: int = 1\n",
            "    runtime_pool_size: int = 1\n    max_num_steps: int = 0\n",
            "max_num_steps: int",
            "config: max_num_steps",
            args.check,
        )
    )
    results.append(
        patch(
            app_file,
            '''def _as_optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, name)
''',
            '''def _as_optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, name)


def _clamp_num_steps(value: int | None) -> int | None:
    limit = int(settings.max_num_steps)
    if value is not None and limit > 0:
        return min(value, limit)
    return value
''',
            "def _clamp_num_steps",
            "app: clamp helper",
            args.check,
        )
    )
    results.append(
        patch(
            app_file,
            '''        num_steps=_as_optional_int(
            _coalesce(opts.num_steps, _extra(payload, "num_steps"), settings.default_num_steps),
            "num_steps",
        ),
''',
            '''        num_steps=_clamp_num_steps(
            _as_optional_int(
                _coalesce(opts.num_steps, _extra(payload, "num_steps"), settings.default_num_steps),
                "num_steps",
            )
        ),
''',
            "num_steps=_clamp_num_steps(",
            "app: use clamp",
            args.check,
        )
    )
    results.append(
        patch(
            runtime_file,
            '''    def _pick(self) -> InferenceRuntime:
        with self._pool_lock:
            runtime = self._pool[self._next % len(self._pool)]
            self._next += 1
        return runtime
''',
            '''    def _pick(self) -> InferenceRuntime:
        with self._pool_lock:
            start = self._next
            self._next += 1
            pool = self._pool
        for offset in range(len(pool)):
            runtime = pool[(start + offset) % len(pool)]
            if not runtime._infer_ctx.busy:
                return runtime
        return pool[start % len(pool)]
''',
            "if not runtime._infer_ctx.busy",
            "runtime: prefer idle runtime",
            args.check,
        )
    )

    print("\n".join(results))
    print("\n完成。" + ("（--check，未写文件）" if args.check else "请重启 irodori-tts 服务。"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
