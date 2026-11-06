# -*- coding: utf-8 -*-
"""一键（可重复执行）应用本仓库对 Irodori-TTS / Irodori-TTS-Server 的全部本地补丁。

每个补丁都是"找到唯一锚点 → 替换"，重复执行不会叠加。
上游改动导致锚点失配时会明确报错并指出是哪个补丁，不会静默跳过。

用法：
    python scripts/apply_all.py                 # 应用到默认路径
    python scripts/apply_all.py --check         # 只检查，不写文件
    python scripts/apply_all.py --lib-dir X --server-dir Y

默认路径：
    lib    /vol2/1000/Irodori-TTS/Irodori-TTS
    server /vol2/1000/Irodori-TTS/Irodori-TTS-Server
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LIB = Path("/vol2/1000/Irodori-TTS/Irodori-TTS")
DEFAULT_SERVER = Path("/vol2/1000/Irodori-TTS/Irodori-TTS-Server")

LIB_FILE = "irodori_tts/inference_runtime.py"
SERVER_APP = "src/irodori_openai_tts/app.py"
SERVER_PYPROJECT = "pyproject.toml"

# ---------------------------------------------------------------- server: header

HEADER_OLD = (
    '        headers["X-Irodori-Messages"] = " | ".join(result.messages)[0:4096]'
)
HEADER_NEW = '''        _messages = " | ".join(result.messages)[0:4096]
        # HTTP headers are latin-1 only; escape non-ASCII (e.g. Chinese paths)
        # instead of failing the whole request with UnicodeEncodeError.
        headers["X-Irodori-Messages"] = _messages.encode(
            "latin-1", "backslashreplace"
        ).decode("latin-1")'''

# ------------------------------------------------------------- server: chunk pause

CHUNK_OLD = "    audio = torch.cat([_audio_as_channels_first(result.audio) for result in results], dim=-1)\n"
CHUNK_NEW = '''    # --- local patch: restore the sentence pause lost at chunk boundaries ---
    # Upstream concatenates chunks with torch.cat, so the pause after 。/！/？ is
    # never spoken: a boundary keeps only the model's own trailing silence
    # (~100-150 ms, vs 240-620 ms for the same break inside one chunk).
    try:
        pause_ms = int(os.environ.get("IRODORI_CHUNK_PAUSE_MS", "450"))
        comma_pause_ms = int(os.environ.get("IRODORI_CHUNK_PAUSE_COMMA_MS", "180"))
    except ValueError:
        pause_ms, comma_pause_ms = 450, 180

    audio_parts: list[torch.Tensor] = []
    for index, result in enumerate(results):
        part = _audio_as_channels_first(result.audio)
        if index > 0 and pause_ms > 0:
            prev_text = chunks[index - 1].rstrip()
            last_char = prev_text[-1] if prev_text else ""
            extra_ms = comma_pause_ms if last_char in "、，," else pause_ms
            pad = int(sample_rate * extra_ms / 1000) if extra_ms > 0 else 0
            if pad > 0:
                audio_parts.append(
                    torch.zeros((part.shape[0], pad), dtype=part.dtype, device=part.device)
                )
        audio_parts.append(part)
    audio = torch.cat(audio_parts, dim=-1)
'''

# ------------------------------------------------------------------- lib: LoRA LRU

LRU_OLD = '''        adapter_name = self._lora_adapter_names.get(resolved_path)
        if adapter_name is None:
            adapter_name = self._adapter_name_for_path(resolved_path)
'''
LRU_NEW = '''        # --- local patch: LRU eviction of resident LoRA adapters ---
        # Upstream never releases an adapter once loaded, so VRAM grows with the
        # number of distinct personas served. Keep at most IRODORI_LORA_CACHE_SIZE
        # resident (0 disables the limit) and evict the least recently used first.
        lru = getattr(self, "_lora_lru", None)
        if lru is None:
            lru = self._lora_lru = OrderedDict()
        try:
            capacity = int(os.environ.get("IRODORI_LORA_CACHE_SIZE", "8"))
        except ValueError:
            capacity = 8

        adapter_name = self._lora_adapter_names.get(resolved_path)
        if adapter_name is None:
            if capacity > 0:
                while len(lru) >= capacity:
                    old_path, _stamp = lru.popitem(last=False)
                    old_name = self._lora_adapter_names.pop(old_path, None)
                    if old_name is None:
                        continue
                    try:
                        self.model.delete_adapter(old_name)
                        msg = f"info: evicted LoRA adapter (LRU): {old_path}"
                    except Exception as exc:  # noqa: BLE001
                        msg = f"warn: failed to evict LoRA adapter {old_name}: {exc}"
                    messages.append(msg)
                    log_fn(msg)
                    if str(self.model_device).startswith("cuda"):
                        torch.cuda.empty_cache()
            adapter_name = self._adapter_name_for_path(resolved_path)
'''

LRU_TAIL_OLD = '''        self._lora_adapter_names[resolved_path] = adapter_name
        self.model = _move_inference_module(
'''
LRU_TAIL_NEW = '''        self._lora_adapter_names[resolved_path] = adapter_name
        lru[resolved_path] = time.monotonic()
        lru.move_to_end(resolved_path)
        self.model = _move_inference_module(
'''

# -------------------------------------------------------------------- lib: fp16

FP16_OLD = '''    raise ValueError(f"Unsupported precision={precision!r}. Expected one of: fp32, bf16.")
'''
FP16_NEW = '''    # local patch: allow fp16 (Turing/sm_75 has real fp16 tensor cores; bf16 is
    # emulated there). Verified on CMP 40HX: no NaN/Inf, rms matches bf16.
    if mode in ("fp16", "float16", "half"):
        if device.type not in ("cuda", "xpu"):
            raise ValueError("precision='fp16' currently requires CUDA or XPU device.")
        return torch.float16
    raise ValueError(
        f"Unsupported precision={precision!r}. Expected one of: fp32, bf16, fp16."
    )
'''

# --------------------------------------------------------- server: editable dep

DEP_OLD = '    "irodori-tts @ git+https://github.com/Aratako/Irodori-TTS.git",\n'
DEP_NEW = '    "irodori-tts",\n'
SOURCES_HEADER = "[tool.uv.sources]\n"
SOURCES_NEW = 'irodori-tts = { path = "../Irodori-TTS", editable = true }\n'

MIRRORS = [
    (
        "pytorch-cpu",
        'url = "https://download.pytorch.org/whl/cpu"',
        'url = "https://mirror.sjtu.edu.cn/pytorch-wheels/cpu"',
    ),
    (
        "pytorch-cu128",
        'url = "https://download.pytorch.org/whl/cu128"',
        'url = "https://mirror.sjtu.edu.cn/pytorch-wheels/cu128"',
    ),
]


@dataclass
class Patch:
    name: str
    target: str          # "lib" | "server"
    marker: str          # already-applied 标记
    old: str
    new: str
    os_anchor: str = ""  # 需要 import os 时插入到该行之后


PATCHES = [
    Patch("server: latin-1 response header", "server", "backslashreplace", HEADER_OLD, HEADER_NEW),
    Patch(
        "server: chunk-boundary pause",
        "server",
        "restore the sentence pause",
        CHUNK_OLD,
        CHUNK_NEW,
        os_anchor="\nimport logging\n",
    ),
    Patch(
        "lib: LoRA adapter LRU eviction",
        "lib",
        "LRU eviction of resident LoRA",
        LRU_OLD,
        LRU_NEW,
        os_anchor="\nimport math\n",
    ),
    Patch("lib: LoRA adapter LRU bookkeeping", "lib", "lru[resolved_path] = time.monotonic()", LRU_TAIL_OLD, LRU_TAIL_NEW),
    Patch("lib: allow fp16 precision", "lib", "local patch: allow fp16", FP16_OLD, FP16_NEW),
]


def apply_text(src: str, patch: Patch, *, path: Path) -> tuple[str, str]:
    """返回 (新内容, 状态)；状态为 applied / already。锚点失配抛 SystemExit。"""
    if patch.marker in src:
        return src, "already"
    count = src.count(patch.old)
    if count != 1:
        raise SystemExit(
            f"[{patch.name}] {path}: 锚点匹配 {count} 次（应为 1）。"
            "上游可能改动过这段代码，请人工检查。"
        )
    if patch.os_anchor and "\nimport os\n" not in src:
        if patch.os_anchor not in src:
            raise SystemExit(f"[{patch.name}] {path}: 找不到 import 锚点 {patch.os_anchor!r}")
        src = src.replace(patch.os_anchor, patch.os_anchor + "import os\n", 1)
    return src.replace(patch.old, patch.new), "applied"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib-dir", default=str(DEFAULT_LIB))
    parser.add_argument("--server-dir", default=str(DEFAULT_SERVER))
    parser.add_argument("--check", action="store_true", help="只检查不写")
    args = parser.parse_args()

    roots = {"lib": Path(args.lib_dir), "server": Path(args.server_dir)}
    for key, root in roots.items():
        if not root.is_dir():
            raise SystemExit(f"{key} 目录不存在: {root}")

    targets = {"lib": roots["lib"] / LIB_FILE, "server": roots["server"] / SERVER_APP}

    # 先按文件聚合补丁，避免多次读写
    pending: dict[Path, list[Patch]] = {}
    for patch in PATCHES:
        pending.setdefault(targets[patch.target], []).append(patch)

    results: list[str] = []
    for path, patches in pending.items():
        src = path.read_text(encoding="utf-8")
        changed = False
        for patch in patches:
            src, status = apply_text(src, patch, path=path)
            results.append(f"  [{status:8s}] {patch.name}")
            changed = changed or status == "applied"
        if changed and not args.check:
            backup = path.with_suffix(path.suffix + ".bak_apply_all")
            if not backup.exists():
                backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            path.write_text(src, encoding="utf-8")

    # pyproject: editable 依赖 + 国内镜像
    pyproject = roots["server"] / SERVER_PYPROJECT
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8")
        pyproject_changed = False
        if 'irodori-tts = { path = "../Irodori-TTS"' in text:
            results.append("  [already ] server: editable irodori-tts dependency")
        elif text.count(DEP_OLD) == 1 and text.count(SOURCES_HEADER) == 1:
            text = text.replace(DEP_OLD, DEP_NEW).replace(
                SOURCES_HEADER, SOURCES_HEADER + SOURCES_NEW, 1
            )
            pyproject_changed = True
            results.append("  [applied ] server: editable irodori-tts dependency")
        else:
            results.append("  [SKIP    ] server: editable irodori-tts dependency（锚点失配，请人工检查）")

        # pyproject: pytorch 索引改国内镜像（官方源实测 251 KB/s，SJTU 78 MB/s）
        for name, old_url, new_url in MIRRORS:
            if new_url in text:
                results.append(f"  [already ] server: {name} mirror")
            elif old_url in text:
                text = text.replace(old_url, new_url, 1)
                pyproject_changed = True
                results.append(f"  [applied ] server: {name} mirror")
            else:
                results.append(f"  [SKIP    ] server: {name} mirror（未找到上游 URL）")

        if pyproject_changed and not args.check:
            backup = pyproject.with_name("pyproject.toml.bak_apply_all")
            if not backup.exists():
                backup.write_text(pyproject.read_text(encoding="utf-8"), encoding="utf-8")
            pyproject.write_text(text, encoding="utf-8")

    print("\n".join(results))
    print("\n完成。" + ("（--check，未写文件）" if args.check else "请执行 uv sync 后重启服务。"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
