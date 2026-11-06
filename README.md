# Irodori-TTS 本地补丁与部署配置

在 **NVIDIA CMP 40HX 8GB（Turing / sm_75）**、Debian 12 的家用服务器上部署
[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) v4.1-Small +
[Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server) 时，踩到的
4 个上游问题，以及对应的最小补丁、部署文件和精度基准脚本。

上游基线：

| 仓库 | commit |
|---|---|
| `Aratako/Irodori-TTS` | `7ed4374` (Add sponsors section) |
| `Aratako/Irodori-TTS-Server` | `ce06beb` (Add sponsor button) |

## 补丁清单

| # | 文件 | 上游问题 | 补丁效果 |
|---|---|---|---|
| 1 | `server/app.py` | 把含 `lora_adapter`/`ref_wav` 路径的合成消息直接写进 `X-Irodori-Messages` 响应头。HTTP 头只允许 latin-1，**路径含中文就整个请求 500** | 非 ASCII 用 `backslashreplace` 转义成 `\uXXXX` |
| 2 | `server/app.py` | 分块合成收尾是裸 `torch.cat`，块间不插静音。自然句间停顿 240–620 ms 被吃成 100–150 ms，长文本听感偏快 | 按上一块末尾标点插入 450 ms（句末）/ 180 ms（逗号）静音 |
| 3 | `lib/inference_runtime.py` | 每个新 adapter 都 `load_adapter` 且**从不 `delete_adapter`**，显存只增不减（约 257 MiB/人，8 GB 卡约 15–16 人后 OOM） | `OrderedDict` LRU + `model.delete_adapter` + `empty_cache`，容量由环境变量控制 |
| 4 | `lib/inference_runtime.py` | `resolve_runtime_dtype` 硬编码只接受 `fp32` / `bf16`。sm_75 没有原生 bf16，是软件仿真 | 放行 `fp16`（Turing 有真实 fp16 张量核心） |
| 5 | `server/pyproject.toml` | `irodori-tts` 是 git 依赖，`uv sync` 会用上游版本覆盖 #3/#4；pytorch 官方源实测仅 251 KB/s | 改为本地可编辑依赖 + SJTU pytorch 镜像 |

## fp16 实测（CMP 40HX 8GB，宵崎奏 LoRA，seed 固定）

| 文本 | 步数 | bf16 | fp16 | 加速 |
|---|---|---|---|---|
| 短（3.2 s） | 40 | 7.23 s | 6.49 s | 1.11× |
| 短（3.2 s） | 24 | 2.40 s | 1.43 s | 1.68× |
| 中（11.0 s） | 40 | 6.78 s | 2.30 s | **2.95×** |
| 中（11.0 s） | 24 | 4.41 s | 1.60 s | 2.76× |
| 长 196 字（27.7 s） | 40 | 18.76 s | 7.08 s | **2.65×** |
| 高能量情绪参考（angry） | 40 | 3.65 s | 2.25 s | 1.62× |

数值检查：全部样本 **NaN=0 / Inf=0**，峰值 0.71–0.90（无削波），
rms 与 bf16 差异 < 0.5%（如 0.1703 vs 0.1708）。

> 结论：fp16 在这张卡上可用且明显更快。但**音色/音质仍需人耳 AB**——
> 数值稳定不等于听感一致。切换只需改 `.env` 的 `IRODORI_MODEL_PRECISION`。

## 用法

```bash
# 1. 应用补丁（幂等，可反复执行；先 --check 看状态）
python scripts/apply_all.py --check
python scripts/apply_all.py --lib-dir /path/to/Irodori-TTS --server-dir /path/to/Irodori-TTS-Server

# 2. 重装依赖（editable 生效）
cd /path/to/Irodori-TTS-Server
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple uv sync --extra cu128

# 3. 部署配置
cp deploy/env.example /path/to/Irodori-TTS-Server/.env
sudo cp deploy/irodori-tts.service /etc/systemd/system/
sudo mkdir -p /etc/systemd/system/irodori-tts.service.d
sudo cp deploy/dropins/*.conf /etc/systemd/system/irodori-tts.service.d/
sudo systemctl daemon-reload && sudo systemctl enable --now irodori-tts

# 4. 精度基准
python scripts/bench_precision.py --precisions fp16,bf16 --steps 40,24
```

`scripts/apply_all.py` 的验证方式：从上游干净 checkout 应用后，生成的文件与线上
逐字节一致（`lib`、`app.py`、`pyproject.toml` 三个文件 `diff -q` 全部 IDENTICAL）。

## 可调环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `IRODORI_LORA_CACHE_SIZE` | 8 | 常驻 adapter 上限，0 = 不限制（退回上游行为）。每个约 257 MiB |
| `IRODORI_CHUNK_PAUSE_MS` | 450 | 分块边界（。！？）补回的静音 |
| `IRODORI_CHUNK_PAUSE_COMMA_MS` | 180 | 分块边界（、，）补回的静音 |
| `IRODORI_MODEL_PRECISION` | `bf16` | 可设 `fp16`（见上方实测） |

## 注意

- 补丁是**文本锚点替换**，上游改动这些函数后需要人工核对。锚点失配时
  `apply_all.py` 会明确报错并指出是哪个补丁，不会静默跳过。
- `pyproject.toml` 里的 `path = "../Irodori-TTS"` 是相对路径，两个仓库需并列存放。
- 补丁只覆盖推理路径，不影响训练。
- 更完整的部署记录见 [`docs/DEPLOY.md`](docs/DEPLOY.md)。

## 许可

补丁本身随上游，均为 MIT。
