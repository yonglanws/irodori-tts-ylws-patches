# Irodori-TTS 本地补丁与部署配置

在 **NVIDIA CMP 40HX 8GB（Turing / sm_75）**、Debian 12 的家用服务器上部署
[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) v4.1-Small +
[Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server) 时，踩到的
上游问题、对应的最小补丁，以及 2 路并发、fp16、LoRA 显存管理的实测数据。

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
| 6 | `server/runtime.py` + `config.py` | 全进程只有一个 `InferenceRuntime`，`MAX_CONCURRENT_SYNTHESIS>1` 也会被 `_infer_lock` 串行化 | 运行时池：预加载 N 个 runtime，**第 2..N 个共享第一个的基座/codec 参数存储**（+约 105 MiB），轮询并优先空闲 runtime |
| 7 | `lib/inference_runtime.py` | 单进程内两个 runtime 共用 PyTorch 默认 CUDA stream，kernel 仍串行；且 `with (lock, prepare_lora())` 会在取锁**之前**执行 LoRA 切换，同一 runtime 上可能把 adapter 换掉 | `_InferenceGuard`（每 runtime 一把锁 + 独立 CUDA stream）；`_LazyContext` 把 LoRA 切换推迟到锁内；`_sync_device` 改为只同步当前 stream |
| 8 | `server/app.py` + `config.py` | 客户端固定发 40 步，流量大时 GPU 长期 99% | `IRODORI_MAX_NUM_STEPS` 在服务端截断步数（0 = 不限制） |

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

> 结论：fp16 在这张卡上可用且明显更快。2026-11-06 起本机生产已切换为 fp16。

## 2 路并发实测

**显存代价**：第 2 个 runtime 只多 **105 MiB**（`memory_allocated` 2014 → 2119 MiB），
因为基座的 714 个参数和 codec 的 317 个参数都指向同一份存储。

**离线（adapter 已预热，5.64 s 音频，40 步）**：

| 场景 | 两个请求总耗时 | 相对串行 |
|---|---|---|
| 串行（同一 runtime） | 5.10 s | 1.00× |
| 并发·默认 stream | 4.05 s | 1.26× |
| 并发·各自 CUDA stream | 3.93 s | **1.30×** |
| 并发·同一 adapter | 3.76 s | 1.36× |

**步数（8.88 s 音频，`cfg_guidance_mode=independent`）**：

| 步数 | 墙钟 | 相对 40 步 |
|---|---|---|
| 40 | 2.41 s | 1.00× |
| 32 | 2.05 s | 1.18× |
| 24 | 1.66 s | **1.45×** |
| 16 | 1.27 s | 1.90× |

**线上（24 步 + 2 路 + LRU 2/runtime）**：5 分钟窗口 46 条请求，
`elapsed` p50 **11.6 s** / p90 **20.3 s**，设备峰值显存 **7515 MiB**（含 Chrome 约 530 MiB，
TTS 本体约 7.0 GiB），0 条 ERROR/Traceback。

### 什么时候 2 并发有用

- GPU 没打满时（短文本、请求稀疏）：两路可以真正重叠，约 **1.3×** 吞吐。
- GPU 已经 90–99% 时（长文本连续请求）：两路只是把每个请求各拖慢一倍，**总吞吐不变**。
  这台机器晚高峰就是后一种情况，所以真正的容量杠杆是**降步数**（40→24 约 1.45×）
  或换 MeanFlow 4 步模型，而不是并发数。
- 并发的主要收益是把换 adapter（冷加载约 2.2 s）、参考音频编码、WAV 编码这些
  **非 GPU 阶段**和另一路的 GPU 阶段叠起来。

## 功耗墙

CMP 40HX 默认 184 W，但这台机器的当前限制是 150 W，满载时正好卡在 150 W。
恢复到默认值可以多出一点频率余量：

```bash
sudo nvidia-smi -pl 184
```

（可随时 `sudo nvidia-smi -pl 150` 改回；温度 74–80 °C，风扇 90%。）

## 用法

```bash
# 1. 应用补丁（幂等，可反复执行；先 --check 看状态）
python scripts/apply_all.py --check
python scripts/apply_all.py   --lib-dir /path/to/Irodori-TTS --server-dir /path/to/Irodori-TTS-Server

# 2. 2 路并发运行时池（在 apply_all.py 之后）
python scripts/apply_2way.py --check
python scripts/apply_2way.py  --lib-dir /path/to/Irodori-TTS --server-dir /path/to/Irodori-TTS-Server

# 3. 重装依赖（editable 生效）
cd /path/to/Irodori-TTS-Server
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple uv sync --extra cu128

# 4. 部署配置
cp deploy/env.example /path/to/Irodori-TTS-Server/.env
sudo cp deploy/irodori-tts.service /etc/systemd/system/
sudo mkdir -p /etc/systemd/system/irodori-tts.service.d
sudo cp deploy/dropins/*.conf /etc/systemd/system/irodori-tts.service.d/
sudo systemctl daemon-reload && sudo systemctl enable --now irodori-tts

# 5. 精度基准
python scripts/bench_precision.py --precisions fp16,bf16 --steps 40,24
```

`scripts/apply_all.py` + `scripts/apply_2way.py` 的验证方式：从上游干净 checkout
应用后，生成的文件与线上逐字节一致（`lib`、`app.py`、`config.py`、`runtime.py`
四个文件 `diff -q` 全部 IDENTICAL）。

## 可调环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `IRODORI_LORA_CACHE_SIZE` | 8 | **每个 runtime** 常驻 adapter 上限，0 = 不限制。2 路并发时建议 2–3 |
| `IRODORI_RUNTIME_POOL_SIZE` | 1 | 运行时池大小；2 = 2 路并发（基座权重共享） |
| `IRODORI_MAX_CONCURRENT_SYNTHESIS` | 1 | API 层并发槽位；要与 `RUNTIME_POOL_SIZE` 一致 |
| `IRODORI_MAX_NUM_STEPS` | 0 | 服务端强制步数上限（0 = 不限制）。线上设 24 |
| `IRODORI_CHUNK_PAUSE_MS` | 450 | 分块边界（。！？）补回的静音 |
| `IRODORI_CHUNK_PAUSE_COMMA_MS` | 180 | 分块边界（、，）补回的静音 |
| `IRODORI_MODEL_PRECISION` | `fp16` | 本机生产已切 fp16；回退改 `bf16` |
| `IRODORI_EMPTY_CACHE_INTERVAL` | 10 | 每 N 次合成释放一次 allocator 缓存；2 路并发建议 2 |

## 注意

- 补丁是**文本锚点替换**，上游改动这些函数后需要人工核对。锚点失配时
  脚本会明确报错并指出是哪个补丁，不会静默跳过。
- `apply_2way.py` 对 `runtime.py` 是**整文件替换**（先备份 `.orig_2way`），
  上游若改了 `RuntimeManager` 需要重新对照。
- `pyproject.toml` 里的 `path = "../Irodori-TTS"` 是相对路径，两个仓库需并列存放。
- 补丁只覆盖推理路径，不影响训练。
- 更完整的部署记录见 [`docs/DEPLOY.md`](docs/DEPLOY.md)。

## 许可

补丁本身随上游，均为 MIT。
