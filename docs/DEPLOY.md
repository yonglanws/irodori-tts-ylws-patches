# Irodori-TTS 本地部署（ylws / CMP 40HX 8GB）

部署日期：2026-11-04

## 目录结构

```
/vol2/1000/Irodori-TTS/
├── Irodori-TTS/                 # 上游库（git, Aratako/Irodori-TTS）
├── Irodori-TTS-Server/          # OpenAI 兼容服务（git, Aratako/Irodori-TTS-Server）
│   ├── .venv/                   # uv 建的 venv（11GB）
│   ├── .env                     # 本地配置（本地模型 / bf16 / 8088）
│   ├── pyproject.toml.orig      # 原始索引配置备份
│   └── uv.lock.orig             # 原始锁文件备份（含官方 pytorch URL）
├── models/
│   ├── Irodori-TTS-v4.1-Small/
│   │   ├── model.safetensors    # 3,064,295,596 B（与 ModelScope 一致）
│   │   └── tokenizer/           # tokenizer.json + config
│   ├── dacvae/weights.pth       # 429,620,065 B（Semantic-DACVAE-Japanese-32dim）
│   └── LoRA/PJSK/<中文人格ID>/  # 角色 LoRA
│       ├── lora_adapter/        # PEFT adapter（r64 alpha128, ~338MB）
│       ├── emotion_refs/        # main/calm/happy/angry/sad/shy/surprised.wav
│       └── json/                # jobs/eval/summary
├── DEPLOY.md                    # 本文档
├── server.log / uv_sync.log
```

### LoRA 目录命名规则（重要）

插件 `gen_irodori_config.py` 用**目录名直接当人格 ID**（`persona = char_dir.name`），
所以必须用中文名（如 `宵崎奏`、`和泉朋花`），romaji 目录（kanade/tomoka/…）插件认不出来。

已装 38 个角色（各 338 MB adapter + 8 个参考 wav）：
- PJSK 19 人：星乃一歌、天马咲希、望月穗波、日野森志步、花里实乃理、桐谷遥、桃井爱莉、
  日野森雫、小豆泽心羽、白石杏、东云彰人、青柳冬弥、天马司、凤笑梦、草薙宁宁、神代类、
  宵崎奏、东云绘名、晓山瑞希
- 其他 19 人：和泉朋花、泽海奏多、琴平凪、梅里千樱梨、矢仓蓬咲、马桥心玖、滨崎茉幌、
  藤都子、四宫宁月、伊泽枣、宫永野乃花、须贺蕾叶、峰月律、千石由乃、仲町阿拉蕾、
  汐见萤、薇欧拉、要乐奈、三角初华

注：交付包里的 romaji 目录名与人 AstrBot 人格 ID 不一定同名，务必按人格表核对
（例：chieri -> 梅里千樱梨、rana -> 要乐奈、uika -> 三角初华）。


