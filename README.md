# SASA 论文复现工程

这是一个从零搭建、可直接扩展的 **SASA（Self-Aware Safety Augmentation）** 复现框架。默认后端是
`llava-hf/llava-1.5-7b-hf`，实现了论文的核心两次前向投影、首 token 特征抽取、线性安全探针、
推理时安全门，以及论文中的主要机理分析。

> 重要：仓库提供的是完整可运行代码，而不是预先生成的论文结果。7B 权重和官方数据集需要你自行下载；
> 完整实验需 NVIDIA GPU。代码默认不会联网上传图片或结果。

## 1. 已实现的模块

- MM-SafetyBench、VLGuard、FigStep 官方目录到统一 JSONL 的数据适配器。
- LLaVA-v1.5-7B 加载、提示模板、原始基线生成。
- 论文公式 8 的 token-wise 安全表征投影：从融合层 `f=15` 投影并替换安全层 `s=13`。
- 投影后首 token logits / 最终隐藏态特征抽取。
- Logistic Regression 安全探针训练、保存、加载和阈值推理。
- SASA 推理门：危险输入直接拒绝，安全输入走原始 LLaVA 生成。
- 慢速逐 token 投影贪心解码（只作为诊断，不假装是论文明确给出的解码算法）。
- 注意力头消融、主奇异向量夹角、安全特异头筛选。
- 分层 logit-lens 可读性、隐藏态 t-SNE、图文模态对齐分析。
- ASR/良性拒绝率、Accuracy/F1/AUC 等评估。
- 不需模型权重的单元测试与静态检查。

## 2. 目录结构

```text
configs/llava15_7b.yaml            # 默认实验配置
examples/manifest.example.jsonl    # 统一数据格式示例
scripts/                            # 00–10：从环境检查到机理分析
src/sasa_repro/
  data/                             # 数据转换、分层采样
  model/                            # LLaVA 适配、投影、模块定位
  probe/                            # 线性安全探针
  evaluation/                       # 拒绝判断与指标
  analysis/                         # 头消融、分层、模态对齐
tests/                              # 快速单元测试
```

## 3. 环境

推荐 Linux、Python 3.10–3.12、CUDA 12.x，以及至少一张 24 GiB GPU。FP16 模型本身约需
14 GiB，运行时还需视觉编码器、激活和生成缓存；做全层激活分析时建议 40 GiB 以上或用多卡
`device_map: auto`。CPU 也能加载部分组件，但不适合完整 7B 实验。

```bash
cd sasa-reproduction
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/00_check_environment.py
```

模型接口按 Hugging Face 的 [LLaVA 文档](https://huggingface.co/docs/transformers/model_doc/llava)
和 [`llava-1.5-7b-hf` 模型卡](https://huggingface.co/llava-hf/llava-1.5-7b-hf) 编写。

## 4. 数据格式与准备

所有阶段都使用一行一个样本的 JSONL：

```json
{"id":"unique-id","image":"/absolute/path/image.jpg","prompt":"...","label":1,"dataset":"vlguard","split":"train","category":"unsafe","answer":"","metadata":{}}
```

`label=1` 表示不安全，`label=0` 表示安全。图片路径建议是绝对路径。

### MM-SafetyBench

下载[官方仓库](https://github.com/isXinLiu/MM-SafetyBench)并按其说明准备图片，然后：

```bash
python scripts/01_prepare_manifest.py mm-safetybench \
  --root /data/MM-SafetyBench \
  --variant SD_TYPO \
  --output data/mm_safetybench_sd_typo.jsonl
```

### VLGuard

下载[官方仓库](https://github.com/ys-zong/VLGuard)及测试图片：

```bash
python scripts/01_prepare_manifest.py vlguard \
  --metadata /data/VLGuard/test.json \
  --image-root /data/VLGuard \
  --subsets unsafes,safe_unsafes,safe_safes \
  --output data/vlguard.jsonl
```

其中 `unsafes`、`safe_unsafes` 标为不安全，`safe_safes` 标为安全。实际图片根目录按下载后的
JSON 中 `image` 字段调整。

### FigStep

下载[官方仓库](https://github.com/CryptoAILab/FigStep)的 SafeBench CSV 和排版攻击图片：

```bash
python scripts/01_prepare_manifest.py figstep \
  --csv /data/FigStep/SafeBench.csv \
  --image-root /data/FigStep/SafeBench \
  --output data/figstep.jsonl
```

合并或分层抽样：

```bash
python scripts/02_merge_manifests.py data/vlguard.jsonl data/figstep.jsonl \
  --output data/all.jsonl --validate-images

python scripts/02_sample_manifest.py --input data/all.jsonl \
  --output data/all_small.jsonl --per-group 20 --seed 42
```

训练探针必须同时含有 `label=0` 和 `label=1`。建议训练集和测试集按原数据集划分隔离，避免同图或
同问题泄漏。

## 5. 最短复现路径

### 5.1 原始 LLaVA 基线

```bash
python scripts/03_baseline_generate.py \
  --config configs/llava15_7b.yaml \
  --manifest data/test.jsonl \
  --output outputs/baseline.jsonl
```

### 5.2 抽取 SASA 特征

```bash
python scripts/04_extract_features.py \
  --config configs/llava15_7b.yaml \
  --manifest data/train.jsonl \
  --output outputs/train_sasa_logits.npz
```

默认使用投影后最后一个提示位置的 logits。消融对照可加 `--no-projection`；隐藏态对照可加
`--kind last_hidden`。

### 5.3 训练线性探针

```bash
python scripts/05_train_probe.py \
  --config configs/llava15_7b.yaml \
  --train-features outputs/train_sasa_logits.npz \
  --output outputs/sasa_probe.joblib
```

如已有独立验证集，先抽取其特征，再传 `--validation-features path/to/validation.npz`。

### 5.4 运行安全门并评估

```bash
python scripts/06_run_guard.py \
  --config configs/llava15_7b.yaml \
  --manifest data/test.jsonl \
  --probe outputs/sasa_probe.joblib \
  --output outputs/sasa_predictions.jsonl

python scripts/07_evaluate.py \
  --predictions outputs/sasa_predictions.jsonl \
  --output outputs/sasa_metrics.json
```

默认策略是：探针判为危险就返回固定拒绝语；判为安全才调用原始模型生成。这既能把检测和生成解耦，
也避免把两次前向投影错误地混入 Hugging Face 的 KV-cache 生成。若想研究投影隐藏态本身是否可生成，
将配置里的 `safe_generation_mode` 改成 `projected_greedy`；该模式每个 token 做两次完整前向，速度很慢。

## 6. 核心实现与论文对应

设融合层隐藏态为 `H_f`，较早安全层隐藏态为 `H_s`。第一遍前向缓存 `H_f`；第二遍前向在安全层
输出处注册 hook，并按隐藏维对每个 token 做：

```text
alpha       = <H_s, H_f> / (||H_f||² + epsilon)
H_projected = alpha * H_f
H_s         = H_projected
```

随后让替换后的隐藏态继续通过后续解码层，取提示末位置输出作为线性探针输入。`hidden_states[0]` 是
embedding，所以代码中第 `f` 个 decoder block 对应 `hidden_states[f + 1]`。

默认配置把论文层号解释为 **从 0 开始的 decoder block 索引**：`fused_layer=15`、
`safety_layer=13`。如果你确认论文实现使用从 1 开始编号，应改为 14 和 12，并把两套结果作为敏感性
实验同时报告。

`projection_mode=replace` 是论文公式对应项；`residual` 和 `interpolate` 仅用于消融。探针默认是无特征
归一化、类别平衡的 Logistic Regression。所有这些选择都集中在 `configs/llava15_7b.yaml`。

## 7. 机理分析

### 注意力头定位

先分别在安全任务样本与通用效用样本上计算头消融分数：

```bash
python scripts/08_locate_safety_heads.py --manifest data/safety_probe.jsonl \
  --output-csv outputs/safety_heads.csv --limit 64 --layers 0-31 --heads 0-31

python scripts/08_locate_safety_heads.py --manifest data/utility_probe.jsonl \
  --output-csv outputs/utility_heads.csv --limit 64 --layers 0-31 --heads 0-31

python scripts/08_locate_safety_heads.py \
  --safety-csv outputs/safety_heads.csv --utility-csv outputs/utility_heads.csv \
  --top-k 10 --comparison-output outputs/safety_specific_heads.json
```

默认忠实采用论文文字描述的“左奇异向量”。`--singular-side right` 提供特征空间向量的替代解释。
全量 `32×32` 头消融非常昂贵，CSV 支持断点续跑；建议先用少量样本和少量层/头确认流程。

### 分层可读性与 t-SNE

```bash
python scripts/09_analyze_layers.py --manifest data/analysis.jsonl \
  --output-dir outputs/layers --limit 100 --layers 0,5,10,13,15,20,25,31
```

### 图文模态对齐

```bash
python scripts/10_analyze_alignment.py --manifest data/analysis.jsonl \
  --output-dir outputs/alignment --limit 100
```

“纯图片”对自回归 LLaVA 并非定义良好的输入，因此这里用固定中性指令 `Describe the image.` 作为
图片主导控制组，并在报告中显式保留这一操作化定义。

## 8. 评估解释

- `attack_success_rate`：不安全样本中未触发拒绝关键词的比例，越低越好。
- `benign_refusal_rate`：安全样本中触发拒绝的比例，越低越好。
- 探针报告 Accuracy、F1、ROC-AUC、不安全召回率和假阳性率。
- `evaluation/keywords.py` 收录论文列出的拒绝关键词，但关键词 ASR 只是可复现的近似指标；若论文主表
  使用人工或 LLM judge，应另行产生逐样本 judge 标签后再比较，不能把关键词结果冒充主表结果。

建议至少报告三组：原始 LLaVA、无投影线性探针、SASA 投影线性探针；同时给出均值、随机种子、
样本数、数据版本、阈值和失败样本数。

## 9. 测试与常见问题

```bash
make check
make test
```

- OOM：把 `max_new_tokens` 调小，使用多卡 `device_map: auto`，或将 dtype 改为 `bfloat16`（硬件支持时）。
- 找不到 decoder 层：检查 Transformers 版本；模块路径集中在 `model/introspection.py`，便于适配新版本。
- 图片 token 数不匹配：不要手工删除 `<image>`，并保持模型与 processor 来自同一 checkpoint。
- 结果与论文不同：先核对层号基准、数据版本/子集、提示模板、首 token 定义、随机种子和判拒标准。
- 安全研究应在隔离环境、公开基准与授权模型上进行，不要把测试流程用于现实伤害。

## 10. 已知边界

论文未附代码时，一些工程细节无法仅靠正文唯一确定。本项目不会隐藏这些分歧：层号基准、探针输入
是否归一化、奇异向量取左还是右、图片控制提示、以及投影是否参与逐 token 解码都可配置或有独立诊断
入口。复现实验应冻结配置文件，并在论文/报告中逐项披露。

MIT License。模型和数据仍分别受其原始许可证与使用条款约束。
