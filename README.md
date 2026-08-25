# SASA Reproduction Project

This repository provides a **SASA (Self-Aware Safety Augmentation: Leveraging Internal Semantic Understanding to Enhance Safety in Vision-Language Models)** reproduction framework built from scratch and designed for direct extension. The default backend is`llava-hf/llava-1.5-7b-hf`. It implements the paper's core two-pass projection, first-token feature extraction, linear safety probe,
inference-time safety gate, and major mechanistic analyses.

> **Important:** This repository provides complete runnable code rather than pre-generated paper results. You must download the 7B model weights and official datasets yourself.
> Full experiments require an NVIDIA GPU. By default, the code does not upload images or results to the Internet.

![Overall workflow of Self-Aware Safety Augmentation (SASA)](assets/sasa_workflow.png)

## 1. Implemented Modules

- Data adapters that convert the official MM-SafetyBench, VLGuard, and FigStep directory structures into a unified JSONL format.
- LLaVA-v1.5-7B loading, prompt templates, and original baseline generation.
- Token-wise safety representation projection from Equation 8 of the paper: project from the fusion layer `f=15` and replace the representation at the safety layer `s=13`.
- Extraction of first-token logits and final hidden-state features after projection.
- Training, saving, loading, and threshold-based inference for a Logistic Regression safety probe.
- SASA inference gate: unsafe inputs are rejected directly, while safe inputs are passed to the original LLaVA generation pipeline.
- Slow token-by-token projected greedy decoding for diagnostic purposes only; it is not presented as a decoding algorithm explicitly specified by the paper.
- Attention-head ablation, principal singular-vector angle analysis, and safety-specific head selection.
- Layer-wise logit-lens interpretability, hidden-state t-SNE, and image-text modality alignment analysis.
- Evaluation of attack success rate (ASR), benign refusal rate, Accuracy, F1, AUC, and related metrics.
- Unit tests and static checks that do not require model weights.

## 2. Directory Structure

```text
configs/llava15_7b.yaml            # Default experiment configuration
examples/manifest.example.jsonl    # Example of the unified data format
scripts/                            # 00–10: from environment checks to mechanistic analyses
src/sasa_repro/
  data/                             # Data conversion and stratified sampling
  model/                            # LLaVA adaptation, projection, and module discovery
  probe/                            # Linear safety probe
  evaluation/                       # Refusal detection and evaluation metrics
  analysis/                         # Head ablation, layer-wise analysis, and modality alignment
tests/                              # Fast unit tests
```

## 3. Environment

Linux, Python 3.10–3.12, CUDA 12.x, and at least one 24 GiB GPU are recommended. The FP16 model itself requires approximately
14 GiB. Additional memory is needed at runtime for the vision encoder, activations, and generation cache. For full-layer activation analysis, a GPU with at least 40 GiB is recommended; alternatively, use multiple GPUs with
`device_map: auto`. Some components can be loaded on CPU, but CPU execution is unsuitable for complete 7B experiments.

```bash
cd sasa-reproduction
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/00_check_environment.py
```

The model interface follows the Hugging Face [LLaVA documentation](https://huggingface.co/docs/transformers/model_doc/llava)
and the [`llava-1.5-7b-hf` model card](https://huggingface.co/llava-hf/llava-1.5-7b-hf).

## 4. Data Format and Preparation

All stages use JSONL files with one sample per line:

```json
{"id":"unique-id","image":"/absolute/path/image.jpg","prompt":"...","label":1,"dataset":"vlguard","split":"train","category":"unsafe","answer":"","metadata":{}}
```

`label=1` denotes an unsafe sample, while `label=0` denotes a safe sample. Absolute image paths are recommended.

### MM-SafetyBench

Download the [official repository](https://github.com/isXinLiu/MM-SafetyBench), prepare the images according to its instructions, and then run:

```bash
python scripts/01_prepare_manifest.py mm-safetybench \
  --root /data/MM-SafetyBench \
  --variant SD_TYPO \
  --output data/mm_safetybench_sd_typo.jsonl
```

### VLGuard

Download the [official repository](https://github.com/ys-zong/VLGuard) and its test images:

```bash
python scripts/01_prepare_manifest.py vlguard \
  --metadata /data/VLGuard/test.json \
  --image-root /data/VLGuard \
  --subsets unsafes,safe_unsafes,safe_safes \
  --output data/vlguard.jsonl
```

The `unsafes` and `safe_unsafes` subsets are labeled unsafe, while `safe_safes` is labeled safe. Adjust the image root to match the
`image` field in the downloaded JSON file.

### FigStep

Download the SafeBench CSV and typographic attack images from the [official repository](https://github.com/CryptoAILab/FigStep):

```bash
python scripts/01_prepare_manifest.py figstep \
  --csv /data/FigStep/SafeBench.csv \
  --image-root /data/FigStep/SafeBench \
  --output data/figstep.jsonl
```

Merge manifests or perform stratified sampling:

```bash
python scripts/02_merge_manifests.py data/vlguard.jsonl data/figstep.jsonl \
  --output data/all.jsonl --validate-images

python scripts/02_sample_manifest.py --input data/all.jsonl \
  --output data/all_small.jsonl --per-group 20 --seed 42
```

Probe training requires both `label=0` and `label=1` samples. The training and test sets should follow the original dataset splits and remain isolated to prevent leakage from duplicated images or questions.

## 5. Minimal Reproduction Pipeline

### 5.1 Original LLaVA Baseline

```bash
python scripts/03_baseline_generate.py \
  --config configs/llava15_7b.yaml \
  --manifest data/test.jsonl \
  --output outputs/baseline.jsonl
```

### 5.2 Extract SASA Features

```bash
python scripts/04_extract_features.py \
  --config configs/llava15_7b.yaml \
  --manifest data/train.jsonl \
  --output outputs/train_sasa_logits.npz
```

By default, the script uses the logits at the final prompt position after projection. Add `--no-projection` for an ablation baseline, or use
`--kind last_hidden` for a hidden-state baseline.

### 5.3 Train the Linear Probe

```bash
python scripts/05_train_probe.py \
  --config configs/llava15_7b.yaml \
  --train-features outputs/train_sasa_logits.npz \
  --output outputs/sasa_probe.joblib
```

If an independent validation set is available, extract its features first and then pass `--validation-features path/to/validation.npz`.

### 5.4 Run the Safety Gate and Evaluate

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

The default policy returns a fixed refusal response when the probe classifies an input as unsafe. Only inputs classified as safe are passed to the original model for generation. This design decouples detection from generation
and prevents the two-pass projection from being incorrectly mixed with Hugging Face KV-cache generation. To investigate whether projected hidden states can themselves support generation,
set `safe_generation_mode` in the configuration file to `projected_greedy`. This mode performs two complete forward passes for every token and is therefore very slow.

## 6. Mapping the Core Implementation to the Paper

Let the hidden state at the fusion layer be `H_f` and the hidden state at the earlier safety layer be `H_s`. The first forward pass caches `H_f`. During the second forward pass,
a hook is registered at the output of the safety layer, and the following projection is applied to every token along the hidden dimension:

```text
alpha       = <H_s, H_f> / (||H_f||² + epsilon)
H_projected = alpha * H_f
H_s         = H_projected
```

The replaced hidden states then continue through the subsequent decoder layers. The output at the final prompt position is used as the input to the linear probe. Because `hidden_states[0]` is the
embedding output, decoder block `f` corresponds to `hidden_states[f + 1]` in the implementation.

The default configuration interprets the layer numbers in the paper as **zero-based decoder-block indices**: `fused_layer=15` and
`safety_layer=13`. If you confirm that the paper's implementation uses one-based numbering, change these values to 14 and 12, respectively, and report both settings as a sensitivity experiment.

`projection_mode=replace` corresponds to the equation in the paper. The `residual` and `interpolate` modes are provided only for ablation studies. By default, the probe is a class-balanced Logistic Regression model without feature
normalization. All these choices are centralized in `configs/llava15_7b.yaml`.

## 7. Mechanistic Analysis

### Attention-Head Localization

First, compute head-ablation scores separately on safety-task samples and general-utility samples:

```bash
python scripts/08_locate_safety_heads.py --manifest data/safety_probe.jsonl \
  --output-csv outputs/safety_heads.csv --limit 64 --layers 0-31 --heads 0-31

python scripts/08_locate_safety_heads.py --manifest data/utility_probe.jsonl \
  --output-csv outputs/utility_heads.csv --limit 64 --layers 0-31 --heads 0-31

python scripts/08_locate_safety_heads.py \
  --safety-csv outputs/safety_heads.csv --utility-csv outputs/utility_heads.csv \
  --top-k 10 --comparison-output outputs/safety_specific_heads.json
```

By default, the implementation follows the paper's description and uses the “left singular vector.” The `--singular-side right` option provides an alternative interpretation based on a feature-space vector.
Exhaustive ablation over all `32×32` heads is very expensive. CSV output supports resuming interrupted runs; begin with a small number of samples and a limited set of layers and heads to verify the pipeline.

### Layer-Wise Interpretability and t-SNE

```bash
python scripts/09_analyze_layers.py --manifest data/analysis.jsonl \
  --output-dir outputs/layers --limit 100 --layers 0,5,10,13,15,20,25,31
```

### Image-Text Modality Alignment

```bash
python scripts/10_analyze_alignment.py --manifest data/analysis.jsonl \
  --output-dir outputs/alignment --limit 100
```

An “image-only” input is not well defined for autoregressive LLaVA. Therefore, this analysis uses the fixed neutral instruction `Describe the image.` as an
image-dominant control and explicitly retains this operational definition in the report.

## 8. Interpreting the Evaluation

- `attack_success_rate`: the proportion of unsafe samples that do not trigger any refusal keyword; lower is better.
- `benign_refusal_rate`: the proportion of safe samples that trigger a refusal; lower is better.
- The probe report includes Accuracy, F1, ROC-AUC, unsafe-class recall, and false positive rate.
- `evaluation/keywords.py` contains the refusal keywords listed in the paper. However, keyword-based ASR is only a reproducible approximation. If the paper's main table
  uses human evaluation or an LLM judge, generate per-sample judge labels separately before comparison; keyword-based results must not be presented as equivalent to the main-table results.

At minimum, report three settings: original LLaVA, a linear probe without projection, and a SASA projection-based linear probe. Also report the mean, random seed,
number of samples, dataset version, threshold, and number of failed samples.

## 9. Testing and Troubleshooting

```bash
make check
make test
```

- **Out of memory (OOM):** Reduce `max_new_tokens`, use multiple GPUs with `device_map: auto`, or change the dtype to `bfloat16` if supported by the hardware.
- **Decoder layers cannot be found:** Check the Transformers version. Module paths are centralized in `model/introspection.py` to simplify adaptation to new versions.
- **Image-token count mismatch:** Do not manually remove `<image>`, and ensure that the model and processor come from the same checkpoint.
- **Results differ from the paper:** First verify the layer-indexing convention, dataset version and subset, prompt template, definition of the first token, random seed, and refusal criterion.
- **Safety notice:** Conduct safety research only in isolated environments using public benchmarks and authorized models. Do not use the testing pipeline to cause real-world harm.

## 10. Known Limitations

When a paper does not release its code, some implementation details cannot be uniquely determined from the manuscript alone. This project makes these ambiguities explicit: the layer-indexing convention, whether probe inputs are
normalized, whether left or right singular vectors are used, the image-control prompt, and whether projection participates in token-by-token decoding are all configurable or supported through separate diagnostic
entry points. Reproduction experiments should freeze the configuration file and disclose each choice individually in the paper or report.

MIT License. The model and datasets remain subject to their respective original licenses and terms of use.
