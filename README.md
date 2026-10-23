# DeepOpen

**English** | [简体中文](readme-cn.md)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-lockup-dark.png" />
    <img src="assets/logo-lockup.png" alt="DeepOpen" width="330" />
  </picture>
</p>

**An intent-classification framework and non-autoregressive SDK for structured decisions.**

DeepOpen evaluates predefined questions over text or JSON and returns labels, scores, and probabilities. It does not generate free-form text. Output types are constrained; predictions can still be wrong, and probabilities require validation on your own data.

DeepOpen combines a decision SDK, dataset-specific training strategies, and reproducible Banking77 and CLINC150 evaluations. See [Resources and attribution](#resources-and-attribution) for the upstream foundations and this repository's contributions.

## Benchmark highlights

| Dataset | DeepOpen test accuracy | Configuration | Correct / total |
|---|---:|---|---:|
| **BANKING77** | **94.1234%** | Single classifier, SAM seed 13 | 2,899 / 3,080 |
| **CLINC150** | **98.0222%** | Four-encoder ensemble | 4,411 / 4,500 |

[Explore both benchmark comparisons, predictions, and evaluation evidence on Hugging Face](https://c41-laya-evaluation.static.hf.space/index.html?version=ensemble-2#comparison).

### BANKING77 comparison

| Method | Test accuracy | Result source |
|---|---:|---|
| SPACE | 94.94% | Selected published reference |
| SPACE 2.0 | 94.77% | Selected published reference |
| Llama 2 7B (FT) | 94.35% | Selected published reference |
| ConvFiT | 94.16% | Selected published reference |
| **DeepOpen (SAM, best observed)** | **94.1234%** | Author-reported; saved predictions |
| PPTOD | 93.86% | Selected published reference |

### CLINC150 comparison

| Method | Test accuracy | Result source |
|---|---:|---|
| TOATOD-small | 98.45% | Selected published reference |
| **DeepOpen (ensemble)** | **98.0222%** | Author-reported; saved predictions |
| TOATOD-base | 98.01% | Selected published reference |
| SPACE | 97.89% | Selected published reference |
| SPACE 2.0 | 97.80% | Selected published reference |

These tables reproduce the selected references in the [HF evaluation page](https://c41-laya-evaluation.static.hf.space/index.html?version=ensemble-2#comparison) (reference snapshot: 7 October 2026); they do not assign an official leaderboard rank. BANKING77 displays the best test result of three SAM seeds, selected after evaluation; the three-seed mean is **94.0368%**, with no statistically established improvement over the replayed baseline. CLINC150 covers **150 in-scope intents only**, without OOS detection, and combines one Laya-based specialist with three DeBERTa-v3-large models. Full protocol, execution caveats, and prediction files are available on the linked page.

## Installation

Use Python **3.10 or newer** and Git. Install from this repository; the installation instructions do not depend on a PyPI release named `deepopen`.

```bash
python -m pip install "git+https://github.com/deepopen-com/deepopen.git"
python -c "import deepopen; print(deepopen.__version__)"
```

For a CPU-only environment, install CPU PyTorch before the SDK:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "git+https://github.com/deepopen-com/deepopen.git"
```

GPU users should install a PyTorch build compatible with their driver and CUDA environment. The benchmark training scripts have separate dependencies and instructions in their respective directories.

## Quick start

This example loads the English upstream checkpoint and answers one classification question. The first call downloads weights and initializes the model; it is not a warm-inference latency measurement.

```python
import deepopen

agent = deepopen.load("convaiinnovations/laya", device="cpu")
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    }
}
result = agent.predict(
    {"body": "I was charged twice. Please refund the duplicate payment."},
    questions,
)
print(result["answers"]["department"])
```

The answer includes `choice`, `probabilities`, and `confidence`. Results depend on the checkpoint and input; a high confidence score is not a guarantee of correctness. Use `device="cuda"` with a compatible GPU environment. A local checkpoint directory can replace the Hub ID.

## Routing and model loading

```python
from deepopen import Router

router = Router(device="cpu")
# Inspect routing without downloading weights or running inference.
print(router.route({"body": "Please refund the duplicate charge."}, questions))

# The first prediction loads the selected checkpoint.
result = router.predict({"body": "Please refund the duplicate charge."}, questions)

# Select the workflow-specific checkpoint explicitly when appropriate.
# result = router.predict(state, questions, model="typed-decisions")
```

Language routing uses script and language heuristics, not an accuracy oracle. The typed-decisions checkpoint is **not automatically selected by default**. Select it explicitly, or opt in to task detection with `auto_task_detection=True` after validating the behavior on your workload.

The default cache keeps one model loaded. For repeated multilingual traffic, use `Router(max_loaded=2)` and `router.preload(["english", "multilingual"])`. `Router(preload=True)` loads all three models and needs memory for all of them. Preloading avoids repeated model loads; it does not guarantee a fixed inference latency.

| Router key | Default Hub location | Encoder | Default context |
|---|---|---|---:|
| `english` | [Laya repository root](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large | 512 |
| `multilingual` | [Laya / multilingual](https://huggingface.co/convaiinnovations/laya/tree/main/multilingual) | mmBERT-base | 1,024 |
| `typed-decisions` | [Laya / typed-decisions](https://huggingface.co/convaiinnovations/laya/tree/main/typed-decisions) | ModernBERT-large | 1,024 |

The Python package is named `deepopen`; upstream model IDs retain `laya` so they match the loader configuration.

## Decision types and workflow presets

| Type | Output | Example |
|---|---|---|
| `choice` | Label, option probabilities, confidence | Department or intent classification |
| `score` | Expected ordinal level and distribution | Urgency scoring |
| `noul` | Estimated probability that a statement is true | Refund request detection |

The SDK provides `router_questions()`, `guard_questions()`, `moderation_questions()`, `triage_questions()`, and `email_questions()`. These are question templates, not production safety guarantees. Validate accuracy and calibrate confidence before automating consequential decisions.

## Earlier intent-classification experiments

The table below preserves earlier local evaluations and training recipes. The HF highlights above additionally include the later BANKING77 SAM continuation result (94.1234%); the older 94.0909% result is a different training and execution configuration. These are not confirmed official leaderboard placements. They require task-specific training; installing the SDK does not reproduce these scores automatically.

| Dataset | Configuration | Test accuracy | Scope |
|---|---|---:|---|
| Banking77 | Laya encoder + classification head, 7 training epochs + 4 continuation epochs | **94.0909%** | 3,080 test examples, 77 classes |
| CLINC150 | Validation-selected R-Drop single model | **97.7556%** | 4,500 in-scope test examples, 150 classes |
| CLINC150 | Laya + three DeBERTa-v3-large models | **98.0222%** | Four-model ensemble; same in-scope test |

CLINC150 results exclude out-of-scope detection. The Banking77 headline is the ordinary classification-head continuation result, not a prototype-branch gain. The ensemble uses four encoders and its latency must not be represented by the single-checkpoint measurements below.

- [Banking77 reproduction guide](banking77/README.md): training, evaluation, sources, and alternative branches (Chinese).
- [CLINC150 reproduction guide](clinc150/README.md): single-model and ensemble protocols, seeds, and scope (Chinese).

## Upstream checkpoint measurements and limitations

The [benchmark report](BENCHMARKS.md) describes separate checkpoint experiments. Do not combine the following rows into a single configuration's performance claim.

| Measurement | Configuration and conditions | Historical result |
|---|---|---:|
| Single-question latency | Multilingual checkpoint, warm inference, NVIDIA T4 | 32.8 ms |
| Ten-question latency | Multilingual checkpoint, one batched call, NVIDIA T4 | 72.3 ms total; 7.2 ms/question |
| Typed-decisions accuracy | Task-fine-tuned checkpoint; 400 cases / 2,000 decisions | 0.766 |
| Mean ECE after temperature refitting | English base checkpoint, calibration experiment | 0.081 |
| Multilingual coverage evaluation | 51 MASSIVE languages, 20 options | 45 / 51 exceed 3× random accuracy |

The encoder's advertised 100+ language coverage does not establish equal quality in every language. Base checkpoints are near chance on the typed-decisions zero-shot task and below its majority-class baseline. Both base checkpoints are overconfident as shipped; temperature fitting must use a separate calibration split. Held-out moderation accuracy is only 0.530 in the reported experiment. Large option sets compete for a shared prompt budget, and ordinal scoring remains a weak point.

Jev comparisons in the historical report use third-party published results, with different prompts and sample sizes. They are indicative comparisons, not a controlled head-to-head evaluation. Self-hosting avoids per-token API fees but still incurs hardware, electricity, and operating costs.

## Resources and attribution

- [Detailed benchmark report](BENCHMARKS.md) and [research scripts](research/README.md).
- [T4 benchmark notebook](research/scripts/laya_benchmark_colab.ipynb).
- [Typed-decisions fine-tuning notebook](notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb).
- [Upstream Laya source](https://github.com/NandhaKishorM/laya) and [pretrained weights](https://huggingface.co/convaiinnovations/laya).

The SDK and base model architecture derive from Laya. This repository provides DeepOpen packaging and documentation, Banking77 adaptation experiments, and CLINC150 R-Drop/SupCon and heterogeneous-ensemble reproduction workflows. See the [Banking77 source manifest](banking77/SOURCE_MANIFEST.json) and [CLINC150 source manifest](clinc150/SOURCE_MANIFEST.json) for pinned source provenance. SDK defaults continue to load upstream checkpoints; the experiment guides explain how to train and export task-specific models.

## Development

```bash
python -m pip install -e .
python tests/test_router.py
python tests/test_criteria.py
```

These tests cover routing and input handling without downloading model weights. `tests/test_local_e2e.py` additionally requires local checkpoints; its module documentation describes the directory layout. CI also checks lint, package builds, and the installed wheel. The SDK version is defined once in `deepopen/_version.py`.

## License

[Apache License 2.0](LICENSE). Retain upstream attribution and review the separate model and dataset licenses for your intended use.
