# Laya · CLINC150 复现指南

## 1. 详细介绍

将 Laya 英文编码器适配为 **150 类意图分类器**，并提供 R-Drop + SupCon 单模型和 Laya + DeBERTa 集成两条复现路线。

**主线：安装 → 准备数据 → 训练基础对照 → 训练 R-Drop → 冻结并测试 → 导出推理。**

| 模型 | Test Accuracy | 推理编码器数 |
|---|---:|---:|
| CE 基线，三种子均值 | 97.015% ± 0.100 pp | 1 |
| CE + SupCon，三种子均值 | 97.326% ± 0.046 pp | 1 |
| R-Drop + SupCon，三种子均值 | 97.6593% ± 0.0898 pp | 1 |
| **验证选出的 R-Drop 单模型** | **97.7556%** | **1** |
| Laya + 三个 DeBERTa-v3-large | 98.0222% | 4 |

`±` 为种子样本标准差，`pp` 为百分点。所有结果均为官方完整 in-scope test 上的本地评测，不包含 OOS 检测。98.0222% 是四模型系统成绩。

**阅读顺序：**[详细介绍](#1-详细介绍) · [复现步骤](#2-复现步骤) · [改进的点](#3-改进的点) · [未来可以探索的方向](#4-未来可以探索的方向) · [源代码与相关数据](#5-源代码与相关数据)

## 2. 复现步骤

### 1. 环境准备

以下命令使用 **Linux / WSL2 + Bash**，解压或克隆后，进入本项目根目录（包含 `train_clinc.py` 的 `clinc150/`），后续命令全部在这里执行。下面采用单卡顺序运行，不依赖原服务器的 `/opt/...` 路径。

**建议使用 16GB 显存的 NVIDIA GPU 进行单模型训练；视训练分支调整 batch。** GPU 需支持 CUDA 和 BF16。 本文按单卡顺序运行，不要求使用原实验的服务器显卡，也不需要多卡并行。

软件环境：Python 3.12、PyTorch 2.9.1+cu128、Transformers 4.57.6。实际显存占用取决于 batch、序列长度和训练分支；单模型与可选的 DeBERTa 集成应分别评估。原实验显卡的总容量不代表最低显存要求。

```bash
# 若已位于包含 train_clinc.py 的目录，跳过下一行。
cd clinc150
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

# upstream 不存在时执行；已有目录先确认固定版本。
git clone https://github.com/NandhaKishorM/laya.git upstream
git -C upstream checkout 42626c348753fbb17572a813127df2278a1ec527

export CUDA_VISIBLE_DEVICES=0
export USE_TF=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
python -m pip freeze > requirements-local.txt
```

检查环境：

```bash
python - <<'PY'
import torch, transformers
assert torch.cuda.is_available(), "未检测到 CUDA GPU"
assert torch.cuda.is_bf16_supported(), "当前训练配置需要 BF16 支持"
print(torch.__version__, transformers.__version__)
print(torch.cuda.get_device_name(0))
PY
```

**完成标志：**打印版本和 GPU 名称。依赖含原实验环境的预发布版本，若包源缺失，见末尾常见问题。

### 2. 准备模型与数据

```bash
python prepare.py
python verify_experiment.py
```

| 对象 | 固定版本 / 数量 |
|---|---|
| 英文基础模型 | `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982` |
| 数据 | `clinc/oos-eval@828f8093932c8fe6ca7936c3d2e52903b1c523de` 的 `data_full.json` |
| Train / validation / test | 15,000 / 3,000 / 4,500 |
| 类别 | 150，仅 in-scope |

**完成标志：**损失性质测试通过，并生成：

```text
data/data_full.json
artifacts/base/model.safetensors
artifacts/base/encoder/
artifacts/base/tokenizer/
artifacts/manifest.json
artifacts/data_audit.json
```

`data_audit.json` 中原始默认格式只保留 126 个候选标记，是已知审计结果。因此本实验改成固定分类头：

```text
文本 → Laya 编码器 → masked mean pooling → dropout → 150 类线性头
```

编码器权重严格加载；原始 `choice/score/noul` 输出接口不再用于这条分类路线。

### 3. 训练第一轮基础对照

先运行 CE / SupCon 各三个种子。这六组产物是第 4 步完整候选比较的前置输入，不能跳过后直接运行最终汇总脚本。

```bash
set -e
for method in ce supcon; do
  for seed in 13 42 87; do
    python train_clinc.py train \
      --method "$method" --seed "$seed" \
      --output "artifacts/runs/${method}_${seed}"
  done
done
```

默认配置为 8 epochs、batch=64、max_length=64、encoder LR=2e-5、head LR=2e-4、AdamW、10% warmup、BF16。SupCon 增加 `0.1 × SupCon`，对比温度为 0.1。

**完成标志：**六个 run 目录中都有 `selection.json`、`model.safetensors` 和 `validation.npz`，例如：

```text
artifacts/runs/
├── ce_13/
├── ce_42/
├── ce_87/
├── supcon_13/
├── supcon_42/
└── supcon_87/
```

所有配置与验证选择完成后，单独运行测试：

```bash
for method in ce supcon; do
  for seed in 13 42 87; do
    python train_clinc.py evaluate \
      --output "artifacts/runs/${method}_${seed}"
  done
done
```

**完成标志：**每个 run 生成 `test_metrics.json` 和 `test_predictions.npz`。与上表 CE / SupCon 结果比较时，读取 `classifier_only.accuracy`，不要混用原型融合后的 `selected` 指标。

只想先验证基本训练流程，可以先跑 `supcon + seed 13` 一组；要继续复现完整第二轮，请补齐其余五组。

### 4. 训练 R-Drop 并完成候选比较

本步包含原实验中预先比较的权重平均、成组 SupCon 和 R-Drop 家族，最终只依据 validation accuracy / NLL 选择。

```bash
python verify_round2.py
python round2.py diagnose
python round2.py soups

# 先导：两个方法各一个种子。
for method in balanced_supcon rdrop; do
  python round2.py train --method "$method" --seed 13
done
touch artifacts/round2/SCREENING_COMPLETE

# 补齐三个种子。
for method in balanced_supcon rdrop; do
  for seed in 42 87; do
    python round2.py train --method "$method" --seed "$seed"
  done
done
touch artifacts/round2/REPLICATES_COMPLETE
```

保留当前 Bash 的 `set -e`，确保前置命令失败时停止，不写入完成标记。上面的 `touch` 只能在对应训练成功结束后执行。

R-Drop 对同一 batch 做两次 dropout 前向，使用：

```text
L = mean(CE₁, CE₂)
  + 0.1 × mean(SupCon₁, SupCon₂)
  + [KL(p₁ || p₂) + KL(p₂ || p₁)] / 2
```

它同时将编码器 dropout 调到 0.1。训练多一次前向，推理仍为一个编码器。成组 SupCon 和权重平均即使不优于 R-Drop，也保留作为完整候选比较的一部分。

**完成标志：**`artifacts/round2/` 下有 `diagnostics.json`，两个训练家族各有三个种子的 `selection.json`，且上述两个完成标记存在。

### 5. 冻结选择、测试和导出

```bash
python round2_finish.py freeze
python round2_finish.py evaluate
python round2_finish.py export
```

三个命令分别完成：

| 命令 | 输出 | 检查内容 |
|---|---|---|
| `freeze` | `artifacts/round2/evaluation_freeze.json` | 验证选中的模型、选择规则、代码 / 数据 / 权重哈希 |
| `evaluate` | `artifacts/round2/test_summary.json` | 完整 4,500 条 test 的结果和三种子统计 |
| `export` | `artifacts/round2/release_single/` | 可独立加载的模型、tokenizer、配置和来源记录 |

查看单模型结果：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("artifacts/round2/release_single")
print("来源:", json.loads((p / "provenance.json").read_text())["source"])
print("指标:", json.loads((p / "test_metrics.json").read_text())["classifier_only"])
PY
```

历史验证选中了 `rdrop_42`，测试为 **97.7556%（4,399 / 4,500）**。本次重训必须保留本次验证选择，不能因为其他种子测试更高而换模型。

### 6. 输入自己的文本

```bash
python infer_clinc.py \
  --checkpoint artifacts/round2/release_single \
  --text "what is my bank balance" \
  --text "please book a flight to London"
```

**完成标志：**每条文本返回 `text`、`intent` 和 `probability`。该模型为固定 150 类闭集分类器，不提供 OOS 拒识。

部署时保存完整 `release_single/`，不能只复制 `model.safetensors`。推理使用本项目的 `infer_clinc.py`，不是 Transformers 通用分类 pipeline。

### 可选：复现 98.0222% 异构集成

先完成第 1–5 步，确保 `artifacts/round2/rdrop_42/` 下有权重、验证 logits 和测试预测。这条路线固定该历史 Laya 模型，另外训练 DeBERTa；不会自动跟随本次 `release_single` 选出其他种子。

#### A. 下载固定 DeBERTa 权重

```bash
python -m pip install -r requirements-hybrid.txt
python prepare_hybrid.py --metadata-only
python prepare_hybrid.py
```

脚本先从官方 Hub 获取固定 revision 的元数据，再下载并校验文件。成功后出现 `artifacts/hybrid/BASE_READY`。

#### B. 先导验证与种子复验

```bash
python hybrid.py --seed 42
touch artifacts/hybrid/PILOT_COMPLETE
python hybrid_replicates.py
```

最后一条命令会写入 `replicate_decision.json` 并输出 `yes` / `no`。历史协议门槛是先导验证准确率超过 98.4%；**只有输出 `yes` 时**才继续补训：

```bash
python hybrid.py --seed 13
python hybrid.py --seed 87
touch artifacts/hybrid/TRAINING_COMPLETE
```

若输出 `no`，本次运行没有满足原协议的扩展条件，应保留这一结果，不强行补训以追逐历史分数。

#### C. 冻结、评测与导出复验

```bash
python hybrid_finish.py freeze
python hybrid_finish.py evaluate
python hybrid_finish.py export
python verify_hybrid_export.py

python infer_hybrid.py \
  --checkpoint artifacts/hybrid/release_system \
  --text "what is my bank balance"
```

历史冻结选择为：

```text
P = 0.5 × P(Laya rdrop_42)
  + [P(DeBERTa seed 13) + P(DeBERTa seed 42) + P(DeBERTa seed 87)] / 6
```

结果为 **98.0222%（4,411 / 4,500）**。`verify_hybrid_export.py` 用导出推理接口重跑 test，历史记录的 `prediction_mismatches` 为 0；本次也应检查导出与本次冻结预测一致。

### 常见问题

| 问题 | 处理方法 |
|---|---|
| `No such file` 或相对路径错误 | 回到包含 `train_clinc.py` 的仓库根目录 |
| 第二轮缺少 `validation.npz` | 完成第 3 步的六组训练 |
| 第二轮缺少旧模型 `test_predictions.npz` | 完成第 3 步末尾的六组测试 |
| `Already frozen` / `Test already evaluated` | 说明该阶段已有证据；不要删除记录强行重跑，使用新的实验目录副本 |
| 安装脚本出现 `/opt/...` 不存在 | 本指南已提供通用命令，无需运行历史 `setup_remote.sh` |
| CUDA OOM | 先确认显卡没有被其他任务占用。第一轮支持 `--batch-size 16` 等更小 batch；第二轮及异构脚本的 batch 在代码内固定。调整 batch 会改变 SupCon 的正负样本集合，需另记配置，不能保证复现原分数 |
| 下载失败或哈希不一致 | 核对固定 revision 和下载是否完整，不要跳过校验 |
| 包源没有预发布依赖 | 使用具有锁定版本的包源；若替换版本，记录新环境并重新验证，不视为原环境逐项一致 |
| 想直接推理，不想训练 | 需要先获得完整导出权重；本仓库尚无公共微调权重下载地址 |

### 复现范围与更多资料

本指南覆盖推荐单模型的完整候选流程，以及可选异构集成。第一轮冻结探针、检索、校准、第三轮蒸馏、第四轮重排器属于额外研究，不是主线运行所必需。

本文命令已对照实际脚本参数与前置产物检查；本次文档整理没有重跑 GPU 实验或验证全新机器安装。结果不保证跨硬件逐 bit 一致；多轮开发已知历史测试汇总，不能称为独立盲测或正式榜单成绩。

- [第二轮方法说明](ROUND2_METHODS.md)
- [第二轮完整结果](ROUND2_REPORT.md)
- [异构集成报告](HYBRID_REPORT.md)
- [第一轮详细技术报告](TECHNICAL_REPORT.md)

## 3. 改进的点

### 3.1 从动态候选选择适配成固定分类

原始 Laya 的输入需要同时容纳问题、候选标签和文本。本实验审计的默认 512-token 格式仅保留 126 个候选标记，无法代表完整 150 类任务。

`train_clinc.py` 严格加载 Laya 编码器，使用非 padding token 均值池化与 150 类线性分类头。标签不再占用输入预算，文本最大长度为 64；原始 train/validation 中最长输入分别为 39/30 token。

这一步改变了模型的任务接口，是 CLINC150 专用适配，不能理解为原生 Laya 全部能力的提升。

### 3.2 监督对比损失

普通 CE 学习正确标签，SupCon 同时鼓励 batch 内同类文本表示靠近：

```text
L = CE + 0.1 × SupCon
z = normalize(masked_mean(encoder(text)))
温度 = 0.1
```

无同类正对的 anchor 不参与 SupCon。第一轮在相同训练设置下，三种子均值从 97.015% 提高至 97.326%，平均增加 0.311 pp。原型与近邻融合没有稳定提高 test，未将其包装为必然有效的改进。

### 3.3 R-Drop 与编码器 dropout

第二轮将编码器 dropout 设为 0.1，并对两次随机前向加入对称 KL。三种子测试均值为 97.6593%，相对第一轮 SupCon 增加 0.3333 pp；验证选中的单模型为 97.7556%。

训练需要双前向，推理仍只需单前向。该实验同时改变 dropout 和一致性目标，未单独消融两者，不能把全部收益归因于 KL。

### 3.4 异构概率融合

固定 Laya，再训练三个 DeBERTa-v3-large，用验证集选择融合比例，最终系统为 98.0222%。相对推荐 Laya 单模型净多判对 12 条，代价是四个编码器的权重和前向计算。

DeBERTa 分量使用不同预训练底座；这是系统层面的收益，不是 Laya 单模型的结构改进。蒸馏、权重平均、成组采样和原生 top-5 重排器等后续实验也保留了未提升的结果，详见相关报告。

## 4. 未来可以探索的方向

以下是下一步可以验证的实验，不代表已经达到的效果。

| 方向 | 建议怎么做 | 判断是否有效 |
|---|---|---|
| 拆解 R-Drop 收益 | 比较 SupCon、SupCon+dropout、SupCon+dropout+KL，补充等计算量对照 | 看配对种子提升，而非只比较最佳 run |
| 降低集成部署成本 | 将四模型概率蒸馏到一个学生，并与等训练预算单模型比较 | 同时报告准确率、延迟、显存和训练成本；既有蒸馏未成功，不保证提升 |
| 扩展 OOS 检测 | 使用单独定义的 OOS 训练/验证协议和检测阈值 | 区分 150 类准确率与 OOS 检测指标 |
| 减少候选输入限制 | 在原生决策结构上测试完整候选或分层候选方案 | 同时报全类别召回与最终准确率，不能仅报召回成功样本 |
| 独立泛化验证 | 将冻结的方法迁移至 Banking77 / 其他意图数据，不用新 test 调参 | 统一报告正负结果与跨数据集差异 |
| 完善可下载产物 | 发布完整导出权重、锁定环境、校验清单和评估入口 | 新环境能加载模型并重算全部预测 |

继续开发时先冻结方法和评测规则，减少反复查看同一个测试集引入的选择偏差。

## 5. 源代码与相关数据

### 5.1 源代码

| 文件 | 作用 |
|---|---|
| [prepare.py](prepare.py) | 固定模型 / 数据下载与 manifest |
| [verify_experiment.py](verify_experiment.py) | 损失性质、输入预算与数据审计 |
| [train_clinc.py](train_clinc.py) | 分类器结构、CE / SupCon、训练与第一轮评测 |
| [round2.py](round2.py) | R-Drop、成组采样、权重平均与验证比较 |
| [round2_finish.py](round2_finish.py) | 选择冻结、完整 test 与导出 |
| [infer_clinc.py](infer_clinc.py) | 专用分类器推理 |
| [prepare_hybrid.py](prepare_hybrid.py) / [hybrid.py](hybrid.py) | DeBERTa 下载与融合训练 |
| [hybrid_finish.py](hybrid_finish.py) / [infer_hybrid.py](infer_hybrid.py) | 异构系统冻结、测试、导出和推理 |
| [verify_hybrid_export.py](verify_hybrid_export.py) | 导出系统逐样本预测复核 |
| [requirements.txt](requirements.txt) / [requirements-hybrid.txt](requirements-hybrid.txt) | 依赖版本 |

上游：[Laya 固定 commit](https://github.com/NandhaKishorM/laya/tree/42626c348753fbb17572a813127df2278a1ec527)。预训练模型：[英文 Laya 固定版本](https://huggingface.co/convaiinnovations/laya/tree/1c5edc17a7acd8701df6fc341c0d179f1c62c982)、[DeBERTa 固定版本](https://huggingface.co/microsoft/deberta-v3-large/tree/64a8c8eab3e352a784c658aef62be1662607476f)。

### 5.2 相关数据

来源：[CLINC 官方仓库固定版本的 data_full.json](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json)。运行 `prepare.py` 后自动保存到 `data/data_full.json`。

| Split | 样本数 | 用途 |
|---|---:|---|
| train | 15,000 | 更新参数、构建训练原型或检索库 |
| val | 3,000 | 选择 epoch、温度和融合比例 |
| test | 4,500 | 冻结选择后评测 |
| oos_* | 不纳入本文主任务 | 不能将闭集结果解释为 OOS 效果 |

官方划分保持不变。规范化文本审计发现 train/val 重复 3 个、train/test 重复 2 个、val/test 无重复；补充结果排除与训练重复的 test 文本，主结果仍采用完整官方 test。

### 5.3 训练后需要保存什么

保存 `artifacts/manifest.json`、`requirements-local.txt`、每个 run 的配置、验证选择、权重及逐样本预测；第二轮保留 `evaluation_freeze.json` 与 `test_summary.json`，部署保留完整 `release_single/` 或 `release_system/`。

代码随仓库提供，数据与基础权重由脚本下载。
