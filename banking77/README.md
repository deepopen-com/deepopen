# Laya · Banking77 复现指南

## 1. 详细介绍

使用 Laya 英文编码器完成 **77 类银行意图分类**。本目录包含训练、测试和推理所需的实验脚本，可以从原始预训练权重开始运行。

**主线：准备环境 → 下载数据和模型 → 全量训练 7 轮 → 继续训练 4 轮 → 测试 → 输入自己的文本。**

| 模型 | Test Accuracy | Macro-F1 |
|---|---:|---:|
| 原生 Laya 决策结构微调 | 93.4416% | 93.4214% |
| Laya 编码器 + 77 类分类头，训练 7 轮 | 93.7987% | 93.8050% |
| **普通分类头继续训练 4 轮（本文主线）** | **94.0909%** | **94.0922%** |
| Semantic 原型分支实验 | 93.7013% | 约 93.71% |

这些是历史实验在官方 3,080 条公开测试集上的本地结果。94.0909% 来自普通训练对照，不能归因于原型分支；重新训练的数值可能因硬件和环境略有变化。

**阅读顺序：**[详细介绍](#1-详细介绍) · [复现步骤](#2-复现步骤) · [改进的点](#3-改进的点) · [未来可以探索的方向](#4-未来可以探索的方向) · [源代码与相关数据](#5-源代码与相关数据)

## 2. 复现步骤

### 1. 环境准备

以下命令在 **Linux / WSL2 的 Bash** 中执行。解压或克隆后，进入本项目根目录（包含 `banking77.py` 的 `banking77/`），后续步骤都留在这个目录。

**建议使用 16GB 显存的 NVIDIA GPU 进行单模型训练；视训练分支调整 batch。** GPU 需支持 CUDA 和 BF16。 本文按单卡顺序训练，不要求使用原实验的服务器显卡，也无需同时占用两张卡。

软件环境：Python 3.12、PyTorch 2.9.1+cu128、Transformers 4.57.6。实际显存占用取决于 batch、序列长度和模型分支；原实验显卡的总容量不代表最低显存要求。

```bash
# 若已位于包含 banking77.py 的目录，跳过下一行。
cd banking77
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

git clone https://github.com/NandhaKishorM/laya.git repo
git -C repo checkout 42626c348753fbb17572a813127df2278a1ec527
python -m pip install --no-deps -e ./repo
git -C repo rev-parse HEAD > UPSTREAM_COMMIT

export CUDA_VISIBLE_DEVICES=0
export USE_TF=0 TOKENIZERS_PARALLELISM=false
python -m pip freeze > requirements-local.txt
```

检查 GPU 和依赖：

```bash
python - <<'PY'
import torch, transformers, laya
assert torch.cuda.is_available(), "未检测到 CUDA GPU"
assert torch.cuda.is_bf16_supported(), "当前训练配置需要 BF16 支持"
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY
```

**完成标志：**打印出版本和 GPU 名称，没有报错。依赖文件按原实验直接依赖锁定，包含预发布版本；包源缺失时不要直接忽略安装错误，见末尾常见问题。

### 2. 准备数据与预训练模型

```bash
python banking77.py prepare
```

该命令下载固定版本的数据和英文 Laya，按原实验规则生成标签映射与内部验证集。

| 内容 | 固定设置 |
|---|---|
| 基础模型 | `convaiinnovations/laya` |
| 模型 revision | `1c5edc17a7acd8701df6fc341c0d179f1c62c982` |
| 数据集 | `mteb/banking77` |
| 数据 revision | `18072d2685ea682290f7b8924d94c62acc19c0b2` |
| 内部划分 | 官方 train 按标签分层划分 85% / 15%，split seed=2026 |
| 标签映射 | 按 `label_text` 排序，共 77 类 |

**完成标志：**终端显示 `PREPARED 8502 1501 3080`，目录中出现 `data.json` 和 `manifest.json`。

```bash
python - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("data.json").read_text())
for key, count in {"train":8502, "validation":1501, "full_train":10003, "test":3080}.items():
    assert len(d[key]) == count
    print(key, len(d[key]))
assert len(d["labels"]) == 77
print("数据检查通过")
PY
```

后面的主线使用**已确定的历史配置**在全部 10,003 条训练样本上重训。若要重新研究超参数，先走“重做验证集选模”，不要使用 test 选轮次。

### 3. 训练基础分类器

基础结构为 Laya 编码器 + Transformers ModernBERT 的 77 类分类头。编码器严格加载原 Laya 的 `encoder.*` 权重，分类头重新初始化；它与原生 Laya 的可变候选决策头不同。

```bash
python classifier.py train \
  --name encoder_full_train \
  --full-train \
  --lr 2e-5 \
  --epochs 7 \
  --seed 42
```

| 参数 | 设置 |
|---|---|
| 训练数据 | 全部 10,003 条官方 train |
| Encoder / head LR | 2e-5 / 2e-4 |
| Batch / max length | 32 / 128 |
| Optimizer / warmup | AdamW / 10%，之后线性衰减 |
| Weight decay / 梯度裁剪 | 0.01（矩阵参数）/ 1.0 |
| 精度 | BF16 |

**完成标志：**出现 `classifier_runs/encoder_full_train/best/model.safetensors` 和 `done.json`。全量模式保存第 7 轮的模型，不再用内部验证集选择 checkpoint。

### 4. 继续训练

从上一步 checkpoint 出发，用更小学习率继续训练 4 轮。这一步复现历史的同轮数普通训练对照。

```bash
python prototype_train.py train \
  --variant control \
  --name matched_control \
  --full-train \
  --source classifier_runs/encoder_full_train/best \
  --epochs 4 \
  --lr 5e-6 \
  --seed 42
```

虽然脚本叫 `prototype_train.py`，`--variant control` 会关闭原型分支和辅助损失，仅继续普通分类训练。Encoder / head LR 为 5e-6 / 5e-5，microbatch=16，累积两次得到有效 batch=32。

**完成标志：**出现以下文件：

```text
research_v2/matched_control/
├── experiment_config.json
├── history.json
├── done.json
└── best/
    ├── prototype_config.json
    ├── prototype.safetensors
    └── backbone/                 # 可独立使用的标准 Transformers 分类器
        ├── config.json
        ├── model.safetensors
        └── tokenizer 文件
```

训练共分两阶段：**7 轮 + 重置优化器和调度器后追加 4 轮**，不能直接改成一次连续训练 11 轮。

### 5. 评测完整测试集

主线模型无需自定义原型类即可评测，直接加载 `best/backbone`：

```bash
mkdir -p evaluation
python deepopen_evaluate.py \
  --model research_v2/matched_control/best/backbone \
  --device cuda \
  --output evaluation/matched_control.json
```

脚本自动下载固定版本 `test.jsonl` 并检查 SHA256，按 checkpoint 的 `label2id` 映射标签。推理口径为 SDPA、BF16、batch=16、max_length=128。

原 checkpoint 的参考结果：

```json
{
  "accuracy": 0.9409090909090909,
  "macro_f1": 0.940922,
  "n": 3080
}
```

其中 F1 展示为近似值。准确率对应 **2,898 / 3,080**，评估 JSON 同时保存 `gold` 和 `predicted`，便于独立复算。

**完成标志：**样本数为 3080，输出文件存在。重新训练不保证得到与原权重逐条完全相同的预测。

### 6. 输入自己的文本

```bash
python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "research_v2/matched_control/best/backbone"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, attn_implementation="sdpa"
).cuda().eval()

texts = ["My card has been stolen", "Why was my cash withdrawal declined?"]
batch = tokenizer(texts, padding=True, truncation=True,
                  max_length=128, return_tensors="pt").to("cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    probabilities = model(**batch).logits.float().softmax(-1)
for text, p in zip(texts, probabilities):
    index = int(p.argmax())
    print({"text": text, "intent": model.config.id2label[index],
           "probability": float(p[index])})
PY
```

**完成标志：**每条输入得到一个 Banking77 意图标签和概率。模型属于固定闭集分类器，对不属于这 77 类的输入也会输出其中一类。

### 可选：重做验证集选模

主线固定复用了历史选择：基础 LR=2e-5、7 轮，追加训练 4 轮。要复现选择过程，先在 8,502 条 train / 1,501 条 validation 上运行：

```bash
python classifier.py train --name encoder_lr2e5 --lr 2e-5 --epochs 8 --seed 42
python classifier.py train --name encoder_lr5e5 --lr 5e-5 --epochs 8 --seed 42

for variant in control prototype semantic; do
  python prototype_train.py train \
    --variant "$variant" --name "${variant}_s42" \
    --source classifier_runs/encoder_lr2e5/best \
    --epochs 4 --seed 42
done
```

先完成基础分类器验证选模，再进行追加训练消融。历史 LR=2e-5 模型在第 7 轮最好；上述追加训练消融均按原协议从该模型开始，而不是自动追随新机器可能选出的不同 LR。脚本只允许这一固定验证起点。

| 追加训练分支 | 历史验证 Accuracy | 最佳追加 epoch |
|---|---:|---:|
| control | 91.61% | 2 |
| prototype | 91.87% | 2 |
| semantic | 92.14% | 4 |

历史研究按验证结果选中了 semantic，随后在全量数据上为 semantic 和匹配 control 都追加 4 轮；因此主线 control 的 4 轮不是它自身验证最优的 2 轮。

如果也要重现 semantic 的最终模型：

```bash
python prototype_train.py train \
  --variant semantic --name full_train_semantic --full-train \
  --source classifier_runs/encoder_full_train/best --epochs 4 --seed 42

python prototype_train.py evaluate \
  --checkpoint research_v2/full_train_semantic/best \
  --split test --name semantic_test
```

semantic 将训练类中心与 20% 类别名称表示混合初始化，再学习余弦原型残差。它的测试准确率为 93.7013%，低于 control。不能用 control 的 94.0909% 证明该结构有效。

### 可选：复现原生 Laya

这条路线保留原编码器、decision head、option scorer，并按分词后的候选长度扩展输入预算，确保 77 个候选标记齐全。使用历史验证选出的配置直接全量重训：

```bash
python banking77.py train \
  --name native_full --full-train \
  --lr 5e-5 --batch 32 --micro 16 \
  --wd 0.01 --warmup 0.06 --seed 123 --epochs 7

python evaluate_checkpoint.py \
  --checkpoint runs/native_full/best \
  --output evaluation/native_test.json
```

原实验参考准确率为 93.4416%。该 checkpoint 使用 Laya SDK 加载；不要用普通分类头的 Transformers 推理代码加载它。

### 常见问题

| 问题 | 处理方法 |
|---|---|
| 缺少 `data.json` / `manifest.json` | 回到第 2 步，确认 `prepare` 成功 |
| 缺少 `encoder_full_train/best` | 先完成第 3 步，不能跳过基础训练 |
| `ModuleNotFoundError: laya` | 激活 `.venv`，执行 `python -m pip install --no-deps -e ./repo` |
| 下载失败 | 检查网络、Hub 缓存和版本；如使用镜像，下载前设置 `HF_ENDPOINT`，保持 revision 不变 |
| CUDA OOM | 先确认显卡没有被其他任务占用。原生 Laya 路线支持减小 `--micro` 并启用 `--gradient-checkpointing`，保持 `--batch 32`；普通分类头与追加训练的 batch 在代码内固定，需修改代码后另记设置 |
| 包源找不到锁定版本 | 依赖含原环境的预发布版本；使用提供这些版本的包源，或记录替代版本后重新做兼容性验证 |
| 在别处运行找不到文件 | 本文所有命令均在 `banking77/` 下执行 |
| 重跑覆盖旧文件 | 使用全新的目录副本或新的 `--name`；主线依赖名称也要同步修改 |

### 实现和复现范围

本目录脚本来自已完成的 Banking77 实验。为便于独立运行，仅把 `banking77.py` 中固定的服务器根路径改成脚本目录，支持用 `LAYA_BANKING77_ROOT` 覆盖；训练目标与超参数未改。

当前没有附带完整微调权重或公共模型下载地址，因此首次使用应从第 1 步开始训练。本文的命令和参数已对照代码检查，本次文档整理没有重新执行 GPU 训练，也没有验证全新环境安装。

主结果仅一个训练种子，多轮开发已知先前测试汇总，未获得正式榜单回执。准备脚本会读取基础编码器的架构配置；严格复刻原环境时还应保留该配置及全部依赖、数据和权重哈希。

CLINC150 属于另一个独立项目，本项目运行无需它的目录或文件。

## 3. 改进的点

### 3.1 让原生 Laya 完整处理 77 个候选

原生模型把意图候选和输入文本共同编码。`banking77.py` 根据 tokenizer 计算全部候选占用的 token 数，扩展 `head_max_len` 和总输入预算，并断言候选标记数等于 77。这样不会因为输入预算不足而丢失类别。

未微调 checkpoint 的历史 test accuracy 从默认格式的 45.3896% 变为完整选项格式的 57.2403%。这反映本任务输入适配的影响，不是训练算法的提升。继续采用 CE 微调原生分类路径后达到 93.4416%。

### 3.2 用专用分类头替代候选打分

`classifier.py` 保留 Laya 预训练编码器，新增 Transformers ModernBERT 的非线性分类头，输出 77 个 logits。输入只保留用户文本，max_length=128，避免每次拼接全部候选。编码器通过 `strict=True` 加载，训练时与分类头共同更新。

这条路线将接口固定为 Banking77 分类。历史 7 轮训练为 93.7987%；不能直接把它解释为原生结构在相同计算预算下的改进，因为架构与训练设置不同。

### 3.3 分阶段继续训练

主线在全量 7 轮分类器上，用较小学习率追加 4 轮。第二阶段重新创建优化器和 warmup 调度，普通对照达到 94.0909%，比原生微调净多判对 20 条，比 7 轮分类器净多判对 9 条。

该结果只来自一个种子。它支持“当前配置下继续训练有效”的观察，还不能证明所有种子都稳定提升。

### 3.4 已尝试但未带来测试提升的原型分支

`prototype_model.py` 增加可训练余弦原型分支：

```text
logits = base_logits + sigmoid(gate) × scale × cosine(mean_hidden, prototypes)
L = CE(logits, y) + 0.2 × CE(prototype_logits, y) + 0.01 × anchor_loss
```

原型分支使用 masked mean 表示，基础分类头保留其原有池化方式；两者共用一个编码器。Semantic 变体的初始原型由 80% 训练类中心和 20% 类别名称表示混合后归一化得到，新增可训练参数 78,850 个。

它在验证集上胜出，最终 test 为 93.7013%，低于普通对照的 94.0909%。因此保留为负结果，不把普通对照的成绩归到新增结构。

## 4. 未来可以探索的方向

以下均为待验证计划，不是已完成结果。

| 方向 | 建议怎么做 | 判断是否有效 |
|---|---|---|
| 多种子稳定性 | 为 7 轮分类器、追加训练 control、semantic 配对运行至少 3 个种子 | 报告所有种子、均值、标准差和配对差值 |
| 分离额外训练收益 | 固定训练预算，比较连续 11 轮、7+4 分阶段、只降低学习率 | 区分训练时长与调度重启的贡献 |
| 简化原型分支 | 分别关闭语义初始化、辅助 CE、锚点约束，固定其余超参数 | 先比较验证集，再冻结一次测试 |
| 置信度与拒识 | 仅用 validation 拟合温度，并另外准备 OOS 数据 | 同时报告分类准确率、校准指标和 OOS 指标 |
| 推理成本 | 同机同 batch 比较原生决策结构与普通分类头 | 分别测分词、前向、峰值显存和端到端延迟 |

为了减少继续使用同一测试集造成的选择偏差，应先写好候选范围与选择规则，再开展实验；有条件时加入独立外部数据验证。

## 5. 源代码与相关数据

### 5.1 源代码

| 文件 | 作用 |
|---|---|
| [banking77.py](banking77.py) | 固定数据划分、原生候选输入适配和原生 Laya 训练 |
| [classifier.py](classifier.py) | Laya 编码器 + 77 类分类头训练 |
| [prototype_train.py](prototype_train.py) | Control / prototype / semantic 的追加训练与评测 |
| [prototype_model.py](prototype_model.py) | 共享编码器的原型残差实现与权重加载 |
| [deepopen_evaluate.py](deepopen_evaluate.py) | 标准分类器完整 test 评测与数据哈希校验 |
| [evaluate_checkpoint.py](evaluate_checkpoint.py) | 原生 checkpoint 评测和 SDK 标签映射核对 |
| [requirements.txt](requirements.txt) | 直接依赖版本 |
| [原始实验结果](RESULTS.md) / [追加实验结果](RESULTS_V2.md) | 历史设置、指标和实验边界 |

上游源码：[NandhaKishorM/laya 固定 commit](https://github.com/NandhaKishorM/laya/tree/42626c348753fbb17572a813127df2278a1ec527)。基础权重：[Laya 固定 revision](https://huggingface.co/convaiinnovations/laya/tree/1c5edc17a7acd8701df6fc341c0d179f1c62c982)。本目录复用脚本的来源与原文件哈希见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。

### 5.2 相关数据

数据来源：[mteb/banking77 固定版本](https://huggingface.co/datasets/mteb/banking77/tree/18072d2685ea682290f7b8924d94c62acc19c0b2)。不需要手工改标签，`prepare` 自动构造划分与标签表。

| 数据文件 | 条数 / 用途 |
|---|---|
| 官方 `train.jsonl` | 10,003；最终全量训练 |
| 内部 `data.json → train` | 8,502；验证阶段训练 |
| 内部 `data.json → validation` | 1,501；选择超参数和 epoch |
| 官方 `test.jsonl` | 3,080；每类 40 条，最终评测 |
| `manifest.json` | 固定版本、标签划分索引和 `data.json` SHA256 |

固定源文件 SHA256：

```text
train.jsonl  d411780d8c0e18e166f5664c6cfe90dc9de399d722aa7cde282e31a771323ea7
test.jsonl   fb1b0043ded745b8767687084786e6dd0a5f0ce03243b6131992a1c7ae2c2595
```

### 5.3 训练后需要保存什么

保留 `requirements-local.txt`、`data.json`、`manifest.json`、配置、训练日志、模型权重、tokenizer 和测试预测。普通 control 的部署入口是 `best/backbone/`；semantic 则需要完整 `best/` 和 `prototype_model.py`。

代码随本目录提供；数据和原始预训练权重由第 2 步下载；微调权重需自行训练。
