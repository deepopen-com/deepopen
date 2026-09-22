# Laya × Banking77 实验结果

最终模型 test accuracy **93.4416%**，macro F1 **93.4214%**，官方测试集 3080 条，77 类。

## 方法与数据

- 使用英文 Laya / ModernBERT-large（原有 encoder、typed decision head、option scorer），没有替换成普通分类器。
- 以交叉熵监督微调全部分类路径；原 action head 保留但不训练。此实验不是 RLCD 复现实验，未重新做概率校准。
- 官方 train 10,003 条、test 3,080 条。训练内部固定分层划分 85%/15%，split seed 2026；所有实验共享划分。
- 以验证 accuracy、macro F1 依次选型；选定后以最佳 epoch 数从原 checkpoint 在全部 10,003 条上重训，测试集不参与选型。
- 类别描述使用下划线替换为空格，固定排序；根据 tokenizer 精确计算选项预算并保留全部 77 类。
- 官方 train/test 完全文本重叠数：0；保留官方划分，不据测试内容改动训练。
- GPU：NVIDIA L20 46 GB × 2；PyTorch 2.9.1+cu128，BF16，梯度裁剪 1.0，线性 warmup/decay。

## 测试结果

| 模型 | Test accuracy | Test macro F1 |
|---|---:|---:|
| laya_default | 45.3896% | 40.8799% |
| laya_full_options | 57.2403% | 55.1850% |
| laya_finetuned | 93.4416% | 93.4214% |

默认预算与完整选项预算两项使用同一未微调英文权重，均在最终选型后统一测试。

## 调参记录

| 实验 | 学习率 | Batch | WD | Warmup | Seed | 最佳 epoch | 验证 accuracy | 验证 macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lr2e5 | 2e-05 | 32 | 0.01 | 0.06 | 42 | 5 | 90.2065% | 90.1897% |
| regularized_b64 | 5e-05 | 64 | 0.1 | 0.1 | 42 | 5 | 90.0067% | 89.7380% |
| seed123 | 5e-05 | 32 | 0.01 | 0.06 | 123 | 7 | 91.6056% | 91.5479% |
| lr5e5 | 5e-05 | 32 | 0.01 | 0.06 | 42 | 7 | 90.8728% | 90.5877% |

选定 `seed123`，全量重训 7 epoch。验证差异可能有随机波动；当前没有宣称统计显著或当前榜单 SOTA。

## 已发表基线（外部参考，非本机复跑）

| 模型 | 全训练集 Banking77 test accuracy |
|---|---:|
| BERT-FIXED | 87.19% |
| BERT-TUNED | 93.66% |
| ConveRT | 93.01% |
| USE+ConveRT | 93.36% |

来源：[Casanueva et al., 2020, Table 3](https://aclanthology.org/2020.nlp4convai-1.5.pdf)。编码器、调参预算和随机设置不同，属于文献参考。未提交任何外部排行榜。

## 复现

工作目录 `/opt/laya-banking77`。训练必需依赖的精确版本在 `requirements-training.txt`，完整环境快照在 `requirements-lock.txt`；`environment.json` 保存硬件和输入指纹。上游源码 commit `42626c348753fbb17572a813127df2278a1ec527`，Laya 模型 revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`，数据 revision `18072d2685ea682290f7b8924d94c62acc19c0b2`。

```bash
cd /opt/laya-banking77
# 重建环境时运行 bash setup.sh；可用 LAYA_PYTHON / LAYA_UV 指定 Python 3.12 和 uv。
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 USE_TF=0
# 从原模型、已保存固定划分重新运行选定配置：
CUDA_VISIBLE_DEVICES=0 venv/bin/python banking77.py train --name reproduced --lr 5e-05 --batch 32 --micro 16 --wd 0.01 --warmup 0.06 --seed 123 --epochs 8
# 最终全量训练配置（请使用新名称，以保留原结果）：
CUDA_VISIBLE_DEVICES=0 venv/bin/python banking77.py train --name reproduced_full --full-train --lr 5e-05 --batch 32 --micro 16 --wd 0.01 --warmup 0.06 --seed 123 --epochs 7
# 独立复算已保存预测的指标：
venv/bin/python audit_metrics.py
# 从保存 checkpoint 重新推理完整测试集，并校验原生 SDK 的标签映射：
CUDA_VISIBLE_DEVICES=0 venv/bin/python evaluate_checkpoint.py --checkpoint best_checkpoint --output evaluation/reproduced.json
```

GPU 浮点和库实现可能造成微小数值差异，未承诺跨硬件逐位一致。

## 产物

- `best_checkpoint/`：最终全量训练模型，可由 `laya.load()` 加载。
- `runs/*/config.json`, `history.json`, `done.json`, `best/`：每次实验配置、逐轮指标、权重。
- `selection_before_test.json`：首次测试前选定的配置。
- `manifest.json`, `data.json`：版本、指纹、索引、原始数据及标签映射。
- `evaluation/*predictions.json`, `classification_report.json`：逐例预测和分类型指标。
- `logs/`：安装、下载、全部训练日志；`results.json`：汇总指标。

推理示例：
```python
import json, laya
labels=json.load(open('/opt/laya-banking77/data.json'))['labels']
agent=laya.load('/opt/laya-banking77/best_checkpoint',device='cuda')
question={'intent':{'type':'choice','instructions':'Which banking intent does `message` express?',
    'criteria':{label.replace('_',' '):None for label in labels}}}
print(agent.predict({'message':'My card has been stolen'},question))
```

## 追加对照：Laya encoder + 固定 77 类分类头

此变体复用原 Laya encoder 权重，新增 Transformers ModernBERT 分类头；不是原生 Laya decision head，也不由 laya.load() 加载。

| 架构 | 最佳验证 accuracy | 全量重训 test accuracy | Test macro F1 |
|---|---:|---:|---:|
| 原生 Laya | 91.6056% | 93.4416% | 93.4214% |
| Laya encoder + 77 类分类头 | 91.2725% | 93.7987% | 93.8050% |

测试前按验证集选定的推荐架构：**native_laya**。选择记录在 `overall_selection_before_test.json`。两个变体的测试结果均保留，未据测试结果继续调参。

分类头实验配置与曲线在 `classifier_runs/`；对照学习率 2e-5/5e-5，head 学习率为 encoder 的 10 倍，batch 32，weight decay 0.01，warmup 0.1，max_length 128，seed 42。最优配置用全训练集重训 7 epoch。

分类头 checkpoint：`classifier_runs/encoder_full_train/best/`，用 `AutoModelForSequenceClassification.from_pretrained()` 和 `AutoTokenizer.from_pretrained()` 加载。

```bash
CUDA_VISIBLE_DEVICES=0 venv/bin/python classifier.py test --checkpoint classifier_runs/encoder_full_train/best
```

独立检查：原生 checkpoint 已重新加载、全测试集再次推理，且前 20 条原生 SDK 输出与批量评估标签一致；参见 `evaluation/reproduced.json`。
