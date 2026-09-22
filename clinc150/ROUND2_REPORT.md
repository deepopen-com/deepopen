# Laya / CLINC150 第二轮改进实验

本轮在知晓第一轮测试汇总成绩后继续开发。只用训练/验证数据设计方法和选择模型，
没有根据第一轮测试错误逐条调参。它仍是同一测试集上的后续研究，不等价于独立盲测。

## 主要结果

| 模型 | 测试准确率 | 说明 |
|---|---:|---|
| 第一轮 SupCon seed 87 | 97.3778% | 第一轮验证选出的改进单模型 |
| 第二轮验证选出的单模型 | 97.7556% | `artifacts/round2/rdrop_42` |
| 第二轮验证选出的集成 | 97.7333% | `rdrop3`，3 个完整模型 |

单模型相对第一轮 +0.3778 个百分点；配对修正 37 条、退步 20 条。
配对 bootstrap 95% 区间 [0.0444, 0.7111] 个百分点；
双侧 exact McNemar p=0.033144，未做多重比较校正。
集成相对第一轮 +0.3556 个百分点；不能把集成收益称为单模型收益。

## 验证候选（包括负结果）

| 候选 | 验证准确率 | 验证 NLL |
|---|---:|---:|
| rdrop_42 | 98.4000% | 0.08819 |
| rdrop_13 | 98.3667% | 0.08655 |
| rdrop_87 | 98.3000% | 0.09025 |
| soup_all6 | 98.0333% | 0.13204 |
| soup_paired_42 | 97.9667% | 0.10635 |
| soup_paired_87 | 97.9333% | 0.10167 |
| balanced_supcon_87 | 97.9000% | 0.10430 |
| soup_supcon3 | 97.8667% | 0.12175 |
| balanced_supcon_13 | 97.8333% | 0.11230 |
| balanced_supcon_42 | 97.8000% | 0.10272 |
| soup_paired_13 | 97.5667% | 0.10435 |

固定等权集成的验证结果：

| 集成 | 模型数 | 验证准确率 |
|---|---:|---:|
| rdrop3 | 3 | 98.4667% |
| supcon3 | 3 | 98.1667% |
| all6 | 6 | 98.1000% |
| balanced_supcon3 | 3 | 98.1000% |
| ce3 | 3 | 98.0667% |

## 复现实验与解释

- 数据：CLINC150 150 类闭集；15000 train / 3000 validation / 4500 test；OOS 不计入。
- `round2.py diagnose` 分析验证错误互补；`soups` 对预先指定的权重组合做等权平均。
- `balanced_supcon` 每批最多 16 类，以四个同类样本为一个块；每轮每样本恰好一次。
- `rdrop` 对编码器设置 0.1 dropout，两个随机前向的 CE、SupCon 取均值，并加入系数 1 的对称 KL。
- 两种训练都从相同 Laya 原始权重开始；8 epochs，encoder LR 2e-5，head LR 2e-4，batch 64。
- GPU 0 释放后，在本轮测试前将两个训练候选都扩展为三个种子；完整保留两家族结果。
- R-Drop 每步做两个前向，训练计算量更多，不能声称完全等计算量的纯算法比较。
- R-Drop 候选同时修改编码器 dropout 和一致性目标，本轮未分离这两项贡献，不能把全部收益归因于 KL。
- 分类头在不同 seed 的初始化不同，跨 seed 权重平均可能失败，因此保留所有固定平均组合的验证成绩。
- `round2_finish.py freeze` 保存选择规则和权重 SHA256，随后独立 `evaluate` 测试。
- 发布单模型位于服务器 `/opt/laya-clinc150/artifacts/round2/release_single`。
- 集成依赖冻结清单所列的多个真实模型；推理需多次编码，不具有单模型成本。
- 不修改原始标注；验证集存在意图含糊或疑似标注问题，仅用于解释，不删除以抬高主指标。
- 基础模型预训练数据未完全披露，不能保证上游从未见过本基准。
- 正式榜单提交尚未完成；本报告分数为自主评测。

## 复现命令

```bash
cd /opt/laya-clinc150
CUDA_VISIBLE_DEVICES=1 venv-final/bin/python infer_clinc.py --checkpoint artifacts/round2/release_single --text "what is my bank balance"
CUDA_VISIBLE_DEVICES=1 venv-final/bin/python infer_ensemble.py --text "what is my bank balance"
```

## 方法来源

- SupCon: https://arxiv.org/abs/2004.11362
- R-Drop: https://arxiv.org/abs/2106.14448
- Model soups: https://arxiv.org/abs/2203.05482
以上是已有研究方法，本项目不声称其为原创。

## 三种子复验

balanced_supcon: 测试均值 97.1778% ± 0.2120 个百分点（seed 标准差）；相对第一轮 SupCon 均值变化 -0.1481 个百分点。
rdrop: 测试均值 97.6593% ± 0.0898 个百分点（seed 标准差）；相对第一轮 SupCon 均值变化 +0.3333 个百分点。

| seed | 第一轮 SupCon | 成组 SupCon | R-Drop + SupCon |
|---|---:|---:|---:|
| 13 | 97.3111% | 97.2889% | 97.6444% |
| 42 | 97.2889% | 96.9333% | 97.7556% |
| 87 | 97.3778% | 97.3111% | 97.5778% |

R-Drop 三种子平均增益的条件配对 bootstrap 95% 区间：[0.1037, 0.5556] 个百分点。
该区间条件于已训练的三个模型，不覆盖所有训练随机性。

## 实际推理与交付

NVIDIA L20 上，单模型热启动 p50 为 10.104 ms，三模型集成为 30.670 ms。
口径为 batch 1、padding 64、BF16 autocast，20 次预热后测 100 次；不含加载、分词和网络。
新单模型为 394,935,446 个参数，没有增加第一轮模型的参数量。真实模型加载和任意文本推理已通过检查。
代码、图表、统计和模型元数据在本地；完整 checkpoint 保存在服务器，证据压缩包不包含大权重。

## 本轮结论

R-Drop + SupCon 在三个种子上均提高测试准确率。成组 SupCon 测试均值下降，属于负结果。
等权集成测试分数比新单模型少判对一条，成本约三倍，因此建议部署单模型；一条差异不代表显著优劣。
seed 42 曾因共享 GPU 0 吞吐降低而中断，日志保留在 aborted_rdrop_42_shared_gpu0；随后在 GPU 1 按相同配置重启。
这是一项可复现的专项改进结果，不足以单独支持原创方法或当前 SOTA 声明。