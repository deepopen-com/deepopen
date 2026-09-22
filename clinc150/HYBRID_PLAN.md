# 异构编码器对照与 Laya 混合实验

本实验在第四轮 Laya 原生重排器训练期间提出，此时尚未获得第四轮测试成绩。
主目标仍是 CLINC150 full 原划分的测试准确率超过 98%。
引入不同预训练编码器必须单独归因，不能把换底座或增加模型的收益称为 Laya 单模型创新。

固定公开通用预训练模型 microsoft/deberta-v3-large，revision
64a8c8eab3e352a784c658aef62be1662607476f；按官方仓库元数据校验所有下载文件。
它是通用预训练 checkpoint，不使用已有 CLINC150 微调模型。
仍只使用原 15000 train 梯度训练、3000 validation 选择、4500 test 最终评估，排除 OOS。

先导 seed 42：均值池化、dropout .1、150 类线性头，CE + .1 SupCon。
batch 16、累积 4、最大长度 64，8 epochs，encoder LR 1e-5、head LR 1e-4，
AdamW weight decay .01，warmup 10% 后线性衰减，梯度裁剪 1，BF16。
若验证选择的独立或混合候选优于 98.4% 的现有 Laya 召回模型，补训 seeds 13/87。

比较独立 DeBERTa 对照与固定 Laya rdrop_42 的概率融合，
Laya 权重 alpha={0,.25,.5,.75,1}，只按验证 accuracy、再按 NLL 选择。
若补训三种子，另比较固定三种子 DeBERTa 等权概率集成与同一 Laya 的融合。
测试前冻结全部配置及代码/数据/权重 SHA256。
报告原 Laya、DeBERTa、Laya+DeBERTa 混合三种结果及三种子变化，
若最终选中 alpha=0，则明确称为 DeBERTa 对照，不能称为 Laya 改进。

第四轮两阶段 Laya 实验保持独立冻结与报告，不按混合实验测试成绩更改其选择。
本轮是已知历史测试汇总后的后续研究，不是新的盲测或官方榜单接收。
