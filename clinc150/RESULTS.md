# CLINC150 实测结果

结果来自固定划分的完整 150 类闭集评测；以下数值由程序从预测文件生成。

| 方法 | 预测方式 | Accuracy 均值 | 种子标准差 | macro-F1 | NLL | ECE |
|---|---|---:|---:|---:|---:|---:|
| frozen | classifier_only | 78.074% | 0.151 pp | 0.77879 | 1.62269 | 0.45138 |
| ce | classifier_only | 97.015% | 0.100 pp | 0.97009 | 0.13868 | 0.01308 |
| ce | selected | 96.956% | 0.156 pp | 0.96947 | 0.16542 | 0.01684 |
| ce | calibrated_selected | 96.956% | 0.156 pp | 0.96947 | 0.13246 | 0.00860 |
| ce | retrieval | 97.030% | 0.084 pp | 0.97020 | 0.13200 | 0.00897 |
| ce | classifier_excluding_train_duplicates | 97.058% | 0.100 pp | 0.97053 | 0.13567 | 0.01265 |
| ce | selected_excluding_train_duplicates | 96.999% | 0.156 pp | 0.96991 | 0.16229 | 0.01646 |
| ce | retrieval_excluding_train_duplicates | 97.073% | 0.084 pp | 0.97064 | 0.12962 | 0.00865 |
| supcon | classifier_only | 97.326% | 0.046 pp | 0.97317 | 0.13257 | 0.01353 |
| supcon | selected | 97.311% | 0.038 pp | 0.97300 | 0.16865 | 0.03576 |
| supcon | calibrated_selected | 97.311% | 0.038 pp | 0.97300 | 0.12631 | 0.00884 |
| supcon | retrieval | 97.230% | 0.134 pp | 0.97219 | 0.16402 | 0.00779 |
| supcon | classifier_excluding_train_duplicates | 97.362% | 0.056 pp | 0.97354 | 0.12985 | 0.01319 |
| supcon | selected_excluding_train_duplicates | 97.347% | 0.046 pp | 0.97336 | 0.16618 | 0.03569 |
| supcon | retrieval_excluding_train_duplicates | 97.273% | 0.134 pp | 0.97263 | 0.16169 | 0.00739 |

classifier_only 是训练所得分类头；selected 使用验证集选择的温度及训练集原型混合。
原型融合收益与训练目标收益必须分别解释。统计检验见 summary.json。

这些结果不等于已完成榜单提交；正式上榜需要平台回执。
