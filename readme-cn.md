# DeepOpen

[English](README.md) | **简体中文**

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-lockup-dark.png" />
    <img src="assets/logo-lockup.png" alt="DeepOpen" width="330" />
  </picture>
</p>

**面向意图分类的框架与非自回归结构化决策 SDK。**

DeepOpen 根据预先定义的问题，对文本或 JSON 返回标签、分数和概率，不生成自由文本。输出类型受约束，但判断仍可能出错，概率也需要在实际业务数据上验证。

DeepOpen 整合决策 SDK、数据集专用训练策略，以及 Banking77 和 CLINC150 的可复现评测。上游基础与本仓库贡献详见[资源与来源说明](#资源与来源说明)。

## 榜单成绩

| 数据集 | DeepOpen 测试准确率 | 配置 | 正确数 / 总数 |
|---|---:|---|---:|
| **BANKING77** | **94.1234%** | 单分类器，SAM seed 13 | 2,899 / 3,080 |
| **CLINC150** | **98.0222%** | 四编码器集成 | 4,411 / 4,500 |

[在 Hugging Face 查看两份榜单对比、逐条预测及评测证据](https://c41-laya-evaluation.static.hf.space/index.html?version=ensemble-2#comparison)。

### BANKING77 对比

| 方法 | 测试准确率 | 结果来源 |
|---|---:|---|
| SPACE | 94.94% | 选取的公开参考成绩 |
| SPACE 2.0 | 94.77% | 选取的公开参考成绩 |
| Llama 2 7B (FT) | 94.35% | 选取的公开参考成绩 |
| ConvFiT | 94.16% | 选取的公开参考成绩 |
| **DeepOpen（SAM，观察到的最高成绩）** | **94.1234%** | 作者评测，提供保存的预测 |
| PPTOD | 93.86% | 选取的公开参考成绩 |

### CLINC150 对比

| 方法 | 测试准确率 | 结果来源 |
|---|---:|---|
| TOATOD-small | 98.45% | 选取的公开参考成绩 |
| **DeepOpen（集成）** | **98.0222%** | 作者评测，提供保存的预测 |
| TOATOD-base | 98.01% | 选取的公开参考成绩 |
| SPACE | 97.89% | 选取的公开参考成绩 |
| SPACE 2.0 | 97.80% | 选取的公开参考成绩 |

以上复列 [HF 评测页](https://c41-laya-evaluation.static.hf.space/index.html?version=ensemble-2#comparison)选取的参考成绩（参考快照：2026 年 10 月 7 日），不代表官方榜单排名。BANKING77 展示的是三个 SAM 种子在测试后选出的最高成绩；三种子均值为 **94.0368%**，相对重放基线尚未证实有统计显著提升。CLINC150 仅评测 **150 类范围内意图**，不含 OOS 检测，使用一个基于 Laya 的专用模型与三个 DeBERTa-v3-large 集成。完整协议、执行差异说明和预测文件见链接页面。

## 安装

需要 **Python 3.10 或更新版本**和 Git。从本仓库安装；此安装方式不依赖名为 `deepopen` 的 PyPI 发行包。

```bash
python -m pip install "git+https://github.com/deepopen-com/deepopen.git"
python -c "import deepopen; print(deepopen.__version__)"
```

仅使用 CPU 时，先安装 CPU 版 PyTorch，再安装 SDK：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "git+https://github.com/deepopen-com/deepopen.git"
```

使用 GPU 时，请安装与驱动和 CUDA 环境兼容的 PyTorch。打榜训练脚本有独立的依赖和运行说明，请查阅对应目录。

## 快速上手

以下示例加载上游英文检查点，完成一个分类问题。首次运行会下载权重并初始化模型，耗时不能当作预热后的推理延迟。

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

结果包含 `choice`、`probabilities` 和 `confidence`。实际输出取决于检查点和输入，高置信度不代表一定正确。GPU 环境可改为 `device="cuda"`；也可以用本地检查点目录替换 Hub 模型 ID。

## 路由与模型加载

```python
from deepopen import Router

router = Router(device="cpu")
# 只查看路由决定，不下载权重、不执行推理。
print(router.route({"body": "Please refund the duplicate charge."}, questions))

# 第一次预测时加载选中的检查点。
result = router.predict({"body": "Please refund the duplicate charge."}, questions)

# 适合对应工作流时，可显式指定任务微调检查点。
# result = router.predict(state, questions, model="typed-decisions")
```

语言路由基于文字系统和语言启发式规则，不保证选中准确率最高的模型。默认**不会自动选用 typed-decisions 检查点**；请显式指定，或在业务验证后设置 `auto_task_detection=True` 开启任务检测。

默认缓存只保留一个模型。频繁处理多语言请求时，可使用 `Router(max_loaded=2)`，再调用 `router.preload(["english", "multilingual"])`。`Router(preload=True)` 会加载全部三个模型，需要相应内存。预加载能减少重复加载，但不保证固定推理延迟。

| 路由键 | 默认 Hub 位置 | 编码器 | 默认上下文 |
|---|---|---|---:|
| `english` | [Laya 仓库根目录](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large | 512 |
| `multilingual` | [Laya / multilingual](https://huggingface.co/convaiinnovations/laya/tree/main/multilingual) | mmBERT-base | 1,024 |
| `typed-decisions` | [Laya / typed-decisions](https://huggingface.co/convaiinnovations/laya/tree/main/typed-decisions) | ModernBERT-large | 1,024 |

Python 包名为 `deepopen`；上游模型 ID 保留 `laya`，与加载器配置一致。

## 决策类型与工作流模板

| 类型 | 输出 | 示例 |
|---|---|---|
| `choice` | 标签、各选项概率、置信度 | 部门或意图分类 |
| `score` | 有序等级的期望值和分布 | 紧急度评分 |
| `noul` | 陈述为真的估计概率 | 是否请求退款 |

SDK 提供 `router_questions()`、`guard_questions()`、`moderation_questions()`、`triage_questions()` 和 `email_questions()`。这些是问题模板，不是生产安全保证。用于重要自动决策前，应验证准确率并校准置信度。

## 较早的意图分类实验

下表保留较早的本地评测与训练流程。上方 HF 主成绩还包含后续 BANKING77 SAM 继续训练结果（94.1234%）；旧的 94.0909% 使用不同训练及执行配置。它们不代表已经获得官方榜单排名。模型经过任务专用训练，安装 SDK 不会自动得到这些成绩。

| 数据集 | 配置 | 测试准确率 | 评测范围 |
|---|---|---:|---|
| Banking77 | Laya 编码器 + 分类头，训练 7 轮后继续训练 4 轮 | **94.0909%** | 3,080 条测试样本，77 类 |
| CLINC150 | 验证集选出的 R-Drop 单模型 | **97.7556%** | 4,500 条范围内测试样本，150 类 |
| CLINC150 | Laya + 三个 DeBERTa-v3-large | **98.0222%** | 四模型集成；相同范围内测试集 |

CLINC150 成绩不包含范围外意图检测（OOS）。Banking77 的主成绩来自普通分类头继续训练，不能归因于原型分支。集成系统有四个编码器，不能用下面单检查点的延迟代表其推理成本。

- [Banking77 复现指南](banking77/README.md)：训练、评测、来源及替代实验分支。
- [CLINC150 复现指南](clinc150/README.md)：单模型与集成协议、随机种子和评测范围。

## 上游检查点评测与限制

[基准报告](BENCHMARKS.md)记录了不同检查点的独立实验。不要将下列各行拼成同一个配置的综合性能。

| 指标 | 配置与条件 | 历史结果 |
|---|---|---:|
| 单问题延迟 | 多语言检查点，预热后推理，NVIDIA T4 | 32.8 毫秒 |
| 十问题延迟 | 多语言检查点，单次批处理，NVIDIA T4 | 总计 72.3 毫秒；每问题 7.2 毫秒 |
| Typed-decisions 准确率 | 任务微调检查点；400 个案例、2,000 个决策 | 0.766 |
| 重新拟合温度后的平均 ECE | 英文基础检查点，校准实验 | 0.081 |
| 多语言覆盖评测 | MASSIVE 的 51 种语言，20 个选项 | 45 / 51 超过随机准确率的 3 倍 |

编码器声称覆盖 100 多种语言，不代表所有语言具有同等质量。基础检查点在 typed-decisions 零样本任务上接近随机水平，低于多数类基线。两个基础检查点出厂时均存在过度自信，温度拟合必须使用独立校准集。报告中的留出内容审核实验准确率仅为 0.530。大量选项会争用提示词预算，有序等级评分也是当前弱项。

历史报告中的 Jev 数据来自第三方公开结果，提示词和样本量不同，只能作为参考，不能视为严格同条件对比。本地部署免除按 Token 支付的 API 费用，但仍有硬件、电力和运维成本。

## 资源与来源说明

- [完整基准报告](BENCHMARKS.md)与[研究脚本](research/README.md)。
- [T4 基准测试 Notebook](research/scripts/laya_benchmark_colab.ipynb)。
- [Typed-decisions 微调 Notebook](notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)。
- [上游 Laya 源码](https://github.com/NandhaKishorM/laya)与[预训练权重](https://huggingface.co/convaiinnovations/laya)。

SDK 和基础模型架构源自 Laya。本仓库提供 DeepOpen 包装和文档、Banking77 适配实验，以及 CLINC150 的 R-Drop/SupCon 和异构集成复现流程。固定版本的源码来源见 [Banking77 来源清单](banking77/SOURCE_MANIFEST.json)与 [CLINC150 来源清单](clinc150/SOURCE_MANIFEST.json)。SDK 默认继续加载上游检查点；任务专用模型的训练与导出请参阅实验指南。

## 开发与检查

```bash
python -m pip install -e .
python tests/test_router.py
python tests/test_criteria.py
```

上述测试检查路由和输入处理，不下载模型权重。`tests/test_local_e2e.py` 还需要本地检查点，文件开头说明了目录布局。CI 同时检查代码、构建安装包并检查已安装的 wheel。SDK 版本号统一定义在 `deepopen/_version.py`。

## 许可证

[Apache License 2.0](LICENSE)。请保留上游署名，并根据使用场景查阅模型与数据集各自的许可证。
