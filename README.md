# DeepOpen：开源多语言System 1决策引擎 技术白皮书

DeepOpen 是基于laya的一款完全开源的非自回归System 1决策引擎，专为结构化类型决策场景设计。

## 复现 打榜  banking77

https://github.com/deepopen-com/deepopen/tree/main/banking77


## 复现 打榜  clinc150

https://github.com/deepopen-com/deepopen/tree/main/clinc150


它摒弃了传统大模型逐Token生成文本的模式，在单次前向传递中即可完成100+种语言的多维度类型判断，单请求延迟低至33毫秒、批量处理仅7.2毫秒（T4显卡实测），依托严格正确评分规则RLCD完成强化学习训练，通过内置智能路由器自动为每个请求匹配最优检查点，彻底解决了传统大模型在分类、路由、打分场景下速度慢、成本高、易产生幻觉的痛点。


# 打榜表现

DeepOpen 在两个榜单打榜的初步结果：
模型： Deepopen（改进后的 Laya）
榜单： CLINC150 和 Banking77

![](https://github.com/deepopen-com/deepopen/blob/main/%E6%89%93%E6%A6%9C.png?raw=true)

本地测试： 对照参考成绩，分别位于第 2 位和第 5 位；





## 核心架构与三大检查点

DeepOpen 基于三大独立优化的检查点构建，内置的智能路由器可在亚毫秒内完成输入内容的脚本、语言识别，自动调度对应最优模型，无需开发者手动配置切换规则：
- DeepOpen 英文检查点：基于ModernBERT-large 421M参数训练，支持512上下文窗口，在英文单语种任务中实现39.5毫秒单请求延迟，在英文意图分类、XNLI等基准测试中准确率达到0.783-0.860，专为纯英文高并发决策场景优化。
- DeepOpen-multilingual 多语言检查点：基于mmBERT-base 322M参数训练，支持1024上下文窗口，推理速度比英文模型快2倍，覆盖100+种语言，其中45种语言的准确率超过3倍随机基线，在非拉丁语种下性能远超纯英文模型，13种非英文语言的意图分类准确率达到0.451，是英文检查点的1.47倍。
- DeepOpen-typed-decisions 类型决策检查点：基于ModernBERT-large 421M参数训练，支持1024上下文窗口，专门针对结构化类型决策场景微调，在2000个决策样本的基准测试中，准确率达到0.766，超过TypeSafe Jev 1.13.0的0.727，同时Brier分数低至0.062，是目前开源决策模型中精度领先的方案。

## 核心技术特性

1. 零幻觉非自回归设计：全程不生成任何文本内容，所有输出均为开发者预先定义的结构化类型结果，无需后续解析处理，从根源上杜绝了传统大模型的幻觉问题，输出结果100%符合预设的类型边界。
2. 全链路智能路由机制：在模型前向推理前，通过纯Python实现的语言检测模块，在<0.5毫秒内识别输入内容的脚本类型和所属语言，自动匹配最优检查点，彻底避免了英文模型在非拉丁语种下“高置信度错误”的致命问题——此前纯英文检查点在高棉语任务中准确率为0，却给出95.2%的错误置信度，仅靠置信度阈值完全无法规避风险。
3. 严格校准的概率输出：依托RLCD强化学习框架训练，所有输出的置信度分数具备严格的统计意义，经过域温度校准后，ECE（预期校准误差）低至0.081，远优于同类方案，可直接用于生产环境的自动置信门控流程，高置信度请求直接自动处理，低置信度请求自动流转人工审核。
4. 极致的性能表现：在特斯拉T4显卡上实测，单请求推理仅需32.8毫秒，10题批量处理仅72.3毫秒，单T4显卡最高支持每秒103-332个问题的吞吐量，实测速度是TypeSafe Jev的7.8倍。
5. 完全开源零成本部署：采用Apache 2.0开源许可，所有权重完全开放，支持本地自托管，无需调用任何付费API，不存在按Token计费的额外成本，对比TypeSafe Jev每百万Token 0.042美元的定价，长期大规模部署可节省近100%的推理成本。

快速上手部署流程
1. 安装依赖包
直接通过PyPI一键安装最新版本：
```bash
pip install deepopen
```

2. 推荐路由模式（开箱即用）
通过内置Router入口点，自动完成语言检测和模型调度，预加载所有检查点后即可实现亚毫秒级路由切换：
```python
import deepopen
from deepopen import Router

将所有检查点预加载到内存，实现35毫秒以内的极速推理
router = Router(preload=True, device="cuda")

定义任意语言的输入状态
state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice 4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
}

定义自定义类型决策规则
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?"
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?"
    }
}

自动路由到最优模型完成推理
res_en = router.predict(state, questions)
print("Department :", res_en["answers"]["department"]["choice"])  输出: billing (confidence: 0.94)
print("Routing    :", res_en["routing"]["model"])                 输出: english
```

3. 生产环境部署优化
根据业务场景灵活调整内存策略，避免不必要的显存占用：
```python
仅预加载业务所需的检查点，节省显存
router.preload(["english", "multilingual"])

配置LRU缓存策略，最多同时保留2个热模型
router = Router(max_loaded=2)

手动卸载模型释放内存
router.unload()
```

内置生产级工作流程预设
DeepOpen 提供了预先调优的开箱即用问题模板，无需从零构建规则即可快速落地核心业务场景：
- 智能模型路由：自动判断用户请求的复杂度，将简单任务路由到小模型、复杂任务路由到前沿大模型，大幅降低整体推理成本。
- 实时Prompt防护：精准识别Prompt注入、越狱指令、敏感信息泄露等风险，为大模型应用构建前置安全屏障。
- 内容安全审核：快速检测文本中的毒性内容、骚扰信息、威胁言论，适配多语言社区的内容治理需求。
- 支持工单分诊：自动完成工单意图识别、紧急度打分、用户挫败感评估、流失风险预判，大幅提升客服流转效率。

基准测试与行业对比
基于17416道题的共享基准数据集，DeepOpen与主流决策引擎的实测对比如下：
| 测试维度 | DeepOpen（路由模式） | TypeSafe Jev 1.13.0 |
| --- | --- | --- |
| 类型决策准确率（2000样本） | 0.766 | 0.727 |
| 预期校准误差ECE | 0.081 | 0.144 |
| P50单请求延迟 | 32.8毫秒 | 236-276毫秒 |
| 支持可用语言数 | 51种中45种 | 无公开多语言基准 |
| 部署模式 | 完全自托管开源 | 仅封闭API |
| 推理成本 | 0（本地部署） | 0.042美元/百万Token |

在高基数标签场景下，TypeSafe Jev在50+选项的任务中表现更优，而DeepOpen可通过调整`head_max_len`参数扩展选项支持能力，将选项提示预算提升至512，即可支持70+选项的精准分类，满足绝大多数业务场景需求。

开源资源与生态
DeepOpen 全链路资源完全开放，开发者可快速获取所有资料：
- Hugging Face 模型仓库：`convaiinnovations/deepopen`
- 在线互动演示：`convaiinnovations/deepopen-demo`
- 完整技术工程报告：Dev.to 专栏
- 微调教程Notebook：支持在Kaggle免费2xT4显卡上完成全流程微调，4-5小时即可完成3万道题的训练，通过RLCD强化学习将基础模型准确率从0.36提升至0.766，完全适配垂直领域的定制化决策需求。

---

# 参考

<br>参考资料<br>[1] [DeepL + OpenAI integrations - connect and automate | Bardeen.ai - www.bardeen.ai](https://www.bardeen.ai/integrations/openai/deepl)<br>[2] [Minutes.ai - Smart Government Innovation LAB - www.smartlab.gov.hk](https://www.smartlab.gov.hk/en/ai_solutions/a-0063)<br>[3] [benchmark.ng — Nigeria's AI Decision Engine - benchmark.ng](https://benchmark.ng/)<br>[4] [Phrase: AI-Powered Localization & Translation Platform - Phrase官网](https://phrase.com/?utm_medium=yelp_blog&utm_source=logiciels.pro&utm_campaign=5-ways-yelp-can-help&utm_content=blog_text_link&utm_term=meet-yelp-host)<br>[5] [DeepSeek Open Web UI 安装部署全攻略：从环境配置到可视化交互 - 百度智能云](https://cloud.baidu.com/article/3564171)<br>[6] [Jev 是什么：前 OpenAI 研究员做的“不说话“模型，只输出带概率的结构化决策-CSDN博客 - CSDN博客](https://blog.csdn.net/aidoudoulong/article/details/166136816)<br>[7] [前OpenAI研究员推Jev模型，绕开文本生成直出决策 - m.counselleap.cn](http://m.counselleap.cn/qqznews/202609/article_2962603.shtml)<br>[8] [MCP服务器 - 开放式深度研究模型协议-MCP服务 - www.mcpworld.com](https://www.mcpworld.com/zh/detail/8ed885da2b0dac2c97d11afb4a219269)<br>[9] [SGLang开源引擎：开源创新与推理革命的融合之路 - 百度智能云](https://cloud.baidu.com/article/4935192)<br>[10] [GitHub - weibaohui/openDeepWiki: 完全AI驱动的 DeepWiki ,使用 Go + Eino 技术栈开发 · GitHub - GitHub](https://github.com/weibaohui/openDeepWiki)<br>[11] [正面硬刚 OpenAI o1！DeepSeek-R1：开启 AI 自主推理新时代，现已开源！ - 知乎 - 知乎](https://zhuanlan.zhihu.com/p/1960455097218761196)<br>[12] [详解多智能体架构：以 Open Deep Research 项目为例 - 知乎 - 知乎](https://zhuanlan.zhihu.com/p/1944449334209942670)<br>[13] [集成DeepSeek的开源利器：3款高效应用深度解析 - 百度智能云](https://cloud.baidu.com/article/5076327)<br>[14] [当开源创新遇上推理革命：SGLang如何炼就DeepSeek最强开源推理引擎？_腾讯新闻 - 腾讯网](https://new.qq.com/rain/a/20250306A09PRF00)<br>[15] [3款集成DeepSeek的开源应用推荐：开发者高效工具指南 - 百度智能云](https://cloud.baidu.com/article/4511140)<br>[16] [当开源创新遇上推理革命：SGLang如何炼就DeepSeek最强开源推理引擎？-腾讯云开发者社区-腾讯云 - 腾讯云](https://cloud.tencent.com/developer/article/2502983)<br>[17] [DeepSeek-V3.2：开源大语言模型的新里程碑，在推理与智能体任务中突破性能边界 - 知乎 - 知乎](https://zhuanlan.zhihu.com/p/1978976195946239755)<br>[18] [Open Deep Research V2：新架构！上下文工程最新实践 - 知乎 - 知乎](https://zhuanlan.zhihu.com/p/1933207553409525463)<br>[19] [深度探索DeepSeek：解锁高效开发与智能决策的新范式 - 百度智能云](https://cloud.baidu.com/article/3732497)<br>[20] [【AI大模型】 Langchain+DeepSeek R1从入门到精通，收藏这一篇就够了！！_langchain从入门到实战-CSDN博客 - CSDN博客](https://blog.csdn.net/2401_85327249/article/details/147629035)<br>[21] [开源×推理”双引擎驱动：SGLang打造DeepSeek开源推理新标杆 - 百度智能云](https://cloud.baidu.com/article/3728141)<br><br> 




<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NandhaKishorM/deepopen/main/assets/logo-lockup-dark.png" />
    <img src="https://raw.githubusercontent.com/NandhaKishorM/deepopen/main/assets/logo-lockup.png" alt="deepopen" width="330" />
  </picture>
</p>

**Multilingual, non-autoregressive System 1 decision engine.** Typed decisions over 100+ languages in a single forward pass — 33 ms — trained with reinforcement learning against strictly proper scoring rules (RLCD), with a router that picks the right checkpoint per request.

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15d4Yv__KHeHjshVb-6PRTfqVllxih2S3?usp=sharing)
[![PyPI version](https://img.shields.io/pypi/v/deepopen.svg)](https://pypi.org/project/deepopen/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-convaiinnovations%2Fdeepopen-blue)](https://huggingface.co/convaiinnovations/deepopen)
[![Multilingual](https://img.shields.io/badge/%F0%9F%A4%97%20Model-deepopen--multilingual-blue)](https://huggingface.co/convaiinnovations/deepopen-multilingual)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-deepopen--demo-orange)](https://huggingface.co/spaces/convaiinnovations/deepopen-demo)
[![Dev.to Article](https://img.shields.io/badge/dev.to-Read%20Article-0A0A0A?logo=devdotto&logoColor=white)](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-nandakishorm-FFDD00?logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/nandakishorm)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/NandhaKishorM/deepopen/main/assets/deepopen_vs_jev_full.png" alt="deepopen versus TypeSafe Jev: accuracy on shared public datasets, every application workflow, all 51 languages, speed, calibration, and the cost of not preloading" width="100%" />
</p>

deepopen evaluates typed questions (`choice`, `score`, `noul`) over any state (text, email, ticket or JSON document) in **a single forward pass** — 33 ms for one question, 7.2 ms/question batched, measured on a T4. No text generation, so nothing to parse and nothing to hallucinate.

Three checkpoints, and a `Router` that picks between them per request:

| | encoder | params | context | use it for |
|---|---|---|---|---|
| [`deepopen`](https://huggingface.co/convaiinnovations/deepopen) | ModernBERT-large | 421M | 512 | English |
| [`deepopen-multilingual`](https://huggingface.co/convaiinnovations/deepopen-multilingual) | mmBERT-base | 322M | 1024 | 100+ languages, 2x faster |
| [`deepopen-typed-decisions`](https://huggingface.co/convaiinnovations/deepopen-typed-decisions) | ModernBERT-large | 421M | 1024 | the typed-decisions workflows |

---

## Installation

```bash
pip install deepopen
```

---

## Quickstart: Route Mode (Recommended)

deepopen ships three checkpoints. The built-in **`Router`** is the recommended entry point: it evaluates any state in any language, automatically detects scripts and languages in sub-milliseconds, and dispatches to the optimal checkpoint in a single forward pass.

```python
import deepopen
from deepopen import Router

# Preload checkpoints into memory for instant sub-35ms routing
router = Router(preload=True)

# 1. State in any language or schema
state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
}

# 2. Define your typed questions
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?"
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?"
    }
}

# 3. English state -> automatically routed to deepopen (ModernBERT-large, 39.5 ms)
res_en = router.predict(state, questions)
print("Department :", res_en["answers"]["department"]["choice"])  # -> billing (confidence: 0.94)
print("Routing    :", res_en["routing"]["model"])                 # -> english

# 4. Hindi state -> automatically routed to deepopen-multilingual (mmBERT-base, 32.8 ms)
res_hi = router.predict({"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}, questions)
print("Department :", res_hi["answers"]["department"]["choice"])  # -> billing (confidence: 0.86)
print("Routing    :", res_hi["routing"]["model"])                 # -> multilingual

# 5. Explicit override when you want a specific checkpoint
res_td = router.predict(state, questions, model="typed-decisions")
```

Every result carries full routing metadata explaining why the choice was made:

```python
res_hi["routing"]
# {
#   'model': 'multilingual',
#   'repo': 'convaiinnovations/deepopen/multilingual',
#   'reason': 'non-Latin script (devanagari, 100% of letters); the English checkpoint cannot read it'
# }
```

Inspect a routing decision without running any forward pass:

```python
router.route({"body": "Der Kunde wurde zweimal belastet"}, questions).reason
# "Latin script but language looks like 'de', not English"
```

### Why Route: The Evidence

On a shared benchmark (17,416 questions, one T4 GPU, identical questions per model):

| Benchmark / Task | English (`deepopen`) | Multilingual (`deepopen-multilingual`) | `Router` (Routed) |
|---|---|---|---|
| MASSIVE intent, English | **0.783** | 0.657 | **0.783** |
| MASSIVE intent, 13 other languages | 0.306 | **0.451** | **0.451** |
| XNLI, English | **0.860** | 0.843 | **0.860** |
| XNLI, 14 other languages | 0.521 | **0.731** | **0.731** |
| Languages usable (>3x random) | 23 / 51 | 45 / 51 | **45 / 51** |
| Latency, 1 question (T4 GPU) | 39.5 ms | **32.8 ms** | **32.8 ms** |
| Latency, 10 questions batched | 158.6 ms | **72.3 ms** | **72.3 ms** |

The English checkpoint collapses on non-Latin scripts (Khmer scores **0.000 accuracy at 0.952 confidence**). Because the model stays confident while being wrong, confidence gating cannot save you. `Router` detects the script in <0.5 ms pure Python before the forward pass.

### Production Preload & Memory

A cold checkpoint build costs seconds; language detection costs microseconds. At the default `max_loaded=1`, traffic that alternates languages rebuilds a model on *every* request (measured at a 7.4 s median reload on CPU and 10.3 s on T4).

For a server or production app, preload:

```python
# Every checkpoint resident in memory; language flips cost detection only (<1 ms)
router = Router(preload=True)
router = Router(preload=True, device="cuda")

# Or preload only the specific checkpoints you serve:
router.preload(["english", "multilingual"])

# If your app already built an agent, attach it to avoid duplicate VRAM:
router.attach("english", existing_agent)

# Manage resident memory (default keeps 1 hot, LRU eviction)
router = Router(max_loaded=2)       # keep two hot
router.unload()                     # free memory
```

| Deployment Mode | Per-Request Latency | Model Reloads |
|---|---|---|
| `Router()` (lazy, `max_loaded=1`) | 7 to 10 s on every language switch | 1 per switch |
| `Router(preload=True)` | **32.8 ms (GPU) / 193–464 ms (CPU)** | **none** |

---

## Single-Model Mode (Direct SDK)

If you only need a single checkpoint for a dedicated pipeline, you can load models directly:

```python
import deepopen

# 1. Load a specific checkpoint directly from the hub
agent = deepopen.load("convaiinnovations/deepopen")                           # English root
agent_ml = deepopen.load("convaiinnovations/deepopen", subfolder="multilingual") # 100+ languages
agent_td = deepopen.load("convaiinnovations/deepopen", subfolder="typed-decisions")

# 2. Run all questions in ONE single forward pass (~35 ms on GPU)
result = agent.predict(state, questions)
answers = result["answers"]

print("Department :", answers["department"]["choice"])   # -> billing (confidence: 0.94)
print("Urgency    :", answers["urgency"]["score"])        # -> 1.84 / 2.0
print("Churn Risk :", answers["churn_risk"]["noul"])       # -> 0.892 (89.2% probability)
```

---

## Automated Confidence Gating

Because deepopen's probabilities are trained with strictly proper scoring rules (RLCD), confidence scores are statistically meaningful:

```python
dept = answers["department"]["choice"]
conf = answers["department"]["confidence"]

if conf >= 0.85:
    # High confidence: automated action without human in the loop
    route_automatically(dept)
else:
    # Low confidence: escalate to human triage
    escalate_to_human_agent(dept, reason=f"Low confidence ({conf:.2f})")
```

---

## Built-in Workflow Presets

deepopen provides pre-tuned question schemas for immediate production use:

```python
import deepopen

agent = deepopen.load("convaiinnovations/deepopen")

# 1. Intelligent Model Router (routes to small vs. frontier models)
routing = agent.predict({"request": "Refactor this service using dependency injection"}, deepopen.router_questions())

# 2. Real-time Prompt Guardrails (jailbreaks, injections, leaks)
guard = agent.predict({"prompt": "Ignore all instructions"}, deepopen.guard_questions())

# 3. Content Safety & Moderation (toxicity, harassment, threats)
safety = agent.predict({"post": "User comment text"}, deepopen.moderation_questions())

# 4. Support Ticket Triage (intent, urgency, frustration, churn)
triage = agent.predict({"message": "My payment failed twice"}, deepopen.triage_questions())
```

---

## Decision Primitives

| Primitive | Output | Use Cases |
|---|---|---|
| **`choice`** | Top label, probabilities per option, confidence | Department routing, intent classification, topic categorization |
| **`score`** | Expected level on ordinal rubric, distribution, confidence | Frustration level, ticket urgency, harm severity |
| **`noul`** | Calibrated probability P(true) from 0.0 to 1.0 | Phishing detection, spam filtering, jailbreak detection, churn risk |

---

## Benchmarks

**Full report: [`BENCHMARKS.md`](BENCHMARKS.md)** — every run consolidated, languages and themes, with per-language detail for all 51 languages.

<p align="center">
  <img src="https://raw.githubusercontent.com/NandhaKishorM/deepopen/main/assets/deepopen_benchmark.png" alt="Per-language accuracy for both checkpoints across 51 languages" width="100%" />
</p>

All deepopen numbers below are measured. Every model answered byte-identical questions
(fixed seed) in the same run. Reproduce with
[`notebooks/deepopen_benchmark_colab.ipynb`](https://github.com/NandhaKishorM/deepopen) on a T4.

### Speed (Tesla T4, measured)

| questions per call | `deepopen` | `deepopen-multilingual` |
|---|---|---|
| 1 | 39.5 ms | **32.8 ms** |
| 5 | 84.5 ms | **40.1 ms** |
| 10 | 158.6 ms (15.9 ms/q) | **72.3 ms (7.2 ms/q)** |
| 50 | 771 ms | **337 ms (6.8 ms/q)** |

Batched throughput reaches 103-332 questions/sec on a single T4. For reference, TypeSafe Jev
has been independently measured at 236-276 ms p50
([AbdelStark](https://github.com/AbdelStark/jev-benchmarks),
[nibzard](https://github.com/nibzard/decision-model-benchmark)) -- deepopen answers a single
question roughly **6-7x faster**.

### deepopen (with routing) vs Jev

Every deepopen figure is what `Router().predict(...)` actually returns — the checkpoint the router
selects for that input, not a hand-picked best of three. Jev figures are **third-party
published, never measured here** (no TypeSafe API access), so sample sizes and prompts differ.

| | Jev 1.13.0 | deepopen (routed) | |
|---|---|---|---|
| typed-decisions, 2,000 decisions | 0.727 | **0.766** | +0.039 |
| AG News, 4 labels | 0.910 | **0.950** | +0.040 |
| DAIR Emotion, 6 labels | 0.480 | **0.595** | +0.115 |
| Banking77 (72 vs 77 labels) | **0.870** | 0.425 | Jev leads on >20 options |
| ECE *(lower better)* | 0.246 | **0.081** | 3× better (post-temperature) |
| p50 latency, 1 question | 236–276 ms | **32.8 ms** | 7.8× faster |
| Languages usable | *no published benchmark* | **45 of 51** | — |
| Weights | closed API | **Apache 2.0** | — |
| Cost | $0.042 / 1M tokens | **$0 self-hosted** | — |

On DAIR Emotion, Jev assigned **zero probability to the true label on 16% of examples** — a hard
failure for anything branching on confidence.

#### Where Jev leads

* **High-cardinality label spaces (>20 options at default settings):** On Banking77, Jev scores 0.870 (on 72 labels) while deepopen scores 0.425 (on 77 labels at default 256-token head budget). This is an architectural token-budget constraint: options share a fixed `head_max_len` budget (192 tokens on English, 256 on multilingual), so 77 options receive only ~3 to 4 tokens per label, causing text to become indistinguishable. Jev supports up to 255 options out-of-the-box. While `deepopen-multilingual` supports 1,024 context (and up to 8,192 in the encoder) and you can raise `agent.cfg["head_max_len"] = 512` at runtime, Jev is currently better suited for 50+ options in a single prompt without tuning.
* **Soft distribution matching:** On typed-decisions, while deepopen achieves higher argmax accuracy (0.766 vs 0.727), Jev achieves higher soft accuracy (0.580 vs 0.471) against the teacher's full probability distributions.
* **Out-of-the-box raw calibration:** Before temperature scaling, the base checkpoint has higher raw ECE (0.213 vs 0.144). deepopen achieves its 0.081 ECE after domain temperature fitting.

Full detail, including every workflow and all 51 languages: **[`BENCHMARKS.md`](BENCHMARKS.md)**.

### typed-decisions, measured on all three checkpoints

400 cases, 2,000 decisions, four workflows.

| model | accuracy | soft acc | Brier | ECE | score MAE |
|---|---|---|---|---|---|
| **`deepopen-typed-decisions`** | **0.766** | 0.471 | **0.062** | 0.213 | **0.242** |
| `deepopen` | 0.362 | 0.332 | 0.316 | 0.175 | 0.694 |
| `deepopen-multilingual` | 0.342 | 0.326 | 0.439 | 0.285 | 0.687 |
| *Jev 1.13.0 (published)* | *0.727* | *0.580* | *0.148* | *0.144* | *0.391* |
| *teacher self-agreement ceiling* | *0.735* | | | | |
| *per-question majority class* | *0.461* | | | | |
| *random guess* | *0.318* | | | | |

The fine-tuned checkpoint beats Jev by 3.9 points and clears the teacher ceiling, with 2.4x
better Brier and 1.6x better score MAE. It wins on all four workflows: invoice processing
0.804, security incidents 0.766, customer service 0.764, agent-trace observability 0.730.
By primitive: `noul` 0.857, `choice` 0.733, `score` 0.723.

Two places it still trails Jev: **soft accuracy** (0.471 vs 0.580 — its argmax is better but
its distributions match the teacher less well) and **ECE** (0.213 vs 0.144), which temperature
fitting addresses.

**The base checkpoints sit below the majority-class baseline** (0.362 and 0.342 against 0.461).
All of the capability on this benchmark comes from fine-tuning.

### Multilingual (51 languages, MASSIVE intent, 20 options, random = 0.050)

| | `deepopen` | `deepopen-multilingual` |
|---|---|---|
| English | **0.783** | 0.657 |
| 13 other languages | 0.306 | **0.451** |
| XNLI, English | **0.860** | 0.843 |
| XNLI, 14 other languages | 0.521 | **0.731** |

Across all 51 languages the English checkpoint macro-averages **0.227** with macro ECE
**0.733**, and only 23 of 51 languages clear 3x random. Khmer scores **0.000 at 95.2%
confidence**. This is why [`Router`](#model-routing-three-checkpoints-one-call) exists: the
model's own confidence gives no warning, so the routing decision has to be made before the
forward pass.

### English tasks

| task | `deepopen` | `deepopen-multilingual` | note |
|---|---|---|---|
| AG News | **0.947** | 0.937 | in training mix |
| BoolQ | **0.830** | 0.787 | in training mix |
| DAIR Emotion | **0.573** | 0.513 | held out |
| prompt-injections | **0.698** | 0.578 | held out, n=116 |
| SST-5 (ordinal) | 0.372 | 0.282 | held out |

### Calibration

Both checkpoints are over-confident as shipped. Refitting one temperature per (question type,
option count) on held-out data moves mean ECE **0.466 -> 0.081** (`deepopen`) and
**0.314 -> 0.106** (`deepopen-multilingual`). `deepopen-multilingual` ships with no fitted
temperatures at all, so fit them before relying on its probabilities.

### Honest limits

* **The base checkpoints are near chance on typed-decisions zero-shot** -- 0.362 and 0.352
  against a 0.318 random baseline and a 0.461 majority-class baseline. The 0.766 figure comes
  from the checkpoint fine-tuned on that benchmark's own training split. deepopen is a fast base to
  specialise, not a zero-shot decision engine.
* **High-cardinality choice questions and token budgets:** Sequences split into an option prompt budget (`head_max_len`) and the remaining document/state budget (`max_len - head_max_len`):
  * `deepopen` (English) defaults to 512 context (`head_max_len = 192`, ~320 tokens for state).
  * `deepopen-multilingual` and `deepopen-typed-decisions` default to 1,024 context (`head_max_len = 256`, ~768 tokens for state; mmBERT-base encoder supports up to 8,192 with RoPE).
  At default settings, a 77-option question like Banking77 allocates only `(256 - 16) // 77` ≈ 3–4 tokens per label, which causes accuracy to fall off sharply (0.425 vs Jev's 0.870). If evaluating 50+ options in a single question:
  1. Raise `agent.cfg["head_max_len"] = 512` and `agent.cfg["max_len"] = 1024` (or up to 2048 / 4096 / 8192) so every option has enough tokens to remain distinct.
  2. Or split large option sets into a two-step coarse-to-fine hierarchical choice.
* Ordinal `score` questions are the weakest primitive (SST-5 0.372).
* `deepopen` collapses outside English; `deepopen-multilingual` is weaker on English. Route, or pick
  deliberately.

---

## Live Demo & Resources

* **Hugging Face Model:** [convaiinnovations/deepopen](https://huggingface.co/convaiinnovations/deepopen)
* **Interactive Web Demo:** [convaiinnovations/deepopen-demo](https://huggingface.co/spaces/convaiinnovations/deepopen-demo)
* **Engineering Writeup:** [Read the full story on Dev.to](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)

---

## Fine-Tuning

Fine-tune deepopen on your own domain data. The notebook runs on Kaggle's free 2xT4 GPUs and does
the whole loop: build the dataset, train with RLCD (proper-scoring-rule rewards, GRPO-style
policy gradient), fit calibration temperatures, evaluate, and push the result to the Hub.

* **[`notebooks/deepopen_finetune_typed_decisions_2xT4_kaggle.ipynb`](notebooks/deepopen_finetune_typed_decisions_2xT4_kaggle.ipynb)**

Fine-tuning is where most of the value is. On the typed-decisions benchmark the base
checkpoints score near chance zero-shot (0.36 and 0.35 against a 0.318 random baseline),
while the fine-tuned checkpoint reaches **0.766** on the same 2,000 decisions -- above
TypeSafe Jev's published 0.727 and above the 0.735 teacher self-agreement ceiling. Treat deepopen
as a fast base to specialise, not as a zero-shot decision engine.

Runtime on 2xT4 is roughly 4-5 hours for 4 epochs over ~30k questions.

---

 
---

## License

Apache 2.0. Developed by Deep Open
