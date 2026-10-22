# DeepOpen：开源多语言 System 1 决策引擎

[English](README.md) | **简体中文**

DeepOpen 是基于laya的一款完全开源的非自回归System 1决策引擎，专为结构化类型决策场景设计。

## 复现 打榜  banking77

https://github.com/deepopen-com/deepopen/tree/main/banking77


## 复现 打榜  clinc150

https://github.com/deepopen-com/deepopen/tree/main/clinc150


它摒弃了传统大模型逐Token生成文本的模式，在单次前向传递中即可完成100+种语言的多维度类型判断，单请求延迟低至33毫秒、批量处理仅7.2毫秒（T4显卡实测），依托严格正确评分规则RLCD完成强化学习训练，通过内置智能路由器自动为每个请求匹配最优检查点，彻底解决了传统大模型在分类、路由、打分场景下速度慢、成本高、易产生幻觉的痛点。


## 打榜表现

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

# 将所有检查点预加载到内存，实现35毫秒以内的极速推理
router = Router(preload=True, device="cuda")

# 定义任意语言的输入状态
state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice 4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
}

# 定义自定义类型决策规则
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

# 自动路由到最优模型完成推理
res_en = router.predict(state, questions)
print("Department :", res_en["answers"]["department"]["choice"])  # 输出: billing (confidence: 0.94)
print("Routing    :", res_en["routing"]["model"])                 # 输出: english
```

3. 生产环境部署优化
根据业务场景灵活调整内存策略，避免不必要的显存占用：
```python
# 仅预加载业务所需的检查点，节省显存
router.preload(["english", "multilingual"])

# 配置LRU缓存策略，最多同时保留2个热模型
router = Router(max_loaded=2)

# 手动卸载模型释放内存
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

## 参考

<br>参考资料<br>[1] [DeepL + OpenAI integrations - connect and automate | Bardeen.ai - www.bardeen.ai](https://www.bardeen.ai/integrations/openai/deepl)<br>[2] [Minutes.ai - Smart Government Innovation LAB - www.smartlab.gov.hk](https://www.smartlab.gov.hk/en/ai_solutions/a-0063)<br>[3] [benchmark.ng — Nigeria's AI Decision Engine - benchmark.ng](https://benchmark.ng/)<br>[4] [Phrase: AI-Powered Localization & Translation Platform - Phrase官网](https://phrase.com/?utm_medium=yelp_blog&utm_source=logiciels.pro&utm_campaign=5-ways-yelp-can-help&utm_content=blog_text_link&utm_term=meet-yelp-host)<br>[5] [DeepSeek Open Web UI 安装部署全攻略：从环境配置到可视化交互 - 百度智能云](https://cloud.baidu.com/article/3564171)<br>[6] [Jev 是什么：前 OpenAI 研究员做的“不说话“模型，只输出带概率的结构化决策-CSDN博客 - CSDN博客](https://blog.csdn.net/aidoudoulong/article/details/166136816)<br>[7] [前OpenAI研究员推Jev模型，绕开文本生成直出决策 - m.counselleap.cn](http://m.counselleap.cn/qqznews/202609/article_2962603.shtml)<br>[8] [MCP服务器 - 开放式深度研究模型协议-MCP服务 - www.mcpworld.com](https://www.mcpworld.com/zh/detail/8ed885da2b0dac2c97d11afb4a219269)<br>[9] [SGLang开源引擎：开源创新与推理革命的融合之路 - 百度智能云](https://cloud.baidu.com/article/4935192)<br>[10] [GitHub - weibaohui/openDeepWiki: 完全AI驱动的 DeepWiki ,使用 Go + Eino 技术栈开发 · GitHub - GitHub](https://github.com/weibaohui/openDeepWiki)<br>[11] [正面硬刚 OpenAI o1！DeepSeek-R1：开启 AI 自主推理新时代，现已开源！ - 知乎 - 知乎](https://zhuanlan.zhihu.com/p/1960455097218761196)<br>[12] [详解多智能体架构：以 Open Deep Research 项目为例 - 知乎 - 知乎](https://zhuanlan.zhihu.com/p/1944449334209942670)<br>[13] [集成DeepSeek的开源利器：3款高效应用深度解析 - 百度智能云](https://cloud.baidu.com/article/5076327)<br>[14] [当开源创新遇上推理革命：SGLang如何炼就DeepSeek最强开源推理引擎？_腾讯新闻 - 腾讯网](https://new.qq.com/rain/a/20250306A09PRF00)<br>[15] [3款集成DeepSeek的开源应用推荐：开发者高效工具指南 - 百度智能云](https://cloud.baidu.com/article/4511140)<br>[16] [当开源创新遇上推理革命：SGLang如何炼就DeepSeek最强开源推理引擎？-腾讯云开发者社区-腾讯云 - 腾讯云](https://cloud.tencent.com/developer/article/2502983)<br>[17] [DeepSeek-V3.2：开源大语言模型的新里程碑，在推理与智能体任务中突破性能边界 - 知乎 - 知乎](https://zhuanlan.zhihu.com/p/1978976195946239755)<br>[18] [Open Deep Research V2：新架构！上下文工程最新实践 - 知乎 - 知乎](https://zhuanlan.zhihu.com/p/1933207553409525463)<br>[19] [深度探索DeepSeek：解锁高效开发与智能决策的新范式 - 百度智能云](https://cloud.baidu.com/article/3732497)<br>[20] [【AI大模型】 Langchain+DeepSeek R1从入门到精通，收藏这一篇就够了！！_langchain从入门到实战-CSDN博客 - CSDN博客](https://blog.csdn.net/2401_85327249/article/details/147629035)<br>[21] [开源×推理”双引擎驱动：SGLang打造DeepSeek开源推理新标杆 - 百度智能云](https://cloud.baidu.com/article/3728141)<br><br>

## 许可证

本项目采用 [Apache 2.0](LICENSE) 许可证。
