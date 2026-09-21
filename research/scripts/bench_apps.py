"""Application-workflow benchmark for the Laya checkpoints, plus the tasks where public
Jev numbers exist so a like-for-like comparison is possible.

Workflows (the demo Space's tabs), each on real labelled data:
  1 support triage      banking77 (77-way intent) + customer-support-tickets queue routing
  2 email + phishing    enron spam + phishing emails
  3 LLM guardrails      lmsys/toxic-chat jailbreaking flag  (held out of Laya training)
  4 RAG passage filter  MS MARCO passage relevance
  5 moderation          lmsys/toxic-chat toxicity flag      (held out of Laya training)
  6 model routing       domain classification over gsm8k / mbpp / writing / factual

Jev-comparable tasks (AbdelStark/jev-benchmarks published Jev accuracy on these):
  ag_news 0.910 | banking77 0.870 | dair emotion 0.480 (Brier 0.846, NLL 5.588)

  USE_TF=0 python3 notebooks/bench_apps.py
"""
import gc
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "laya"))
sys.path.insert(0, os.path.join(REPO, "notebooks"))

import laya  # noqa: E402
from bench_local import load, metrics, score_cases, softmax_t, temp_for  # noqa: E402

OUT = os.path.join(REPO, "app_benchmark_results.json")
SEED = 13
N = int(os.environ.get("BENCH_N", "400"))

# Published Jev numbers on identical public datasets (NOT measured here - no TypeSafe API access).
JEV_PUBLISHED = {
    "ag_news": {"accuracy": 0.910, "coverage_at_5pct_error": 0.830, "latency_p50_ms": 236,
                "n": 100, "source": "AbdelStark/jev-benchmarks v0.1.0"},
    "banking77": {"accuracy": 0.870, "coverage_at_5pct_error": 0.860, "latency_p50_ms": 246,
                  "n": 100, "labels": 72, "source": "AbdelStark/jev-benchmarks v0.1.0"},
    "emotion": {"accuracy": 0.480, "brier": 0.846, "nll": 5.588, "coverage_at_5pct_error": 0.0,
                "zero_prob_failures": 0.16, "n": 100, "source": "AbdelStark/jev-benchmarks v0.1.0"},
    "typed_decisions": {"accuracy": 0.727, "soft_accuracy": 0.580, "brier": 0.148, "ece": 0.144,
                        "score_mae": 0.391, "ms_per_case": 710, "source": "laya repo comparison table"},
    "_independent_": {"banking77_accuracy": 0.763, "sms_spam_accuracy": 0.930,
                      "permuted_accuracy": 0.767, "ece": 0.246, "latency_p50_ms": "264-276",
                      "option_order_flip_rate": 0.13,
                      "source": "nibzard/decision-model-benchmark"},
}

SUITES = {}


def register(name, cases, gold, **meta):
    SUITES[name] = {"cases": cases, "gold": gold, "meta": meta}
    print("   %-26s %4d cases  %s" % (name, len(cases), meta.get("note", "")), flush=True)


def choice_q(qid, instructions, keys, gold_key):
    crit = {k: None for k in keys}
    return {qid: {"type": "choice", "instructions": instructions, "criteria": crit}}, keys.index(gold_key)


def build():
    from datasets import load_dataset
    rng = random.Random(SEED)

    # ---------------------------------------------------------- Jev-comparable: ag_news
    try:
        d = load_dataset("fancyzhx/ag_news", split="test")
        crit = {"world": "world news and international politics", "sports": "sports",
                "business": "business and economy", "sci_tech": "science and technology"}
        keys = list(crit)
        cases, gold = [], []
        for r in list(d)[:N]:
            cases.append(({"article": r["text"]},
                          {"topic": {"type": "choice", "instructions": "What is the topic of `article`?",
                                     "criteria": dict(crit)}}))
            gold.append(keys.index(keys[int(r["label"])]))
        register("jev.ag_news", cases, gold, note="Jev 0.910", jev="ag_news", in_training=True)
    except Exception as e:
        print("   FAIL ag_news", str(e)[:80])

    # ---------------------------------------------------------- Jev-comparable: emotion
    try:
        d = load_dataset("dair-ai/emotion", "split", split="test")
        names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        cases, gold = [], []
        for r in list(d)[:N]:
            cases.append(({"text": r["text"]},
                          {"emotion": {"type": "choice",
                                       "instructions": "Which emotion is most strongly expressed in `text`?",
                                       "criteria": {n: None for n in names}}}))
            gold.append(int(r["label"]))
        register("jev.emotion", cases, gold, note="Jev 0.480 (Brier 0.846)", jev="emotion",
                 in_training=False)
    except Exception as e:
        print("   FAIL emotion", str(e)[:80])

    # ------------------------------------------- Jev-comparable + triage: banking77 (77 labels)
    try:
        d = load_dataset("mteb/banking77", split="test")
        labels = sorted(set(d["label_text"]))
        cases, gold = [], []
        for r in list(d)[:N]:
            qs, gi = choice_q("intent", "Which banking intent does `message` express?",
                              [x.replace("_", " ") for x in labels],
                              r["label_text"].replace("_", " "))
            cases.append(({"message": r["text"]}, qs)); gold.append(gi)
        register("jev.banking77_full", cases, gold,
                 note="%d labels at once | Jev 0.870 (72 labels)" % len(labels),
                 jev="banking77", in_training=False, n_labels=len(labels))
    except Exception as e:
        print("   FAIL banking77", str(e)[:80])

    # ---------------------------------------------------------- 1. support triage (queues)
    try:
        d = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
        QUEUES = {"Technical Support": "technical problems, bugs, outages, integrations",
                  "Product Support": "help using a product or feature",
                  "Customer Service": "general account or service questions",
                  "IT Support": "internal IT, devices, access, networks",
                  "Billing and Payments": "invoices, charges, refunds, payment methods",
                  "Returns and Exchanges": "returning or exchanging an item",
                  "Service Outages and Maintenance": "downtime, outages, scheduled maintenance",
                  "Sales and Pre-Sales": "pricing, quotes, buying",
                  "Human Resources": "employment, payroll, leave, hiring",
                  "General Inquiry": "anything else"}
        keys = list(QUEUES)
        cases, gold = [], []
        for r in d:
            if r.get("language") != "en" or r.get("queue") not in QUEUES or not r.get("body"):
                continue
            cases.append(({"subject": r["subject"] or "", "body": r["body"].replace("\\n", "\n")[:3000]},
                          {"queue": {"type": "choice",
                                     "instructions": "Which support queue should handle this ticket?",
                                     "criteria": dict(QUEUES)}}))
            gold.append(keys.index(r["queue"]))
            if len(cases) >= N:
                break
        register("app.support_triage", cases, gold, note="10-way queue routing", in_training=True)
    except Exception as e:
        print("   FAIL support_triage", str(e)[:80])

    # ---------------------------------------------------------- 2. email + phishing
    try:
        d = load_dataset("SetFit/enron_spam", split="test")
        cases, gold = [], []
        for r in list(d)[:N]:
            st = laya.email_state(r.get("subject") or "", (r.get("message") or "")[:3000])
            cases.append((st, {"is_spam": {"type": "noul",
                                           "instructions": "Is this email unsolicited spam or bulk marketing?"}}))
            gold.append(int(r["label"]))
        register("app.email_spam", cases, gold, note="enron spam", in_training=True)
    except Exception as e:
        print("   FAIL email_spam", str(e)[:80])

    try:
        d = load_dataset("zefang-liu/phishing-email-dataset", split="train")
        rows = [r for r in list(d)[:6000]
                if (r.get("Email Text") or "").strip() and r.get("Email Type") in ("Safe Email", "Phishing Email")]
        rng.shuffle(rows)
        cases, gold = [], []
        for r in rows[:N]:
            cases.append(({"email": r["Email Text"][:3000]},
                          {"is_phishing": {"type": "noul",
                                           "instructions": "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
                                           "criteria": {"true": "phishing, scam, or fraud",
                                                        "false": "a legitimate email (even if promotional)"}}}))
            gold.append(int(r["Email Type"] == "Phishing Email"))
        register("app.phishing", cases, gold, note="phishing emails", in_training=True)
    except Exception as e:
        print("   FAIL phishing", str(e)[:80])

    # ------------------------------------- 3 & 5. guardrails + moderation (toxic-chat, HELD OUT)
    try:
        d = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
        rows = [r for r in d if (r.get("user_input") or "").strip()]
        jb = [r for r in rows if int(r.get("jailbreaking", 0)) == 1][:N // 2]
        nj = [r for r in rows if int(r.get("jailbreaking", 0)) == 0][:N - len(jb)]
        mix = jb + nj
        rng.shuffle(mix)
        cases, gold = [], []
        for r in mix:
            cases.append(({"prompt": r["user_input"][:3000]},
                          {"jailbreak": {"type": "noul",
                                         "instructions": "Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?"}}))
            gold.append(int(r["jailbreaking"]))
        register("app.guardrails_jailbreak", cases, gold,
                 note="toxic-chat jailbreaking (HELD OUT)", in_training=False)

        tox = [r for r in rows if int(r.get("toxicity", 0)) == 1][:N // 2]
        ntox = [r for r in rows if int(r.get("toxicity", 0)) == 0][:N - len(tox)]
        mix2 = tox + ntox
        rng.shuffle(mix2)
        cases, gold = [], []
        for r in mix2:
            cases.append(({"post": r["user_input"][:3000]},
                          {"toxic": {"type": "noul",
                                     "instructions": "Is `post` toxic: rude, disrespectful or likely to make someone leave the discussion?"}}))
            gold.append(int(r["toxicity"]))
        register("app.moderation_toxicity", cases, gold,
                 note="toxic-chat toxicity (HELD OUT)", in_training=False)
    except Exception as e:
        print("   FAIL toxic-chat", str(e)[:80])

    # ---------------------------------------------------------- 4. RAG passage filter
    try:
        d = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
        cases, gold = [], []
        for r in d:
            texts, sel = r["passages"]["passage_text"], r["passages"]["is_selected"]
            pos = [t for t, s in zip(texts, sel) if s == 1]
            neg = [t for t, s in zip(texts, sel) if s == 0]
            if not pos or not neg:
                continue
            take_pos = len(cases) % 2 == 0
            p = rng.choice(pos if take_pos else neg)
            cases.append(({"query": r["query"], "passage": p},
                          {"relevant": {"type": "noul",
                                        "instructions": "Does `passage` help answer `query`?"}}))
            gold.append(1 if take_pos else 0)
            if len(cases) >= N:
                break
        register("app.rag_relevance", cases, gold, note="MS MARCO relevance", in_training=True)
    except Exception as e:
        print("   FAIL rag", str(e)[:80])

    # ---------------------------------------------------------- 6. model routing (domain)
    try:
        DOM = {"code": "software engineering, programming, refactoring, architecture, debugging",
               "math_or_logic": "mathematics, logic puzzles, proofs, complex calculation",
               "writing": "creative writing, essays, emails, blog posts, copywriting",
               "factual_lookup": "facts, definitions, trivia, history",
               "data_analysis": "statistics, SQL, data manipulation, metrics",
               "chitchat": "casual conversation, greetings, small talk"}
        keys = list(DOM)
        pool = []
        g = load_dataset("openai/gsm8k", "main", split="test")
        pool += [(r["question"], "math_or_logic") for r in list(g)[:N // 3]]
        m = load_dataset("google-research-datasets/mbpp", "full", split="test")
        pool += [(r["text"], "code") for r in list(m)[:N // 3]]
        t = load_dataset("fancyzhx/ag_news", split="test")
        pool += [(r["text"][:400], "factual_lookup") for r in list(t)[:N // 3]]
        rng.shuffle(pool)
        cases, gold = [], []
        for text, dom in pool[:N]:
            cases.append(({"request": text},
                          {"domain": {"type": "choice",
                                      "instructions": "What domain does `request` belong to?",
                                      "criteria": dict(DOM)}}))
            gold.append(keys.index(dom))
        register("app.model_routing_domain", cases, gold,
                 note="gsm8k/mbpp/ag_news -> domain", in_training=False)
    except Exception as e:
        print("   FAIL routing", str(e)[:80])


def main():
    print("=== building suites (N=%d per task) ===\n" % N, flush=True)
    build()
    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "device": "cpu",
                        "n_per_task": N, "seed": SEED, "laya": laya.__version__},
               "jev_published": JEV_PUBLISHED, "suites": {}}
    for mname in ("english", "multilingual", "typed-decisions"):
        print("\n=== %s ===" % mname, flush=True)
        ag = load(mname)
        for sname, S in SUITES.items():
            try:
                lgs, idx, secs, dropped = score_cases(ag, S["cases"], tag="%s/%s" % (mname, sname))
                rows = [(S["gold"][ci], softmax_t(z, temp_for(ag, qt, k)) if z is not None else None)
                        for (ci, _, qt, k), z in zip(idx, lgs)]
                m = metrics(rows)
                m["seconds"] = round(secs, 1)
                m["ms_per_case"] = round(1000 * secs / max(1, len(S["cases"])), 1)
                m["dropped"] = dropped
                m.update({k: v for k, v in S["meta"].items() if k != "note"})
                results["suites"].setdefault(sname, {})[mname] = m
                jev = JEV_PUBLISHED.get(S["meta"].get("jev") or "", {}).get("accuracy")
                delta = ("  vs Jev %.3f -> %+.3f" % (jev, m["accuracy"] - jev)) if jev else ""
                print("   %-26s acc %.3f  f1 %.3f  ECE %.3f  %5.1f ms/case%s"
                      % (sname, m["accuracy"], m["macro_f1"], m["ece"], m["ms_per_case"], delta),
                      flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                results["suites"].setdefault(sname, {})[mname] = {"error": str(e)[:200]}
        del ag; gc.collect()
        json.dump(results, open(OUT, "w"), indent=2)
    json.dump(results, open(OUT, "w"), indent=2)
    print("\nwrote %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
