"""Emit laya_benchmark_colab.ipynb: an extensive head-to-head benchmark of
convaiinnovations/laya (ModernBERT-large, English) vs convaiinnovations/laya-multilingual
(mmBERT-base, 100+ languages), on a Colab T4.

Writes one laya_benchmark_results.json that can be handed back for analysis.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# Laya vs Laya-Multilingual — extensive head-to-head benchmark

Runs both open checkpoints over the same questions and writes a single
`laya_benchmark_results.json`.

| | `convaiinnovations/laya` | `convaiinnovations/laya-multilingual` |
|---|---|---|
| encoder | ModernBERT-large (English) | mmBERT-base (100+ languages) |
| params | 421.3M (394.8M enc + 26.5M head) | 321.9M (306.9M enc + 15.0M head) |
| context | 512 tokens | 1024 tokens |
| shipped temperatures | fitted per type & option-count | **none (all 1.0)** |

**What is measured**

1. **typed-decisions** — 400 cases / 2,000 decisions, 4 workflows (the benchmark behind the published 0.766 claim). Zero-shot for both of these checkpoints.
2. **Multilingual** — MASSIVE intent + scenario across 14 languages, XNLI across 15 languages.
3. **English zero-shot** — SST-5, emotion, prompt-injections, plus banking77 as a 77-option stress test.
4. **Latency** — p50/p95 at 1 / 5 / 10 / 50 questions per call.
5. **Option-order robustness** — how often the answer flips when the options are permuted (the independent Jev benchmark measured 13% for Jev; worst LLM 37%).
6. **Calibration repair** — ECE before and after refitting temperature, which matters because `laya-multilingual` shipped uncalibrated.

**Setup:** Runtime → Change runtime type → **T4 GPU**. Then Run All (~25–40 min).
Both models see byte-identical questions (fixed seed), so every difference is the model.
"""),

    md("## 1. GPU and environment"),
    code("""
import subprocess, sys, os, json, time
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,compute_cap","--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip() or "NO GPU - set Runtime > Change runtime type > T4 GPU")
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available(), "Enable the T4 GPU runtime before running."
"""),

    md("## 2. Dependencies"),
    code("""
!pip -q install -U "laya>=0.1.6" "transformers>=4.45" "datasets>=3.0" safetensors huggingface_hub 2>&1 | tail -3
import laya, transformers, datasets
print("laya", getattr(laya, "__version__", "?"), "| transformers", transformers.__version__, "| datasets", datasets.__version__)
"""),

    md("""## 3. Download both checkpoints

`reference_compile` is forced off after loading: the `laya` package does not disable it (only
`rl_agent_api.py` does), and torch.compile is a loss on small batches / few SMs like a T4."""),
    code("""
import torch, json, os, time
from huggingface_hub import snapshot_download
from safetensors import safe_open

REPOS = {"laya": "convaiinnovations/laya", "laya-multilingual": "convaiinnovations/laya-multilingual"}
PATHS, MODEL_META = {}, {}

def patch_tokenizer_config(model_dir):
    '''laya-multilingual ships extra_special_tokens as a LIST (inherited from the mmBERT/Gemma
    tokenizer); transformers expects a dict and raises
    AttributeError: list object has no attribute keys. Without this, AutoTokenizer -- and so
    laya.load() -- fails outright on that repo. Also normalises tokenizer_class for 4.x/5.x.'''
    p = os.path.join(model_dir, "tokenizer", "tokenizer_config.json")
    if not os.path.exists(p):
        return []
    c = json.load(open(p))
    changed = []
    est = c.get("extra_special_tokens")
    if isinstance(est, list):
        c["extra_special_tokens"] = {("extra_%d" % i): t for i, t in enumerate(est)}
        changed.append("extra_special_tokens list->dict")
    if c.get("tokenizer_class") in (None, "TokenizersBackend"):
        c["tokenizer_class"] = "PreTrainedTokenizerFast"
        c.pop("backend", None); c.pop("is_local", None)
        changed.append("tokenizer_class")
    if changed:
        json.dump(c, open(p, "w"), indent=2)
    return changed

for name, repo in REPOS.items():
    t = time.time()
    PATHS[name] = snapshot_download(repo)
    fixed = patch_tokenizer_config(PATHS[name])
    if fixed:
        print("   patched tokenizer config for %s: %s" % (name, ", ".join(fixed)))
    cfg = json.load(open(os.path.join(PATHS[name], "rl_agent_config.json")))
    ecfg = json.load(open(os.path.join(PATHS[name], "encoder", "config.json")))
    tot = enc = 0
    with safe_open(os.path.join(PATHS[name], "model.safetensors"), "pt") as f:
        for k in f.keys():
            n = 1
            for d in f.get_slice(k).get_shape():
                n *= d
            tot += n
            enc += n if k.startswith("encoder.") else 0
    MODEL_META[name] = {
        "repo": repo, "params_total_m": round(tot/1e6, 2), "params_encoder_m": round(enc/1e6, 2),
        "params_head_m": round((tot-enc)/1e6, 2), "hidden_size": ecfg.get("hidden_size"),
        "num_layers": ecfg.get("num_hidden_layers"), "vocab_size": ecfg.get("vocab_size"),
        "max_len": cfg.get("max_len"), "head_max_len": cfg.get("head_max_len"),
        "temperature": cfg.get("temperature"), "temperature_by_options": cfg.get("temperature_by_options", {}),
        "training": cfg.get("training", {}), "download_s": round(time.time()-t, 1),
        "tokenizer_config_patched": fixed,
    }
    print(name, "->", PATHS[name])
    print("   ", json.dumps({k: v for k, v in MODEL_META[name].items() if k != "temperature_by_options"}))

def load_agent(name):
    import laya
    ag = laya.load(PATHS[name], device="cuda")
    try:
        ag.model.encoder.config.reference_compile = False   # torch.compile hurts on T4-class GPUs
    except Exception as e:
        print("could not disable reference_compile:", e)
    ag.model.eval()
    return ag
"""),

    md("## 4. Benchmark engine — batched scoring and metrics"),
    code(r"""
%%writefile bench_engine.py
import json, math, sys, time
import numpy as np
import torch
from laya.common import QTYPES, build_sequence, collate_items, render_options, temp_bucket

def to_internal(qdef):
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}

@torch.no_grad()
def score_cases(agent, cases, max_tokens=16384, max_seqs=128, progress=""):
    '''cases: [(state, {qid: qdef})] -> (raw_logits per question, index, seconds, n_dropped).

    Every question of every case is one sequence. Sequences are length-sorted and packed into
    token-budgeted batches so padding stays small. Returns UNCALIBRATED logits; temperature is
    applied later so raw/calibrated variants come from one forward pass.
    '''
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    items, index, dropped = [], [], 0
    for ci, (state, questions) in enumerate(cases):
        for qid, qdef in questions.items():
            q = to_internal(qdef)
            try:
                ids, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
            except Exception:
                index.append((ci, qid, QTYPES[q["t"]], 0)); items.append(None); dropped += 1
                continue
            if len(markers) != len(render_options(q)):
                index.append((ci, qid, QTYPES[q["t"]], 0)); items.append(None); dropped += 1
                continue
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            index.append((ci, qid, QTYPES[q["t"]], len(markers)))

    valid = [i for i, it in enumerate(items) if it is not None]
    order = sorted(valid, key=lambda i: len(items[i]["ids"]))
    out = [None] * len(items)
    t0, done, i = time.time(), 0, 0
    while i < len(order):
        j, L = i, 0
        while j < len(order) and j - i < max_seqs and max(L, len(items[order[j]]["ids"])) * (j - i + 1) <= max_tokens:
            L = max(L, len(items[order[j]]["ids"])); j += 1
        j = max(j, i + 1)
        sel = [items[order[t]] for t in range(i, j)]
        b = collate_items([sel], agent.tok.pad_token_id)
        with torch.autocast("cuda", dtype=agent.dtype, enabled=True):
            logits, _ = agent.model(b["input_ids"].cuda(), b["attention_mask"].cuda(),
                                    b["marker_pos"].cuda(), b["marker_mask"].cuda(), b["qtype"].cuda())
        logits = logits.float().cpu().numpy()
        for r in range(j - i):
            out[order[i + r]] = logits[r, :len(sel[r]["markers"])]
        done += j - i
        if progress:
            el = time.time() - t0
            msg = "\r  [%s] %d/%d seq | %.0f seq/s | ETA %ds    " % (
                progress, done, len(order), done/max(el, 1e-9),
                int(el*(len(order)-done)/max(1, done)))
            sys.stdout.write(msg)
            sys.stdout.flush()
        i = j
    if progress:
        print("\r  [%s] %d sequences in %.1fs (%d dropped: options did not fit)%s" % (
            progress, len(order), time.time()-t0, dropped, " "*20))
    return out, index, time.time() - t0, dropped

def softmax_t(z, t=1.0):
    z = np.asarray(z, dtype=np.float64) / max(1e-3, float(t))
    e = np.exp(z - z.max())
    return e / e.sum()

def temp_for(agent, qt, k, calibrated=True):
    if not calibrated:
        return 1.0
    return float(agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt]))

# ------------------------------------------------------------------ metrics
def ece_score(conf, correct, bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)

def macro_f1(gold, pred):
    gold, pred = np.asarray(gold), np.asarray(pred)
    f1 = []
    for c in sorted(set(gold.tolist()) | set(pred.tolist())):
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        f1.append(2*tp / max(1, 2*tp + fp + fn))
    return float(np.mean(f1))

def aurc(conf, correct):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    o = np.argsort(-conf)
    return float((np.cumsum(1 - correct[o]) / np.arange(1, len(o)+1)).mean())

def hard_metrics(rows):
    '''rows: [(gold_idx, probs)] -> standard single-label classification + calibration metrics.'''
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        return {"n": 0}
    gold = np.array([g for g, _ in rows])
    pred = np.array([int(np.argmax(p)) for _, p in rows])
    conf = np.array([float(np.max(p)) for _, p in rows])
    corr = (pred == gold).astype(float)
    m = {"n": len(rows),
         "accuracy": float(corr.mean()),
         "macro_f1": macro_f1(gold, pred),
         "ece": ece_score(conf, corr),
         "brier": float(np.mean([((np.asarray(p) - np.eye(len(p))[g])**2).sum() for g, p in rows])),
         "nll": float(np.mean([-math.log(max(float(p[g]), 1e-12)) for g, p in rows])),
         "aurc": aurc(conf, corr),
         "mean_confidence": float(conf.mean())}
    for cov in (0.5, 0.8):
        k = max(1, int(len(conf)*cov))
        m["acc_at_%d_coverage" % int(cov*100)] = float(corr[np.argsort(-conf)[:k]].mean())
    return m

def fit_temperature(pairs, lo=0.2, hi=10.0, steps=160):
    '''pairs: [(logits, gold_idx)] -> single temperature minimising NLL (grid search, no autograd).'''
    pairs = [(np.asarray(z, float), int(g)) for z, g in pairs if z is not None]
    if len(pairs) < 25:
        return 1.0
    best_t, best = 1.0, float("inf")
    for t in np.geomspace(lo, hi, steps):
        tot = 0.0
        for z, g in pairs:
            zz = z/t
            zz = zz - zz.max()
            tot += -(zz[g] - math.log(np.exp(zz).sum()))
        if tot < best:
            best, best_t = tot, float(t)
    return round(float(best_t), 4)
"""),

    md("""## 5. Task suites

Every suite is built **once**, with a fixed seed, and reused for both models — so the two
checkpoints answer byte-identical questions. Option sets for many-label tasks use
`gold + N-1 distractors`, matching how the models were trained (the training recipe samples
5–20 options per choice question rather than showing the full label space)."""),

    code("""
import random, json
import numpy as np
from datasets import load_dataset

SEED = 13
N_OPTS = 20          # options per many-label choice question (gold + 19 distractors)
PER_LANG = 300       # cases per language
SUITES = {}          # name -> {"cases": [(state, questions)], "gold": [{qid: gold_idx}], "meta": {...}}

def build_choice_questions(rng, gold, all_labels, n_opts, instructions, qid="label"):
    pool = [x for x in all_labels if x != gold]
    keys = [gold] + rng.sample(pool, min(n_opts-1, len(pool)))
    rng.shuffle(keys)
    crit = {k: k.replace("_", " ").replace(".", ": ") for k in keys}
    return {qid: {"type": "choice", "instructions": instructions, "criteria": crit}}, keys.index(gold)

def register(name, cases, golds, **meta):
    SUITES[name] = {"cases": cases, "gold": golds, "meta": meta}
    nq = sum(len(c[1]) for c in cases)
    print("  %-34s %4d cases / %5d questions   %s" % (name, len(cases), nq, meta.get("note", "")))
"""),

    md("### 5a. typed-decisions (English, 400 cases / 2,000 decisions)"),
    code("""
td_cases, td_golds, td_wf = [], [], []
try:
    td = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    for r in td:
        questions = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
        gold = json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]
        state = r["state"]
        try:
            state = json.loads(state)
        except Exception:
            pass
        gmap = {}
        for qid, qdef in questions.items():
            g = gold[qid]
            if qdef["type"] == "choice":
                keys = list(qdef["criteria"].keys())
                gmap[qid] = {"idx": keys.index(str(g["label"])), "keys": keys,
                             "soft": [float(g.get("probabilities", {}).get(k, 0.0)) for k in keys]}
            elif qdef["type"] == "noul":
                idx = 1 if str(g["label"]).lower() == "true" else 0
                pt = float(g.get("probabilities", {}).get("true", g.get("noul", 0.5)))
                gmap[qid] = {"idx": idx, "keys": ["false", "true"], "soft": [1-pt, pt]}
            else:  # score
                n = len(qdef["criteria"])
                gmap[qid] = {"idx": int(g["label"]), "keys": [str(i) for i in range(n)],
                             "soft": [float(g.get("probabilities", {}).get(str(i), 0.0)) for i in range(n)],
                             "gold_score": float(g.get("score", float(g["label"])))}
        td_cases.append((state, questions))
        td_golds.append(gmap)
        td_wf.append(r["workflow"])
    register("typed_decisions", td_cases, td_golds, note="4 workflows, soft teacher labels",
             workflows=td_wf, soft=True)
except Exception as e:
    print("typed-decisions FAILED:", type(e).__name__, e)
"""),

    md("### 5b. Multilingual — MASSIVE intent & scenario (14 languages), XNLI (15 languages)"),
    code("""
MASSIVE_LANGS = ["en","de","fr","es","pt","ru","tr","ar","hi","ta","zh-CN","ja","ko","sw"]
XNLI_LANGS    = ["en","de","fr","es","ru","tr","ar","hi","ur","vi","th","el","bg","zh","sw"]

for task, short, instr in [("mteb/amazon_massive_intent", "massive_intent",
                            "What is the user asking for in `utterance`?"),
                           ("mteb/amazon_massive_scenario", "massive_scenario",
                            "Which domain does `utterance` belong to?")]:
    for lg in MASSIVE_LANGS:
        try:
            d = load_dataset(task, lg, split="test")
            labels = sorted(set(d["label_text"]))
            rng = random.Random(SEED)
            cases, golds = [], []
            for r in list(d)[:PER_LANG]:
                qs, gi = build_choice_questions(rng, r["label_text"], labels, N_OPTS, instr)
                cases.append(({"utterance": r["text"]}, qs))
                golds.append({"label": {"idx": gi}})
            register("%s.%s" % (short, lg), cases, golds, lang=lg, n_options=N_OPTS,
                     label_space=len(labels), family=short)
        except Exception as e:
            print("  FAIL %s %s: %s" % (short, lg, str(e)[:90]))

NLI_CRIT = {"entailment": "the premise implies the hypothesis is true",
            "neutral": "the premise neither implies nor contradicts the hypothesis",
            "contradiction": "the premise implies the hypothesis is false"}
for lg in XNLI_LANGS:
    try:
        d = load_dataset("facebook/xnli", lg, split="test")
        cases, golds = [], []
        for r in list(d)[:PER_LANG]:
            qs = {"relation": {"type": "choice",
                               "instructions": "What is the relationship between `premise` and `hypothesis`?",
                               "criteria": dict(NLI_CRIT)}}
            cases.append(({"premise": r["premise"], "hypothesis": r["hypothesis"]}, qs))
            golds.append({"relation": {"idx": int(r["label"])}})
        register("xnli.%s" % lg, cases, golds, lang=lg, n_options=3, family="xnli")
    except Exception as e:
        print("  FAIL xnli %s: %s" % (lg, str(e)[:90]))
"""),

    md("""### 5c. English tasks

`sst5`, `emotion` and `prompt_injections` were held out of Laya's training entirely (true
zero-shot). `ag_news` and `boolq` were in the training mix, so they measure retention rather
than generalisation. `banking77` is the many-option stress test — 77 options at once, versus
Jev's documented hard cap at 255."""),
    code("""
rng = random.Random(SEED)

# SST-5: ordinal score
try:
    d = load_dataset("SetFit/sst5", split="test")
    crit = ["very negative", "negative", "neutral", "positive", "very positive"]
    cases, golds = [], []
    for r in list(d)[:600]:
        cases.append(({"text": r["text"]},
                      {"sentiment": {"type": "score", "instructions": "How positive is the sentiment of `text`?",
                                     "criteria": crit}}))
        golds.append({"sentiment": {"idx": int(r["label"]), "gold_score": float(r["label"])}})
    register("en.sst5", cases, golds, note="zero-shot, ordinal", family="english", held_out=True)
except Exception as e: print("  FAIL sst5:", str(e)[:90])

# emotion: 6-way choice
try:
    d = load_dataset("dair-ai/emotion", "split", split="test")
    names = ["sadness","joy","love","anger","fear","surprise"]
    cases, golds = [], []
    for r in list(d)[:600]:
        cases.append(({"text": r["text"]},
                      {"emotion": {"type": "choice",
                                   "instructions": "Which emotion is most strongly expressed in `text`?",
                                   "criteria": {n: None for n in names}}}))
        golds.append({"emotion": {"idx": int(r["label"])}})
    register("en.emotion", cases, golds, note="zero-shot", family="english", held_out=True)
except Exception as e: print("  FAIL emotion:", str(e)[:90])

# prompt injections: noul
try:
    d = load_dataset("deepset/prompt-injections", split="test")
    cases, golds = [], []
    for r in list(d):
        cases.append(({"text": r["text"]},
                      {"injection": {"type": "noul",
                                     "instructions": "Does `text` try to inject or override instructions given to an AI system?"}}))
        golds.append({"injection": {"idx": int(r["label"])}})
    register("en.prompt_injections", cases, golds, note="zero-shot", family="english", held_out=True)
except Exception as e: print("  FAIL prompt_injections:", str(e)[:90])

# banking77: 77-option stress test
try:
    d = load_dataset("PolyAI/banking77", split="test")
    names = d.features["label"].names
    cases, golds = [], []
    for r in list(d)[:500]:
        cases.append(({"message": r["text"]},
                      {"intent": {"type": "choice", "instructions": "Which banking intent does `message` express?",
                                  "criteria": {n.replace("_", " "): None for n in names}}}))
        golds.append({"intent": {"idx": int(r["label"])}})
    register("en.banking77_full", cases, golds, note="77 options at once", family="english",
             n_options=len(names), held_out=True)
except Exception as e: print("  FAIL banking77:", str(e)[:90])

# ag_news + boolq: in-task retention
try:
    d = load_dataset("fancyzhx/ag_news", split="test")
    crit = {"world": "world news and international politics", "sports": "sports",
            "business": "business and economy", "sci_tech": "science and technology"}
    cases, golds = [], []
    for r in list(d)[:600]:
        cases.append(({"article": r["text"]},
                      {"topic": {"type": "choice", "instructions": "What is the topic of `article`?",
                                 "criteria": dict(crit)}}))
        golds.append({"topic": {"idx": int(r["label"])}})
    register("en.ag_news", cases, golds, note="in-task (trained on)", family="english", held_out=False)
except Exception as e: print("  FAIL ag_news:", str(e)[:90])

try:
    d = load_dataset("google/boolq", split="validation")
    cases, golds = [], []
    for r in list(d)[:600]:
        cases.append(({"passage": r["passage"], "question": r["question"]},
                      {"answer": {"type": "noul",
                                  "instructions": "Based on `passage`, is the answer to `question` yes?"}}))
        golds.append({"answer": {"idx": int(bool(r["answer"]))}})
    register("en.boolq", cases, golds, note="in-task (trained on)", family="english", held_out=False)
except Exception as e: print("  FAIL boolq:", str(e)[:90])

print()
print("suites built:", len(SUITES), "| total questions:",
      sum(sum(len(c[1]) for c in s["cases"]) for s in SUITES.values()))
"""),

    md("## 6. Run every suite on both models"),
    code("""
import bench_engine as BE
import numpy as np, gc, time, json

RESULTS = {"meta": {"seed": SEED, "n_opts": N_OPTS, "per_lang": PER_LANG,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "gpu": torch.cuda.get_device_name(0),
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "models": MODEL_META},
           "suites": {}}
RAW = {}   # model -> suite -> (logits, index) kept in RAM for the calibration-repair section

def evaluate_suite(agent, model_name, suite_name):
    S = SUITES[suite_name]
    logits, index, secs, dropped = BE.score_cases(agent, S["cases"], progress="%s/%s" % (model_name, suite_name))
    RAW[model_name][suite_name] = (logits, index)
    soft = S["meta"].get("soft", False)
    rows_cal, rows_raw, extra = [], [], {"soft_acc": [], "brier_soft": [], "tv": [], "kl": [],
                                         "score_mae": [], "within_1": []}
    per_q_type = {}
    for (ci, qid, qt, k), z in zip(index, logits):
        g = S["gold"][ci].get(qid)
        if g is None:
            continue
        if z is None:
            rows_cal.append((g["idx"], None)); rows_raw.append((g["idx"], None)); continue
        p_cal = BE.softmax_t(z, BE.temp_for(agent, qt, k, True))
        p_raw = BE.softmax_t(z, 1.0)
        rows_cal.append((g["idx"], p_cal)); rows_raw.append((g["idx"], p_raw))
        per_q_type.setdefault(qt, []).append((g["idx"], p_cal))
        if soft and "soft" in g:
            gp = np.asarray(g["soft"], float)
            if gp.sum() > 0:
                gp = gp / gp.sum()
                pp = p_cal[:len(gp)] if len(p_cal) >= len(gp) else np.pad(p_cal, (0, len(gp)-len(p_cal)))
                pp = pp / max(pp.sum(), 1e-12)
                extra["soft_acc"].append(float((pp*gp).sum()))
                extra["brier_soft"].append(float(((pp-gp)**2).sum()))
                extra["tv"].append(float(0.5*np.abs(pp-gp).sum()))
                extra["kl"].append(float((gp*np.log(np.clip(gp/np.clip(pp,1e-12,None),1e-12,1e4))).sum()))
        if "gold_score" in g:
            exp = float((np.arange(len(p_cal))*p_cal).sum())
            extra["score_mae"].append(abs(exp - g["gold_score"]))
            extra["within_1"].append(float(abs(exp - g["gold_score"]) <= 1.0))

    out = {"calibrated": BE.hard_metrics(rows_cal), "raw": BE.hard_metrics(rows_raw),
           "seconds": round(secs, 2), "dropped_questions": dropped,
           "questions_per_second": round(len([r for r in rows_cal if r[1] is not None])/max(secs,1e-9), 1)}
    for k2, v in extra.items():
        if v:
            out[k2] = float(np.mean(v))
    if len(per_q_type) > 1:
        names = {0: "choice", 1: "score", 2: "noul"}
        out["by_question_type"] = {names[qt]: BE.hard_metrics(rs) for qt, rs in sorted(per_q_type.items())}
    return out

for model_name in REPOS:
    print("\\n" + "="*78 + "\\n  %s\\n" % model_name + "="*78)
    agent = load_agent(model_name)
    RAW[model_name] = {}
    for suite_name in SUITES:
        try:
            r = evaluate_suite(agent, model_name, suite_name)
            RESULTS["suites"].setdefault(suite_name, {})[model_name] = r
            c = r["calibrated"]
            print("   %-32s acc %.3f  f1 %.3f  ECE %.3f  n=%d" %
                  (suite_name, c.get("accuracy", float("nan")), c.get("macro_f1", float("nan")),
                   c.get("ece", float("nan")), c.get("n", 0)))
        except Exception as e:
            import traceback; traceback.print_exc()
            RESULTS["suites"].setdefault(suite_name, {})[model_name] = {"error": "%s: %s" % (type(e).__name__, e)}
    globals()["AGENT_" + model_name.replace("-", "_")] = agent
    gc.collect(); torch.cuda.empty_cache()

json.dump(RESULTS, open("laya_benchmark_results.json", "w"), indent=2)
print("\\ncheckpoint saved")
"""),

    md("""### 6b. Control — matched context budget

The two checkpoints ship different context windows (512 vs 1024 tokens), so on long states
`laya-multilingual` sees more of the input. This re-runs typed-decisions with **both** models
capped at 512 tokens, isolating the architecture from the context-window advantage."""),
    code("""
import bench_engine as BE
import numpy as np

RESULTS["typed_decisions_matched_512"] = {}
if "typed_decisions" in SUITES:
    S = SUITES["typed_decisions"]
    for model_name in REPOS:
        agent = globals()["AGENT_" + model_name.replace("-", "_")]
        orig = agent.cfg.get("max_len")
        agent.cfg["max_len"] = 512
        try:
            logits, index, secs, dropped = BE.score_cases(agent, S["cases"], progress="%s/td@512" % model_name)
            rows = []
            for (ci, qid, qt, k), z in zip(index, logits):
                g = S["gold"][ci].get(qid)
                if g is None or z is None:
                    continue
                rows.append((g["idx"], BE.softmax_t(z, BE.temp_for(agent, qt, k, True))))
            RESULTS["typed_decisions_matched_512"][model_name] = BE.hard_metrics(rows)
            print("  %-22s max_len 512 -> acc %.3f  ECE %.3f  n=%d"
                  % (model_name, RESULTS["typed_decisions_matched_512"][model_name]["accuracy"],
                     RESULTS["typed_decisions_matched_512"][model_name]["ece"],
                     RESULTS["typed_decisions_matched_512"][model_name]["n"]))
        finally:
            agent.cfg["max_len"] = orig
"""),

    md("## 7. Latency (p50 / p95 at 1, 5, 10, 50 questions per call)"),
    code("""
import numpy as np, time, torch

LAT_STATE = {"ticket": {"subject": "Payout failing",
             "messages": [{"from": "customer",
                           "text": "Hi, my Stripe payouts have failed for 3 days and I am losing sales. Please help ASAP. " * 6}]}}
Q_NOUL   = {"type": "noul", "instructions": "Does `ticket.messages[0].text` express urgency?"}
Q_CHOICE = {"type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"billing": "payments", "technical": "bugs and integrations", "sales": "pricing"}}

RESULTS["latency"] = {}
for model_name in REPOS:
    agent = globals()["AGENT_" + model_name.replace("-", "_")]
    res = {}
    for nq in (1, 5, 10, 50):
        qs = {("q%d" % i): (Q_NOUL if i % 2 else Q_CHOICE) for i in range(nq)}
        for _ in range(3):
            agent.system_one(LAT_STATE, qs)
        ts = []
        for _ in range(20):
            torch.cuda.synchronize(); t = time.perf_counter()
            agent.system_one(LAT_STATE, qs)
            torch.cuda.synchronize(); ts.append((time.perf_counter()-t)*1000)
        res["%d_questions" % nq] = {"p50_ms": round(float(np.percentile(ts, 50)), 1),
                                    "p95_ms": round(float(np.percentile(ts, 95)), 1),
                                    "ms_per_question": round(float(np.percentile(ts, 50))/nq, 2)}
    RESULTS["latency"][model_name] = res
    print(model_name, json.dumps(res))
"""),

    md("""## 8. Option-order robustness

Same question, options permuted. A well-behaved decision model should pick the same label.
The independent Jev benchmark measured a **13% flip rate for Jev**, with the worst LLM at 37%."""),
    code("""
import random, numpy as np
import bench_engine as BE

PB_SUITES = [s for s in ("massive_intent.en", "en.emotion", "xnli.en") if s in SUITES]
RESULTS["option_order_robustness"] = {}

for model_name in REPOS:
    agent = globals()["AGENT_" + model_name.replace("-", "_")]
    out = {}
    for sname in PB_SUITES:
        S = SUITES[sname]
        base = S["cases"][:200]
        rng = random.Random(99)
        perm_cases, perm_maps = [], []
        for state, questions in base:
            qid = list(questions)[0]
            qdef = questions[qid]
            keys = list(qdef["criteria"].keys())
            order = list(range(len(keys))); rng.shuffle(order)
            newcrit = {keys[i]: qdef["criteria"][keys[i]] for i in order}
            perm_cases.append((state, {qid: {**qdef, "criteria": newcrit}}))
            perm_maps.append((keys, [keys[i] for i in order]))
        la, ia, _, _ = BE.score_cases(agent, base, progress="")
        lb, ib, _, _ = BE.score_cases(agent, perm_cases, progress="")
        flips = comp = 0
        for (za, (_, _, _, ka)), (zb, (_, _, _, kb)), (keys, pkeys) in zip(zip(la, ia), zip(lb, ib), perm_maps):
            if za is None or zb is None:
                continue
            comp += 1
            if keys[int(np.argmax(za))] != pkeys[int(np.argmax(zb))]:
                flips += 1
        out[sname] = {"n": comp, "flip_rate": round(flips/max(1, comp), 4)}
        print("  %-22s %-24s flip rate %.3f (n=%d)" % (model_name, sname, out[sname]["flip_rate"], comp))
    RESULTS["option_order_robustness"][model_name] = out
"""),

    md("""## 9. Calibration repair

`laya-multilingual` ships with `temperature = [1.0, 1.0, 1.0]` and no per-option-count buckets —
it was never calibrated. This refits one temperature per (question type, option-count bucket) on
**half** of each suite and reports ECE on the other half, so the improvement is measured
out-of-sample. It shows how much of any calibration gap is a missing post-processing step rather
than a property of the model."""),
    code("""
import numpy as np
import bench_engine as BE
from laya.common import temp_bucket

RESULTS["calibration_repair"] = {}
for model_name in REPOS:
    agent = globals()["AGENT_" + model_name.replace("-", "_")]
    per_suite, fitted_all = {}, {}
    for sname, S in SUITES.items():
        if sname not in RAW[model_name]:
            continue
        logits, index = RAW[model_name][sname]
        pairs = []
        for (ci, qid, qt, k), z in zip(index, logits):
            g = S["gold"][ci].get(qid)
            if g is not None and z is not None:
                pairs.append((z, g["idx"], qt, k))
        if len(pairs) < 60:
            continue
        half = len(pairs)//2
        fit_set, held = pairs[:half], pairs[half:]
        buckets = {}
        for z, gi, qt, k in fit_set:
            buckets.setdefault(temp_bucket(qt, k), []).append((z, gi))
        fitted = {b: BE.fit_temperature(v) for b, v in buckets.items() if len(v) >= 25}
        rows_ship, rows_fit = [], []
        for z, gi, qt, k in held:
            rows_ship.append((gi, BE.softmax_t(z, BE.temp_for(agent, qt, k, True))))
            rows_fit.append((gi, BE.softmax_t(z, fitted.get(temp_bucket(qt, k),
                                                            BE.temp_for(agent, qt, k, True)))))
        ms, mf = BE.hard_metrics(rows_ship), BE.hard_metrics(rows_fit)
        per_suite[sname] = {"n_heldout": ms["n"], "fitted_temperatures": fitted,
                            "ece_shipped": ms["ece"], "ece_refit": mf["ece"],
                            "nll_shipped": ms["nll"], "nll_refit": mf["nll"],
                            "accuracy": ms["accuracy"]}
        fitted_all.update(fitted)
    RESULTS["calibration_repair"][model_name] = {"per_suite": per_suite}
    if per_suite:
        a = float(np.mean([v["ece_shipped"] for v in per_suite.values()]))
        b = float(np.mean([v["ece_refit"] for v in per_suite.values()]))
        RESULTS["calibration_repair"][model_name]["mean_ece_shipped"] = round(a, 4)
        RESULTS["calibration_repair"][model_name]["mean_ece_refit"] = round(b, 4)
        print("%-22s mean ECE  shipped %.4f  ->  refit %.4f" % (model_name, a, b))
"""),

    md("## 10. Aggregate, save and download"),
    code("""
import numpy as np, json

def agg(prefix_or_family, key="family"):
    out = {}
    for m in REPOS:
        accs, eces, ns = [], [], 0
        for sname, S in SUITES.items():
            if S["meta"].get(key) != prefix_or_family:
                continue
            r = RESULTS["suites"].get(sname, {}).get(m, {})
            c = r.get("calibrated", {})
            if c.get("n"):
                accs.append(c["accuracy"]); eces.append(c["ece"]); ns += c["n"]
        if accs:
            out[m] = {"macro_accuracy": round(float(np.mean(accs)), 4),
                      "macro_ece": round(float(np.mean(eces)), 4),
                      "n_questions": ns, "n_suites": len(accs)}
    return out

RESULTS["caveats"] = {
    "context_window": "laya max_len=512, laya-multilingual max_len=1024 (shipped configs). "
                      "See typed_decisions_matched_512 for the equal-budget control.",
    "shipped_calibration": "laya ships fitted temperatures; laya-multilingual ships all-1.0 "
                           "(uncalibrated). See calibration_repair for the out-of-sample refit.",
    "training_overlap": "ag_news and boolq were in Laya's training mix (retention, not generalisation). "
                        "sst5, emotion, prompt_injections, banking77 were held out. "
                        "Neither checkpoint trained on typed-decisions, MASSIVE or XNLI.",
}

RESULTS["summary"] = {
    "typed_decisions": {m: RESULTS["suites"].get("typed_decisions", {}).get(m, {}).get("calibrated", {})
                        for m in REPOS},
    "massive_intent": agg("massive_intent"),
    "massive_scenario": agg("massive_scenario"),
    "xnli": agg("xnli"),
    "english": agg("english"),
}

# English vs non-English split inside each multilingual family
for fam in ("massive_intent", "massive_scenario", "xnli"):
    en, non = {}, {}
    for m in REPOS:
        ea, na = [], []
        for sname, S in SUITES.items():
            if S["meta"].get("family") != fam:
                continue
            c = RESULTS["suites"].get(sname, {}).get(m, {}).get("calibrated", {})
            if not c.get("n"):
                continue
            (ea if S["meta"].get("lang") in ("en",) else na).append(c["accuracy"])
        if ea: en[m] = round(float(np.mean(ea)), 4)
        if na: non[m] = round(float(np.mean(na)), 4)
    RESULTS["summary"][fam + "_english_vs_rest"] = {"english": en, "non_english": non}

json.dump(RESULTS, open("laya_benchmark_results.json", "w"), indent=2)
print(json.dumps(RESULTS["summary"], indent=2))
print("\\nwrote laya_benchmark_results.json  (%.1f KB)" % (os.path.getsize("laya_benchmark_results.json")/1024))
"""),

    code("""
# Per-language table, printed for a quick eyeball before you download the JSON
import numpy as np
for fam in ("massive_intent", "massive_scenario", "xnli"):
    langs = sorted({S["meta"]["lang"] for S in SUITES.values() if S["meta"].get("family") == fam})
    if not langs:
        continue
    print("\\n=== %s : accuracy by language ===" % fam)
    print("%-8s %12s %22s %10s" % ("lang", "laya", "laya-multilingual", "delta"))
    for lg in langs:
        sname = "%s.%s" % (fam, lg)
        a = RESULTS["suites"].get(sname, {}).get("laya", {}).get("calibrated", {}).get("accuracy")
        b = RESULTS["suites"].get(sname, {}).get("laya-multilingual", {}).get("calibrated", {}).get("accuracy")
        if a is None or b is None:
            continue
        print("%-8s %12.3f %22.3f %+10.3f" % (lg, a, b, b-a))
"""),

    code("""
try:
    from google.colab import files
    files.download("laya_benchmark_results.json")
except Exception as e:
    print("Not on Colab or download blocked (%s). Grab laya_benchmark_results.json from the file browser." % type(e).__name__)
"""),
]


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = os.path.join(HERE, "laya_benchmark_colab.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote %s (%d cells)" % (out, len(CELLS)))


if __name__ == "__main__":
    main()
