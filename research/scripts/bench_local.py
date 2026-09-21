"""Local extensive benchmark of the three Laya checkpoints.

Part A  MASSIVE intent across every language the dataset ships (~51), 20-option choice.
Part B  typed-decisions (400 cases / 2,000 decisions) on all three checkpoints, so the
        fine-tuned laya-typed-decisions can be compared with Jev's published 0.727 on the
        same benchmark.

Writes local_benchmark_results.json.

  USE_TF=0 python3 notebooks/bench_local.py [--langs N] [--per-lang N] [--skip-a] [--skip-b]
"""
import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time

os.environ.setdefault("USE_TF", "0")          # TensorFlow's abseil runtime deadlocks model build
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "laya"))

import laya  # noqa: E402
from laya.common import QTYPES, build_sequence, collate_items, render_options, temp_bucket  # noqa: E402

ROOT = os.path.expanduser("~/laya_models")
MODELS = {"english": os.path.join(ROOT, "laya"),
          "multilingual": os.path.join(ROOT, "laya-multilingual"),
          "typed-decisions": os.path.join(ROOT, "laya-typed-decisions")}
OUT = os.path.join(REPO, "local_benchmark_results.json")
SEED, N_OPTS = 13, 20


# ------------------------------------------------------------------ engine
def to_internal(qdef):
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    return {"t": t, "ins": ins if isinstance(ins, str) else json.dumps(ins), "crit": crit}


@torch.no_grad()
def score_cases(agent, cases, max_tokens=8192, max_seqs=64, tag=""):
    max_len = agent.cfg.get("max_len", 512)
    hml = agent.cfg.get("head_max_len", 192)
    items, index, dropped = [], [], 0
    for ci, (state, questions) in enumerate(cases):
        for qid, qdef in questions.items():
            q = to_internal(qdef)
            try:
                ids, mk = build_sequence(agent.tok, state, q, max_len, hml)
            except Exception:
                index.append((ci, qid, QTYPES[q["t"]], 0)); items.append(None); dropped += 1; continue
            if len(mk) != len(render_options(q)):
                index.append((ci, qid, QTYPES[q["t"]], 0)); items.append(None); dropped += 1; continue
            items.append({"ids": ids, "markers": mk, "qtype": QTYPES[q["t"]]})
            index.append((ci, qid, QTYPES[q["t"]], len(mk)))
    order = sorted([i for i, it in enumerate(items) if it is not None],
                   key=lambda i: len(items[i]["ids"]))
    out = [None] * len(items)
    t0, done, i = time.time(), 0, 0
    while i < len(order):
        j, L = i, 0
        while j < len(order) and j - i < max_seqs and \
                max(L, len(items[order[j]]["ids"])) * (j - i + 1) <= max_tokens:
            L = max(L, len(items[order[j]]["ids"])); j += 1
        j = max(j, i + 1)
        sel = [items[order[t]] for t in range(i, j)]
        b = collate_items([sel], agent.tok.pad_token_id)
        lg, _ = agent.model(b["input_ids"].to(agent.device), b["attention_mask"].to(agent.device),
                            b["marker_pos"].to(agent.device), b["marker_mask"].to(agent.device),
                            b["qtype"].to(agent.device))
        lg = lg.float().cpu().numpy()
        for r in range(j - i):
            out[order[i + r]] = lg[r, :len(sel[r]["markers"])]
        done += j - i
        if tag and done % 500 < (j - i):
            el = time.time() - t0
            sys.stderr.write("\r   [%s] %d/%d %.0f q/s ETA %ds    "
                             % (tag, done, len(order), done / max(el, 1e-9),
                                int(el * (len(order) - done) / max(1, done))))
            sys.stderr.flush()
        i = j
    if tag:
        sys.stderr.write("\r" + " " * 70 + "\r")
    return out, index, time.time() - t0, dropped


def softmax_t(z, t=1.0):
    z = np.asarray(z, float) / max(1e-3, float(t))
    e = np.exp(z - z.max())
    return e / e.sum()


def temp_for(agent, qt, k):
    return float(agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt]))


def ece_score(conf, corr, bins=15):
    conf, corr = np.asarray(conf, float), np.asarray(corr, float)
    if not len(conf):
        return float("nan")
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(conf[s].mean() - corr[s].mean())
    return float(e)


def macro_f1(g, p):
    g, p = np.asarray(g), np.asarray(p)
    f = []
    for c in sorted(set(g.tolist()) | set(p.tolist())):
        tp = int(((p == c) & (g == c)).sum()); fp = int(((p == c) & (g != c)).sum())
        fn = int(((p != c) & (g == c)).sum())
        f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f))


def metrics(rows):
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        return {"n": 0}
    g = np.array([x[0] for x in rows]); p = np.array([int(np.argmax(x[1])) for x in rows])
    c = np.array([float(np.max(x[1])) for x in rows]); corr = (p == g).astype(float)
    return {"n": len(rows), "accuracy": round(float(corr.mean()), 4),
            "macro_f1": round(macro_f1(g, p), 4), "ece": round(ece_score(c, corr), 4),
            "brier": round(float(np.mean([((np.asarray(x[1]) - np.eye(len(x[1]))[x[0]]) ** 2).sum()
                                          for x in rows])), 4),
            "nll": round(float(np.mean([-math.log(max(float(x[1][x[0]]), 1e-12)) for x in rows])), 4),
            "mean_confidence": round(float(c.mean()), 4),
            "acc_at_50_coverage": round(float(corr[np.argsort(-c)[:max(1, len(c)//2)]].mean()), 4)}


def load(name):
    ag = laya.load(MODELS[name], device="cpu")
    ag.model.eval()
    return ag


# ------------------------------------------------------------------ part A
def massive_languages():
    from huggingface_hub import HfApi
    files = [s.rfilename for s in (HfApi().dataset_info("mteb/amazon_massive_intent").siblings or [])]
    return sorted({m.group(1) for f in files for m in [re.match(r"test/([A-Za-z\-]+)\.json", f)] if m})


def build_massive(langs, per_lang):
    from datasets import load_dataset
    suites = {}
    for lg in langs:
        try:
            d = load_dataset("mteb/amazon_massive_intent", lg, split="test")
            labels = sorted(set(d["label_text"]))
            rng = random.Random(SEED)
            cases, gold = [], []
            for r in list(d)[:per_lang]:
                pool = [x for x in labels if x != r["label_text"]]
                keys = [r["label_text"]] + rng.sample(pool, min(N_OPTS - 1, len(pool)))
                rng.shuffle(keys)
                cases.append(({"utterance": r["text"]},
                              {"intent": {"type": "choice",
                                          "instructions": "What is the user asking for in `utterance`?",
                                          "criteria": {k: k.replace("_", " ").replace(".", ": ") for k in keys}}}))
                gold.append(keys.index(r["label_text"]))
            suites[lg] = (cases, gold, len(labels))
            print("   built %-8s %d cases (%d labels)" % (lg, len(cases), len(labels)), flush=True)
        except Exception as e:
            print("   FAIL %-8s %s" % (lg, str(e)[:70]), flush=True)
    return suites


def run_part_a(results, langs, per_lang):
    print("\n=== PART A: MASSIVE intent, %d languages, %d cases each, %d options ===\n"
          % (len(langs), per_lang, N_OPTS), flush=True)
    suites = build_massive(langs, per_lang)
    results["part_a"] = {"config": {"languages": sorted(suites), "per_lang": per_lang,
                                    "n_options": N_OPTS, "seed": SEED}, "by_model": {}}
    for mname in ("english", "multilingual"):
        print("\n--- %s ---" % mname, flush=True)
        ag = load(mname)
        per = {}
        for lg, (cases, gold, _) in sorted(suites.items()):
            lgs, idx, secs, dropped = score_cases(ag, cases, tag="%s/%s" % (mname, lg))
            rows = [(gold[ci], softmax_t(z, temp_for(ag, qt, k)) if z is not None else None)
                    for (ci, _, qt, k), z in zip(idx, lgs)]
            m = metrics(rows); m["seconds"] = round(secs, 1); m["dropped"] = dropped
            per[lg] = m
            print("   %-8s acc %.3f  f1 %.3f  ECE %.3f  conf %.3f  (%.0f q/s)"
                  % (lg, m["accuracy"], m["macro_f1"], m["ece"], m["mean_confidence"],
                     m["n"] / max(secs, 1e-9)), flush=True)
        accs = [v["accuracy"] for v in per.values()]
        results["part_a"]["by_model"][mname] = {
            "per_language": per,
            "macro_accuracy": round(float(np.mean(accs)), 4),
            "macro_ece": round(float(np.mean([v["ece"] for v in per.values()])), 4),
            "n_languages": len(per),
            "languages_above_random": int(sum(1 for a in accs if a > 3.0 / N_OPTS)),
        }
        print("   MACRO acc %.4f | ECE %.4f | %d/%d langs > 3x random"
              % (results["part_a"]["by_model"][mname]["macro_accuracy"],
                 results["part_a"]["by_model"][mname]["macro_ece"],
                 results["part_a"]["by_model"][mname]["languages_above_random"], len(per)), flush=True)
        del ag; gc.collect()
        json.dump(results, open(OUT, "w"), indent=2)


# ------------------------------------------------------------------ part B
def build_typed_decisions():
    import pandas as pd
    p = os.path.join(REPO, "typed-decisions", "all", "test-00000-of-00001.parquet")
    df = pd.read_parquet(p)
    cases, gold, wfs = [], [], []
    for _, r in df.iterrows():
        qs = json.loads(r["questions"]); g = json.loads(r["gold"])
        st = r["state"]
        try:
            st = json.loads(st)
        except Exception:
            pass
        gm = {}
        for qid, qd in qs.items():
            gg = g[qid]
            if qd["type"] == "choice":
                keys = list(qd["criteria"].keys())
                gm[qid] = {"idx": keys.index(str(gg["label"])),
                           "soft": [float(gg.get("probabilities", {}).get(k, 0.0)) for k in keys]}
            elif qd["type"] == "noul":
                pt = float(gg.get("probabilities", {}).get("true", gg.get("noul", 0.5)))
                gm[qid] = {"idx": 1 if str(gg["label"]).lower() == "true" else 0, "soft": [1 - pt, pt]}
            else:
                n = len(qd["criteria"])
                gm[qid] = {"idx": int(gg["label"]),
                           "soft": [float(gg.get("probabilities", {}).get(str(i), 0.0)) for i in range(n)],
                           "gold_score": float(gg.get("score", float(gg["label"])))}
        cases.append((st, qs)); gold.append(gm); wfs.append(r["workflow"])
    return cases, gold, wfs


def run_part_b(results):
    print("\n=== PART B: typed-decisions, 400 cases / 2,000 decisions, all 3 checkpoints ===\n",
          flush=True)
    cases, gold, wfs = build_typed_decisions()
    # reference points that are NOT measured here - published / independently measured
    results["part_b"] = {"reference_points": {
        "jev_1.13.0_published": {"accuracy": 0.727, "soft_accuracy": 0.580, "brier": 0.148,
                                 "ece": 0.144, "score_mae": 0.391, "ms_per_case": 710,
                                 "source": "figure quoted in the laya repo's own comparison table"},
        "teacher_self_agreement_ceiling": {"accuracy": 0.735},
        "modernbert_base_specialist": {"accuracy": 0.646},
        "random_guess": {"accuracy": 0.3175},
        "per_question_majority_class": {"accuracy": 0.4610},
        "note": "Jev was NOT run here - no TypeSafe API credential is available. These are "
                "published numbers reproduced for context, not a measured head-to-head.",
    }, "by_model": {}}
    for mname in ("english", "multilingual", "typed-decisions"):
        print("--- %s ---" % mname, flush=True)
        ag = load(mname)
        lgs, idx, secs, dropped = score_cases(ag, cases, tag=mname)
        rows, soft, brier_s, mae, w1, by_wf, by_qt = [], [], [], [], [], {}, {}
        for (ci, qid, qt, k), z in zip(idx, lgs):
            g = gold[ci][qid]
            if z is None:
                rows.append((g["idx"], None)); continue
            p = softmax_t(z, temp_for(ag, qt, k))
            rows.append((g["idx"], p))
            by_wf.setdefault(wfs[ci], []).append((g["idx"], p))
            by_qt.setdefault({0: "choice", 1: "score", 2: "noul"}[qt], []).append((g["idx"], p))
            gp = np.asarray(g["soft"], float)
            if gp.sum() > 0:
                gp = gp / gp.sum()
                pp = p[:len(gp)] if len(p) >= len(gp) else np.pad(p, (0, len(gp) - len(p)))
                pp = pp / max(pp.sum(), 1e-12)
                soft.append(float((pp * gp).sum())); brier_s.append(float(((pp - gp) ** 2).sum()))
            if "gold_score" in g:
                exp = float((np.arange(len(p)) * p).sum())
                mae.append(abs(exp - g["gold_score"])); w1.append(float(abs(exp - g["gold_score"]) <= 1))
        m = metrics(rows)
        m.update(soft_accuracy=round(float(np.mean(soft)), 4) if soft else None,
                 brier_vs_soft=round(float(np.mean(brier_s)), 4) if brier_s else None,
                 score_mae=round(float(np.mean(mae)), 4) if mae else None,
                 within_1_level=round(float(np.mean(w1)), 4) if w1 else None,
                 seconds=round(secs, 1), dropped=dropped,
                 ms_per_case=round(1000 * secs / len(cases), 1),
                 by_workflow={k: metrics(v) for k, v in sorted(by_wf.items())},
                 by_question_type={k: metrics(v) for k, v in sorted(by_qt.items())})
        results["part_b"]["by_model"][mname] = m
        print("   acc %.4f | soft %.4f | brier(soft) %s | ECE %.4f | MAE %s | %.0f ms/case"
              % (m["accuracy"], m["soft_accuracy"] or 0, m["brier_vs_soft"], m["ece"],
                 m["score_mae"], m["ms_per_case"]), flush=True)
        for wf, v in m["by_workflow"].items():
            print("      %-28s acc %.3f (n=%d)" % (wf, v["accuracy"], v["n"]), flush=True)
        del ag; gc.collect()
        json.dump(results, open(OUT, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", type=int, default=0, help="cap number of languages (0 = all)")
    ap.add_argument("--per-lang", type=int, default=120)
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    a = ap.parse_args()

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "device": "cpu",
                        "torch": torch.__version__, "laya": laya.__version__,
                        "threads": torch.get_num_threads()}}
    if os.path.exists(OUT):
        try:
            results = json.load(open(OUT)); results["meta"]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    if not a.skip_a:
        langs = massive_languages()
        if a.langs:
            langs = langs[:a.langs]
        run_part_a(results, langs, a.per_lang)
    if not a.skip_b:
        run_part_b(results)
    json.dump(results, open(OUT, "w"), indent=2)
    print("\nwrote %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
