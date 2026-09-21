"""Inference-speed benchmark, including what routing actually costs.

Routing changes the latency picture in two ways:
  * every call pays language detection (pure Python, no model)
  * a call that switches checkpoint pays a model load, unless that model is already resident

So there are two regimes -- hot (right model already loaded) and cold (swap) -- and the realistic
number for a mixed-language workload sits between them, determined by `max_loaded`.

  USE_TF=0 python3 notebooks/bench_latency.py
"""
import gc
import json
import os
import statistics
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "laya"))

import laya  # noqa: E402
from laya.lang import analyse  # noqa: E402
from laya.router import Router  # noqa: E402

ROOT = os.path.expanduser("~/laya_models")
MODELS = {"english": os.path.join(ROOT, "laya"),
          "multilingual": os.path.join(ROOT, "laya-multilingual"),
          "typed-decisions": os.path.join(ROOT, "laya-typed-decisions")}
OUT = os.path.join(REPO, "latency_benchmark_results.json")

STATE_EN = {"ticket": {"subject": "Payout failing", "messages": [{"from": "customer",
            "text": "Hi, my Stripe payouts have failed for 3 days and I am losing sales. Please help ASAP. " * 6}]}}
STATE_HI = {"ticket": {"subject": "भुगतान विफल", "messages": [{"from": "customer",
            "text": "मेरा भुगतान तीन दिनों से विफल हो रहा है और मुझे नुकसान हो रहा है, कृपया तुरंत मदद करें। " * 6}]}}
Q_NOUL = {"type": "noul", "instructions": "Does `ticket.messages[0].text` express urgency?"}
Q_CHOICE = {"type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"billing": "payments", "technical": "bugs and integrations", "sales": "pricing"}}


def qs(n):
    return {("q%d" % i): (Q_NOUL if i % 2 else Q_CHOICE) for i in range(n)}


def timed(fn, warmup=3, reps=15):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": round(float(np.percentile(ts, 50)), 2),
            "p95_ms": round(float(np.percentile(ts, 95)), 2),
            "mean_ms": round(float(np.mean(ts)), 2),
            "min_ms": round(float(np.min(ts)), 2)}


def main():
    res = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "device": "cpu",
                    "torch": torch.__version__, "threads": torch.get_num_threads(),
                    "laya": laya.__version__,
                    "note": "CPU numbers. GPU (T4) reference from the Colab run is in "
                            "laya_benchmark_results.json -> latency."}}

    # ---------------------------------------------------------------- 1. detection overhead
    print("=== 1. language detection overhead (no model) ===", flush=True)
    det = {}
    for label, st in (("english", STATE_EN), ("hindi", STATE_HI),
                      ("short english", {"m": "refund me"}),
                      ("large json", {"rows": [{"id": i, "text": "some ticket body here"} for i in range(200)]})):
        det[label] = timed(lambda s=st: analyse(s), warmup=20, reps=200)
        print("   %-16s p50 %7.3f ms   p95 %7.3f ms" % (label, det[label]["p50_ms"], det[label]["p95_ms"]), flush=True)
    res["detection_overhead"] = det

    # ---------------------------------------------------------------- 2. per-model raw latency
    print("\n=== 2. raw Agent.system_one latency (no routing) ===", flush=True)
    raw, load_times = {}, {}
    for mname, path in MODELS.items():
        t = time.perf_counter()
        ag = laya.load(path, device="cpu")
        load_times[mname] = round((time.perf_counter() - t) * 1000, 1)
        per = {}
        for n in (1, 5, 10, 50):
            q = qs(n)
            r = timed(lambda: ag.system_one(STATE_EN, q), warmup=2, reps=10)
            r["ms_per_question"] = round(r["p50_ms"] / n, 2)
            per["%d_questions" % n] = r
            print("   %-16s %2dq  p50 %8.1f ms  p95 %8.1f ms  %6.2f ms/q"
                  % (mname, n, r["p50_ms"], r["p95_ms"], r["ms_per_question"]), flush=True)
        raw[mname] = per
        print("   %-16s cold load: %.0f ms" % (mname, load_times[mname]), flush=True)
        del ag; gc.collect()
    res["raw_latency"] = raw
    res["cold_load_ms"] = load_times

    # ---------------------------------------------------------------- 3. router hot path
    print("\n=== 3. Router hot path (target model already resident) ===", flush=True)
    r3 = Router(models=MODELS, device="cpu", max_loaded=3)
    r3.predict(STATE_EN, qs(5))      # warm english
    r3.predict(STATE_HI, qs(5))      # warm multilingual
    hot = {}
    for label, st in (("english", STATE_EN), ("hindi", STATE_HI)):
        for n in (1, 10):
            q = qs(n)
            k = "%s_%dq" % (label, n)
            hot[k] = timed(lambda s=st, qq=q: r3.predict(s, qq), warmup=2, reps=10)
            hot[k]["ms_per_question"] = round(hot[k]["p50_ms"] / n, 2)
            print("   %-10s %2dq  p50 %8.1f ms  (%.2f ms/q)" % (label, n, hot[k]["p50_ms"],
                                                                hot[k]["ms_per_question"]), flush=True)
    res["router_hot"] = hot

    # overhead vs raw, like for like
    base_en = raw["english"]["10_questions"]["p50_ms"]
    res["routing_overhead_hot_ms"] = round(hot["english_10q"]["p50_ms"] - base_en, 2)
    print("   routing overhead when hot: %+.2f ms on a 10-question English call"
          % res["routing_overhead_hot_ms"], flush=True)

    # ---------------------------------------------------------------- 4. router cold path (swap)
    print("\n=== 4. Router cold path (max_loaded=1 forces a swap on every language flip) ===", flush=True)
    r1 = Router(models=MODELS, device="cpu", max_loaded=1)
    r1.predict(STATE_EN, qs(5))
    swap = []
    for i in range(4):
        st = STATE_HI if i % 2 == 0 else STATE_EN
        t = time.perf_counter()
        r1.predict(st, qs(10))
        swap.append((time.perf_counter() - t) * 1000)
    res["router_cold_swap"] = {"samples_ms": [round(x, 1) for x in swap],
                               "median_ms": round(statistics.median(swap), 1)}
    print("   swap calls (10q): %s  median %.0f ms"
          % ([round(x) for x in swap], res["router_cold_swap"]["median_ms"]), flush=True)

    # ---------------------------------------------------------------- 5. mixed workload
    print("\n=== 5. mixed-language workload, 100 calls of 5 questions ===", flush=True)
    import random
    rng = random.Random(13)
    mix = {}
    for share in (0.0, 0.1, 0.3, 0.5):
        stream = [(STATE_HI if rng.random() < share else STATE_EN) for _ in range(100)]
        for cap, tag in ((1, "max_loaded=1"), (3, "max_loaded=3")):
            rr = Router(models=MODELS, device="cpu", max_loaded=cap)
            rr.predict(STATE_EN, qs(5)); rr.predict(STATE_HI, qs(5))   # warm what fits
            q = qs(5)
            t = time.perf_counter()
            for st in stream:
                rr.predict(st, q)
            el = (time.perf_counter() - t)
            key = "%d%%_non_english/%s" % (int(share * 100), tag)
            mix[key] = {"total_s": round(el, 2), "mean_ms_per_call": round(el * 1000 / len(stream), 1),
                        "calls_per_s": round(len(stream) / el, 1)}
            print("   %-32s %6.1f ms/call   %5.1f calls/s"
                  % (key, mix[key]["mean_ms_per_call"], mix[key]["calls_per_s"]), flush=True)
            rr.unload(); del rr; gc.collect()
    res["mixed_workload"] = mix

    json.dump(res, open(OUT, "w"), indent=2)
    print("\nwrote %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
