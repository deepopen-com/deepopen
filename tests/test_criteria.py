"""Criteria rendering: structured values must not crash or leak Python reprs.

Regression tests for the bug reported in PR #2 (trocker): a `noul` question whose criteria
values were dicts raised `TypeError: can only concatenate str (not "dict") to str`, and
`choice`/`score` stringified dicts as Python reprs instead of JSON.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import render_criterion, render_options  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


# --------------------------------------------------------------- render_criterion
check("criterion/str passes through", render_criterion("phishing or scam"), "phishing or scam")
check("criterion/dict -> json", render_criterion({"desc": "phishing"}), '{"desc": "phishing"}')
check("criterion/list -> json", render_criterion(["a", "b"]), '["a", "b"]')
check("criterion/int -> json", render_criterion(3), "3")
check("criterion/bool -> json", render_criterion(False), "false")
check("criterion/non-ascii kept", render_criterion({"d": "münchen"}), '{"d": "münchen"}')
check_true("criterion/unserialisable falls back to str",
           isinstance(render_criterion({"o": object()}), str))


# --------------------------------------------------------------- the reported crash
q = {"t": "noul", "ins": "Is this phishing?",
     "crit": {"true": {"desc": "phishing, scam or fraud"}, "false": {"desc": "legitimate"}}}
try:
    out = render_options(q)
    check("noul/dict criteria does not crash", len(out), 2)
    check_true("noul/false renders as json", out[0] == 'false: {"desc": "legitimate"}', out[0])
    check_true("noul/true renders as json", out[1] == 'true: {"desc": "phishing, scam or fraud"}', out[1])
    check_true("noul/no python repr leaked", "'" not in "".join(out), out)
except TypeError as e:
    FAIL.append("noul/dict criteria CRASHED: %s" % e)


# --------------------------------------------------------------- choice and score
out = render_options({"t": "choice", "ins": "x",
                      "crit": {"billing": {"desc": "payments"}, "tech": None, "sales": ""}})
check("choice/dict -> json", out[0], 'billing: {"desc": "payments"}')
check("choice/None -> bare key", out[1], "tech")
check("choice/empty string -> bare key", out[2], "sales")
check_true("choice/no python repr", "{'" not in "".join(out), out)

# 0 and False are real criterion values, not "missing"
out = render_options({"t": "choice", "ins": "x", "crit": {"zero": 0, "no": False}})
check("choice/0 is kept", out[0], "zero: 0")
check("choice/False is kept", out[1], "no: false")

out = render_options({"t": "score", "ins": "x", "crit": [{"d": "low"}, "high", 2]})
check("score/dict level -> json", out[0], 'level 0: {"d": "low"}')
check("score/str level unchanged", out[1], "level 1: high")
check("score/int level -> json", out[2], "level 2: 2")


# --------------------------------------------------------------- unchanged behaviour
check("noul/default false text", render_options({"t": "noul", "ins": "x", "crit": None})[0],
      "false: no, the statement does not hold")
check("noul/default true text", render_options({"t": "noul", "ins": "x", "crit": None})[1],
      "true: yes, the statement holds")
check("noul/string criteria still work",
      render_options({"t": "noul", "ins": "x", "crit": {"true": "yes it is", "false": "no"}}),
      ["false: no", "true: yes it is"])
check("choice/string criteria still work",
      render_options({"t": "choice", "ins": "x", "crit": {"a": "first", "b": None}}),
      ["a: first", "b"])
check("score/string criteria still work",
      render_options({"t": "score", "ins": "x", "crit": ["low", "high"]}),
      ["level 0: low", "level 1: high"])

# every rendered option must be a str, whatever went in
for qq in [{"t": "choice", "ins": "x", "crit": {"a": {"n": 1}, "b": [1, 2], "c": 3.5}},
           {"t": "score", "ins": "x", "crit": [{"a": 1}, [2], None]},
           {"t": "noul", "ins": "x", "crit": {"true": [1], "false": {"z": 0}}}]:
    check_true("all options are str (%s)" % qq["t"],
               all(isinstance(o, str) for o in render_options(qq)))

# the JSON we emit is parseable back
parsed = json.loads(render_options(
    {"t": "noul", "ins": "x", "crit": {"true": {"a": 1}, "false": {"b": 2}}})[1].split("true: ", 1)[1])
check("emitted json round-trips", parsed, {"a": 1})


# --------------------------------------------------------------- CPU-fallback warning (#9 follow-up)
# The warning must fire only when a fallback actually happened -- not merely because the machine
# has CUDA. `laya.load(path, device="cpu")` on a GPU box is a deliberate choice, not a problem.
import inspect  # noqa: E402

from laya import agent as _agent  # noqa: E402

_src = inspect.getsource(_agent.Agent.__init__)
check_true("fallback/flag is initialised", "fell_back_from = fell_back_why = None" in _src)
check_true("fallback/warns only on a real fallback", "if fell_back_from is not None:" in _src)
check_true("fallback/reports the underlying reason", "Reason: %s" in _src)
check_true("fallback/keeps the actionable advice", "download.pytorch.org/whl/nightly" in _src)
check_true("fallback/no bare cuda probe for the warning",
           "torch.cuda.is_available() or getattr(torch.version" not in _src)


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
