"""Routing and language-detection tests. No model weights are loaded: `Router.route` is pure."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.lang import analyse, detect_script, guess_latin_language, is_english, state_text  # noqa: E402
from laya.router import (  # noqa: E402
    BUNDLE_REPO,
    DEFAULT_MODELS,
    STANDALONE_MODELS,
    _repo_str,
    Router,
    match_typed_decisions_workflow,
    normalise_name,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


# --------------------------------------------------------------------- script detection
SCRIPTS = [
    ("english", "The customer was charged twice and wants a refund.", "latin"),
    ("armenian", "Հայերեն", "armenian"),
    ("armenian uppercase", "ՀԱՅԵՐԵՆ", "armenian"),
    ("armenian punctuation only", "։֊", "unknown"),
    ("french", "Le client a été facturé deux fois et demande un remboursement.", "latin"),
    ("hindi", "ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।", "devanagari"),
    ("japanese", "お客様は二重に請求されたため返金を希望しています。", "kana"),
    ("chinese", "客户被重复扣款要求退款", "han"),
    ("korean", "고객이 두 번 청구되어 환불을 원합니다", "hangul"),
    ("arabic", "تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال", "arabic"),
    ("tamil", "வாடிக்கையாளரிடம் இருமுறை கட்டணம் வசூலிக்கப்பட்டது", "tamil"),
    ("russian", "С клиента дважды сняли деньги и он хочет возврат", "cyrillic"),
    ("thai", "ลูกค้าถูกเรียกเก็บเงินสองครั้งและต้องการเงินคืน", "thai"),
    ("greek", "Ο πελάτης χρεώθηκε δύο φορές και θέλει επιστροφή χρημάτων", "greek"),
    ("hebrew", "הלקוח חויב פעמיים ורוצה החזר כספי", "hebrew"),
    ("empty", "", "unknown"),
    ("digits only", "12345 6789", "unknown"),
]
for label, text, want in SCRIPTS:
    check("script/" + label, detect_script(text), want)


# --------------------------------------------------------------------- english vs not
for label, text, want in [
    ("plain english", "Please refund the duplicate charge on invoice 4411 today.", True),
    ("armenian", "Հայերեն", False),
    ("english short", "refund me", True),
    ("hindi", "ग्राहक से दो बार शुल्क लिया गया", False),
    ("japanese", "お客様は二重に請求されました", False),
    ("russian", "С клиента дважды сняли деньги", False),
    ("french long", "Le client a été facturé deux fois et il demande un remboursement pour la "
                    "facture qui a été payée le mois dernier avec la carte de crédit", False),
    ("german long", "Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die "
                    "Rechnung die nicht korrekt ist und auch nicht bezahlt wurde", False),
]:
    check("is_english/" + label, is_english(text), want)


# --------------------------------------------------------------------- Latin language guess
for label, text, want in [
    ("english", "The customer was charged twice and wants a refund for this invoice", "en"),
    ("french", "Le client a ete facture deux fois et il demande un remboursement pour la facture", "fr"),
    ("german", "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung", "de"),
    ("spanish", "El cliente fue cobrado dos veces y quiere que le devuelvan el dinero por la factura", "es"),
    ("too short", "refund", None),
]:
    check("latin_lang/" + label, guess_latin_language(text), want)
# a non-English guess must never fire on ordinary English
check("latin_lang/long english stays en",
      guess_latin_language("Please refund the duplicate charge on invoice 4411 today because "
                           "we have been waiting for three days and nobody has replied to us"), "en")


# --------------------------------------------------------------------- state flattening
check("state_text/dict", "charged twice" in state_text({"body": "charged twice", "n": 3}), True)
check("state_text/nested", "deep" in state_text({"a": {"b": ["deep"]}}), True)
check("state_text/list", "x" in state_text(["x", {"y": "z"}]), True)
check("state_text/none", state_text(None), "")
# keys must not drive detection: English keys around Hindi content stay non-English
check("state_text/keys ignored",
      analyse({"subject": "नमस्ते", "body": "ग्राहक से दो बार शुल्क लिया गया"})["is_english"], False)


# --------------------------------------------------------------------- workflow signatures
check("profile/armenian", analyse("Հայերեն")["script_profile"], {"armenian": 1.0})
check("profile/armenian mixed with Latin", analyse("Հայերեն abc")["non_latin_fraction"], 0.7)

TD = {
    "agent_trace_observability": ["action", "needs_review", "outcome", "risk", "urgency"],
    "customer_service": ["action", "category", "churn_risk", "needs_human", "urgency"],
    "invoice_processing": ["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"],
    "security_incidents": ["credential_compromise", "disposition", "severity", "true_positive", "urgency"],
}
for wf, ids in TD.items():
    check("workflow/" + wf, match_typed_decisions_workflow({i: {} for i in ids}), wf)
check("workflow/partial overlap", match_typed_decisions_workflow({"urgency": {}, "category": {}}), None)
check("workflow/superset", match_typed_decisions_workflow({i: {} for i in TD["customer_service"] + ["extra"]}), None)
check("workflow/empty", match_typed_decisions_workflow({}), None)


# --------------------------------------------------------------------- name normalisation
for alias, want in [("en", "english"), ("laya", "english"), ("multi", "multilingual"),
                    ("ML", "multilingual"), ("typed", "typed-decisions"),
                    ("typed_decisions", "typed-decisions"), ("English", "english"),
                    ("convaiinnovations/laya".split("/")[-1], "english")]:
    check("alias/" + alias, normalise_name(alias), want)
try:
    normalise_name("nope")
    FAIL.append("alias/unknown: should have raised")
except ValueError:
    PASS.append("alias/unknown raises")


# --------------------------------------------------------------------- routing decisions
r = Router()
Q_GENERIC = {"dept": {"type": "choice", "instructions": "Which team?",
                      "criteria": {"billing": None, "tech": None}}}
Q_TD = {i: {"type": "noul", "instructions": "x"} for i in TD["customer_service"]}

cases = [
    ("english text", {"body": "I was charged twice, please refund."}, Q_GENERIC, {}, "english"),
    ("armenian text", {"body": "Հայերեն"}, Q_GENERIC, {}, "multilingual"),
    ("armenian explicit override", {"body": "Հայերեն"}, Q_GENERIC, {"model": "english"}, "english"),
    ("hindi text", {"body": "मुझसे दो बार शुल्क लिया गया"}, Q_GENERIC, {}, "multilingual"),
    ("japanese text", {"body": "二重に請求されました"}, Q_GENERIC, {}, "multilingual"),
    ("korean text", {"body": "두 번 청구되었습니다"}, Q_GENERIC, {}, "multilingual"),
    ("arabic text", {"body": "تم خصم المبلغ مرتين"}, Q_GENERIC, {}, "multilingual"),
    ("german text", {"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung "
                             "fuer die Rechnung die nicht korrekt ist"}, Q_GENERIC, {}, "multilingual"),
    ("explicit model", {"body": "anything"}, Q_GENERIC, {"model": "multilingual"}, "multilingual"),
    ("explicit model overrides script", {"body": "मुझसे दो बार"}, Q_GENERIC,
     {"model": "english"}, "english"),
    ("explicit task", {"body": "x"}, Q_GENERIC, {"task": "typed_decisions"}, "typed-decisions"),
    ("explicit lang en", {"body": "मुझसे दो बार"}, Q_GENERIC, {"lang": "en"}, "english"),
    ("explicit lang de", {"body": "hello there"}, Q_GENERIC, {"lang": "de"}, "multilingual"),
    ("td workflow, auto OFF", {"body": "I was charged twice"}, Q_TD, {}, "english"),
    ("empty state", {}, Q_GENERIC, {}, "english"),
    ("none state", None, Q_GENERIC, {}, "english"),
]
for label, state, qs, kw, want in cases:
    check("route/" + label, r.route(state, qs, **kw)["model"], want)

# auto task detection is opt-in
r_auto = Router(auto_task_detection=True)
check("route/td workflow, auto ON",
      r_auto.route({"body": "I was charged twice"}, Q_TD)["model"], "typed-decisions")
check("route/auto ON but generic questions",
      r_auto.route({"body": "I was charged twice"}, Q_GENERIC)["model"], "english")
# explicit model still beats auto-detected workflow
check("route/explicit beats workflow",
      r_auto.route({"body": "x"}, Q_TD, model="multilingual")["model"], "multilingual")

# decision payload shape
d = r.route({"body": "मुझसे दो बार शुल्क लिया गया"}, Q_GENERIC)
check("decision/has repo", d["repo"], "convaiinnovations/laya/multilingual")
check("decision/has reason", isinstance(d["reason"], str) and len(d["reason"]) > 0, True)
check("decision/detection script", d["detection"]["script"], "devanagari")
check("decision/.model property", d.model, "multilingual")
check("decision/is dict", isinstance(d, dict), True)

# default override
check("route/custom default", Router(default="multilingual").route("12345", Q_GENERIC)["model"],
      "multilingual")


# --------------------------------------------------------------------- LRU bookkeeping
class _Stub:
    def __init__(self, name):
        self.name = name

    def system_one(self, state, questions):
        return {"model": self.name, "answers": {}, "usage": {}}


def stubbed_router(max_loaded):
    rr = Router(max_loaded=max_loaded)
    rr.load = lambda n, _r=rr: _load_stub(_r, n)
    return rr


def _load_stub(rr, name):
    key = normalise_name(name)
    if key in rr._agents:
        rr._touch(key)
        return rr._agents[key]
    rr._agents[key] = _Stub(key)
    rr._order.append(key)
    rr._evict()
    return rr._agents[key]


rr = stubbed_router(1)
rr.load("english"); rr.load("multilingual")
check("lru/cap 1 keeps newest", rr.loaded, ["multilingual"])
check("lru/cap 1 agents match order", sorted(rr._agents), ["multilingual"])

rr = stubbed_router(2)
rr.load("english"); rr.load("multilingual"); rr.load("typed-decisions")
check("lru/cap 2 evicts oldest", rr.loaded, ["multilingual", "typed-decisions"])

rr = stubbed_router(2)
rr.load("english"); rr.load("multilingual"); rr.load("english")   # touch english
rr.load("typed-decisions")
check("lru/touch protects", sorted(rr.loaded), ["english", "typed-decisions"])

rr.unload("english")
check("lru/unload one", "english" in rr.loaded, False)
rr.unload()
check("lru/unload all", rr.loaded, [])


# --------------------------------------------------------------------- bundle vs standalone
check("bundle/english is repo root", DEFAULT_MODELS["english"], (BUNDLE_REPO, None))
check("bundle/multilingual subfolder", DEFAULT_MODELS["multilingual"], (BUNDLE_REPO, "multilingual"))
check("bundle/typed subfolder", DEFAULT_MODELS["typed-decisions"], (BUNDLE_REPO, "typed-decisions"))
check("repo_str/root", _repo_str((BUNDLE_REPO, None)), "convaiinnovations/laya")
check("repo_str/sub", _repo_str((BUNDLE_REPO, "multilingual")), "convaiinnovations/laya/multilingual")
check("repo_str/plain string", _repo_str("some/repo"), "some/repo")

r_bundle = Router()
r_alone = Router(standalone_repos=True)
check("bundle/default router uses bundle",
      r_bundle.route({"m": "मुझसे दो बार"}, Q_GENERIC)["repo"], "convaiinnovations/laya/multilingual")
check("standalone/opt-in uses own repo",
      r_alone.route({"m": "मुझसे दो बार"}, Q_GENERIC)["repo"], "convaiinnovations/laya-multilingual")
check("standalone/english unchanged",
      r_alone.route({"m": "I was charged twice"}, Q_GENERIC)["repo"], "convaiinnovations/laya")
check("standalone map complete", sorted(STANDALONE_MODELS), sorted(DEFAULT_MODELS))
# a local-path override must still work (the Space and tests rely on it)
r_local = Router(models={"english": "/tmp/en", "multilingual": "/tmp/ml"})
check("override/local path kept", r_local.route({"m": "मुझसे दो बार"}, Q_GENERIC)["repo"], "/tmp/ml")


# --------------------------------------------------------------------- preload
rr = stubbed_router(1)
rr.preload = lambda names=None, _r=rr: (
    [_load_stub(_r, n) for n in (names or list(_r.models))],
    _r)[1]
# max_loaded must grow to fit what was preloaded, or the LRU evicts it immediately
rp = stubbed_router(1)
rp.max_loaded = max(rp.max_loaded, 3)
for n in ("english", "multilingual", "typed-decisions"):
    _load_stub(rp, n)
check("preload/all three stay resident", sorted(rp.loaded),
      ["english", "multilingual", "typed-decisions"])
check("preload/max_loaded raised", rp.max_loaded >= 3, True)

rp2 = stubbed_router(1)
rp2.max_loaded = max(rp2.max_loaded, 2)
for n in ("english", "multilingual"):
    _load_stub(rp2, n)
check("preload/subset stays resident", sorted(rp2.loaded), ["english", "multilingual"])
# routing to an already-resident checkpoint must not evict anything
_load_stub(rp2, "english")
check("preload/touch does not evict", sorted(rp2.loaded), ["english", "multilingual"])


# --------------------------------------------------------------------- attach
ra = stubbed_router(1)
sentinel = _Stub("already-built")
ra.attach("english", sentinel)
check("attach/registers under the name", ra._agents["english"], sentinel)
check("attach/counts as resident", "english" in ra.loaded, True)
check("attach/raises max_loaded to hold it", ra.max_loaded >= 1, True)
# attaching then loading another must not evict the attached one
ra.max_loaded = max(ra.max_loaded, 2)
_load_stub(ra, "multilingual")
check("attach/survives a later load", sorted(ra.loaded), ["english", "multilingual"])
check("attach/still the same object", ra._agents["english"] is sentinel, True)
check("attach/accepts aliases", stubbed_router(1).attach("en", _Stub("x")) is not None, True)


# --------------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all routing tests passed")
sys.exit(1 if FAIL else 0)
