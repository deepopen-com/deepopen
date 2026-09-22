"""Laya-derived encoder classifier with a trainable cosine-prototype residual.

This is a fixed-label classifier, not the native Laya variable-option head.
Prototype classifiers and semantic initialization are established techniques;
this implementation does not claim a novel algorithm.
"""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from transformers import AutoModelForSequenceClassification


class PrototypeClassifier(nn.Module):
    def __init__(self, backbone, variant="semantic", anchors=None):
        super().__init__()
        self.backbone = backbone
        self.variant = variant
        d, c = backbone.config.hidden_size, backbone.config.num_labels
        if anchors is None:
            anchors = torch.zeros(c, d)
        self.register_buffer("anchors", anchors.float().clone())
        self.prototypes = nn.Parameter(anchors.float().clone())
        self.log_scale = nn.Parameter(torch.tensor(20.).log())
        self.gate = nn.Parameter(torch.tensor(-1.))

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        h = self.backbone.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        w = attention_mask.unsqueeze(-1)
        mean = (h * w).sum(1) / w.sum(1).clamp_min(1)
        pooled = h[:, 0] if self.backbone.config.classifier_pooling == "cls" else mean
        ce_logits = self.backbone.classifier(self.backbone.drop(self.backbone.head(pooled)))
        if self.variant == "control":
            logits, proto = ce_logits, None
        else:
            proto = self.log_scale.clamp(0, 4.6).exp() * (F.normalize(mean.float(), dim=-1) @ F.normalize(self.prototypes, dim=-1).T)
            logits = ce_logits.float() + self.gate.sigmoid() * proto
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float(), labels)
            if proto is not None:
                loss = loss + .2 * F.cross_entropy(proto, labels)
                loss = loss + .01 * (1 - F.cosine_similarity(self.prototypes, self.anchors)).mean()
        return {"logits": logits, "loss": loss, "mean": mean, "base_logits": ce_logits}

    def save_pretrained(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path / "backbone")
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items() if not k.startswith("backbone.")}, str(path / "prototype.safetensors"))
        (path / "prototype_config.json").write_text(json.dumps({"variant": self.variant, "version": 1, "aux_ce": .2, "anchor_regularization": .01}), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path):
        path = Path(path)
        cfg = json.loads((path / "prototype_config.json").read_text())
        model = cls(AutoModelForSequenceClassification.from_pretrained(path / "backbone", attn_implementation="sdpa"), cfg["variant"])
        extra = load_file(str(path / "prototype.safetensors"))
        expected = {k for k in model.state_dict() if not k.startswith("backbone.")}
        assert set(extra) == expected
        model.load_state_dict({**model.state_dict(), **extra}, strict=True)
        return model
