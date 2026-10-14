"""
Central configuration for the OCR pipeline.

    cfg = Config.from_json("configs/default.json")
    cfg.recognizer.as_model_kwargs()
    cfg.save_json("runs/exp001/config.json")

JSON-serialisable so you can log the exact config next to every checkpoint.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RecognizerConfig:
    dims: List[int] = field(default_factory=lambda: [128, 256, 384])
    plan: List[str] = field(default_factory=lambda: ["LLL", "LGLG", "LGG"])
    heads: List[int] = field(default_factory=lambda: [4, 8, 8])
    mlp_ratio: float = 2.0
    drop_path: float = 0.1
    in_height: int = 48
    dropout: float = 0.1
    max_width: int = 2560

    def as_model_kwargs(self) -> Dict:
        return {"dims": tuple(self.dims), "plan": tuple(self.plan), "heads": tuple(self.heads),
                "mlp_ratio": self.mlp_ratio, "drop_path": self.drop_path,
                "in_height": self.in_height, "dropout": self.dropout}


@dataclass
class DetectorConfig:
    long_side: int = 1600
    pretrained_backbone: bool = True
    # DB target / loss knobs (see make_db_targets, db_loss)
    shrink_ratio: float = 0.4
    thresh_min: float = 0.3
    thresh_max: float = 0.7
    alpha: float = 1.0        # weight of dice term
    beta: float = 10.0        # weight of threshold term
    # Postprocess knobs
    bin_thresh: float = 0.3
    box_thresh: float = 0.6
    unclip_ratio: float = 1.8
    min_size: int = 3
    max_candidates: int = 1000

    def as_model_kwargs(self) -> Dict:
        return {"pretrained_backbone": self.pretrained_backbone}

    def pp_kwargs(self) -> Dict:
        return {"bin_thresh": self.bin_thresh, "box_thresh": self.box_thresh,
                "unclip_ratio": self.unclip_ratio, "min_size": self.min_size,
                "max_candidates": self.max_candidates}


@dataclass
class TableConfig:
    S: int = 640
    max_span: int = 20
    max_len: int = 600
    feat_dim: int = 128
    hidden: int = 256
    att_dim: int = 256
    emb_dim: int = 64
    pretrained_backbone: bool = True
    box_weight: float = 2.0

    def as_model_kwargs(self) -> Dict:
        return {"feat_dim": self.feat_dim, "hidden": self.hidden, "att_dim": self.att_dim,
                "emb_dim": self.emb_dim, "pretrained_backbone": self.pretrained_backbone}


@dataclass
class TrainConfig:
    # Recognizer
    train_manifest: str = "data/train.jsonl"
    val_manifest: str = "data/val.jsonl"
    charset_path: str = "data/charset.json"
    out_dir: str = "runs/rec"

    # Shared schedule knobs
    epochs: int = 60
    batch_size: int = 64
    grad_accum: int = 1        # for detector/table, set to >1 to simulate larger batches
    lr: float = 3e-4
    weight_decay: float = 5e-4
    warmup_frac: float = 0.02
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    amp: bool = True

    num_workers: int = 4
    log_every: int = 50
    val_every: int = 1
    save_every: int = 1
    seed: int = 42
    augment: bool = True
    review_thresh: float = 0.8

    # Detector/table specific
    det_train_manifest: str = "data/det_train.jsonl"
    det_val_manifest: str = "data/det_val.jsonl"
    det_out_dir: str = "runs/det"
    det_tune_every: int = 5            # run tune_unclip_ratio every N epochs on a small val subset
    det_tune_max_pages: int = 50

    table_train_manifest: str = "data/table_train.jsonl"
    table_val_manifest: str = "data/table_val.jsonl"
    table_out_dir: str = "runs/table"


@dataclass
class Config:
    recognizer: RecognizerConfig = field(default_factory=RecognizerConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    table: TableConfig = field(default_factory=TableConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict:
        return {"recognizer": dict(self.recognizer.__dict__),
                "detector": dict(self.detector.__dict__),
                "table": dict(self.table.__dict__),
                "train": dict(self.train.__dict__)}

    def save_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str) -> "Config":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        def _mk(cls_, sub):
            return cls_(**{k: v for k, v in d.get(sub, {}).items()
                           if k in cls_.__dataclass_fields__})
        return cls(recognizer=_mk(RecognizerConfig, "recognizer"),
                   detector=_mk(DetectorConfig, "detector"),
                   table=_mk(TableConfig, "table"),
                   train=_mk(TrainConfig, "train"))


if __name__ == "__main__":
    cfg = Config()
    cfg.save_json("configs/default.json")
    print("wrote configs/default.json")