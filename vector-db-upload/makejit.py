# export_embedder_swin_t222.py
# Export a TorchScript embedder for torchvision.models.swin_b that is stable across versions
# and verified to load with the current Torch/Torchvision in this environment.

import os, sys, json, time, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_b

# ---------- Choose your target runtime versions ----------
# (Set these to the *exact* versions you want to guarantee.)
TARGET_TORCH = "2.2.2+cpu"
TARGET_TV    = "0.17.2+cpu"
STRICT_VERSION = True  # set False to only warn instead of abort
# --------------------------------------------------------

# ---------- Your export settings ----------
CKPT_PATH  = r".\models\siamese_model_val_arak.pth"  # state_dict or Lightning ckpt
OUT_PATH   = r".\embedder_jit_t222_cpu.pt"
EMB_DIM    = 128
INPUT_SIZE = 224
# -----------------------------------------

def _print_env():
    try:
        import torchvision as tv
        print(f"[env] torch={torch.__version__}  torchvision={tv.__version__}")
        return torch.__version__, tv.__version__
    except Exception:
        print(f"[env] torch={torch.__version__}  torchvision=(unknown)")
        return torch.__version__, None

def _assert_versions(cur_torch, cur_tv):
    problems = []
    if cur_torch != TARGET_TORCH:
        problems.append(f"torch wanted {TARGET_TORCH}, got {cur_torch}")
    if cur_tv is not None and cur_tv != TARGET_TV:
        problems.append(f"torchvision wanted {TARGET_TV}, got {cur_tv}")
    if problems:
        msg = "[version] " + "; ".join(problems)
        if STRICT_VERSION:
            raise RuntimeError(msg)
        else:
            print(msg)

# Stable embedder: don’t rely on .features internals; remove the classifier head instead.
class Embedder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        m = swin_b(weights=None)
        in_dim = m.head.in_features          # (usually 1024 for swin_b)
        m.head = nn.Identity()               # pooled feature vector from forward()
        self.backbone = m
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)                 # [B, in_dim]
        z = self.fc(f)                       # [B, EMB_DIM]
        return F.normalize(z, p=2, dim=1)

def _load_checkpoint_strictish(model: nn.Module, ckpt_path: str):
    print(f"[load] ckpt: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # Robust key remap (handles common training wrappers)
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):            nk = nk[7:]
        if nk.startswith("siamese."):           nk = nk[8:]
        if nk.startswith("feature_extractor."): nk = "backbone." + nk
        new_sd[nk] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print("[load] missing:", missing)
    print("[load] unexpected:", unexpected)

def main():
    cur_torch, cur_tv = _print_env()
    _assert_versions(cur_torch, cur_tv)

    model = Embedder(emb_dim=EMB_DIM).eval()

    if CKPT_PATH and os.path.isfile(CKPT_PATH):
        _load_checkpoint_strictish(model, CKPT_PATH)
    else:
        print("[warn] no checkpoint found — exporting randomly initialized embedder")

    ex = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    # Prefer trace; fall back to script if needed
    try:
        ts = torch.jit.trace(model, ex, strict=True)
        _ = ts(ex)  # sanity
        print("[trace] traced OK")
    except Exception as e:
        print("[trace] failed:", e)
        print("[script] attempting torch.jit.script ...")
        ts = torch.jit.script(model)

    # Freeze reduces IR variability across loads
    try:
        ts = torch.jit.freeze(ts)
        print("[freeze] applied")
    except Exception as e:
        print("[freeze] skipped:", e)

    # Save with metadata so the runtime can assert compatibility
    meta = {
        "export_time": int(time.time()),
        "torch": cur_torch,
        "torchvision": cur_tv,
        "input_size": [3, INPUT_SIZE, INPUT_SIZE],
        "embedding_dim": EMB_DIM,
        "arch": "swin_b + linear->512->norm->relu->dropout->linear->128 + l2norm",
    }
    extra = {"meta.json": json.dumps(meta)}
    torch.jit.save(ts, OUT_PATH, _extra_files=extra)
    print(f"[save] wrote {OUT_PATH}  (size={os.path.getsize(OUT_PATH)} bytes)")
    print("[meta]", json.dumps(meta, indent=2))

    # Verify load with THIS Torch (must succeed before you ship)
    extra2 = {"meta.json": ""}
    ts2 = torch.jit.load(OUT_PATH, map_location="cpu", _extra_files=extra2)
    y = ts2(ex)
    print("[verify] jit.load OK; output shape:", tuple(y.shape))
    print("[verify] embedded meta.json:", extra2["meta.json"])
    
    # Clean up to prevent memory issues
    del model, ts, ts2, ex, y
    gc.collect()

if __name__ == "__main__":
    main()
