#!/usr/bin/env python3
import os
import time
import torch
from ultralytics import YOLO

# ---------- USER SETTINGS ----------
DATA_YAML = "/home/shahzaib/Desktop/FYP_Project/yolo_training/configs/citypersons.yaml"
PRETRAINED = "/home/shahzaib/Desktop/FYP_Project/Dataset_Annotation/yolo11x.pt"  # pretrained weights
PROJECT = "/home/shahzaib/Desktop/FYP_Project/YOLO11/person_finetune"
NAME = "yolo11x_person"
DEVICE = "0" if torch.cuda.is_available() else "cpu"


# Phase settings
PHASE1 = { "epochs": 10,  "batch": 16, "imgsz": 640, "lr0": 1e-3, "freeze": True }   # head-only
PHASE2 = { "epochs": 60,  "batch": 12, "imgsz": 640, "lr0": 5e-4, "freeze": False }  # full fine-tune

# Misc
SAVE_PERIOD = 5  # ultralytics: save every n epochs
EXIST_OK = True
# -----------------------------------

def train_phase(model, data, epochs, batch, imgsz, lr0, freeze, project, name, resume=False):
    print(f"\n--- TRAIN PHASE: epochs={epochs}, batch={batch}, imgsz={imgsz}, lr0={lr0}, freeze={freeze} ---")
    overrides = {
        "data": data,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "lr0": lr0,
        "project": project,
        "name": name,
        "exist_ok": EXIST_OK,
        "save_period": SAVE_PERIOD,
        "device": DEVICE,
        # Use optimizer auto or set explicit as needed:
        # "optimizer": "AdamW",
    }

    # If freeze requested, freeze the backbone via ultralytics 'freeze' param (number of layers or True may work)
    # The ultralytics API accepts 'freeze' argument as layer indices or None. We'll try True / False fallback.
    if freeze:
        overrides["freeze"] = 10  # freeze first ~10 layers (conservative). You can tune this number.
    else:
        overrides["freeze"] = None


    # Don't resume from a finished checkpoint unless you explicitly want to
    overrides["resume"] = resume

    results = model.train(**overrides)
    return results

def main():
    os.makedirs(PROJECT, exist_ok=True)
    print("Ultralytics:", __import__("ultralytics").__version__)
    print("Device:", DEVICE)
    # load pretrained model (this will not start training)
    model = YOLO(PRETRAINED)

    # PHASE 1: head-only warmup
    r1 = train_phase(
        model,
        DATA_YAML,
        epochs=PHASE1["epochs"],
        batch=PHASE1["batch"],
        imgsz=PHASE1["imgsz"],
        lr0=PHASE1["lr0"],
        freeze=PHASE1["freeze"],
        project=PROJECT,
        name=NAME + "_phase1",
        resume=False
    )

    # Optional: inspect r1.metrics or r1.best_map50
    print("Phase 1 done. Best mAP50 (phase1):", getattr(r1, "best_map50", "N/A"))

    # PHASE 2: unfreeze and fine-tune
    r2 = train_phase(
        model,
        DATA_YAML,
        epochs=PHASE2["epochs"],
        batch=PHASE2["batch"],
        imgsz=PHASE2["imgsz"],
        lr0=PHASE2["lr0"],
        freeze=PHASE2["freeze"],
        project=PROJECT,
        name=NAME + "_phase2",
        resume=False
    )

    print("Phase 2 done. Best mAP50 (phase2):", getattr(r2, "best_map50", "N/A"))
    print("Training finished. Final runs saved to:", os.path.join(PROJECT, NAME + "_phase2"))

if __name__ == "__main__":
    main()











