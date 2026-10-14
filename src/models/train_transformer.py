"""src/models/train_transformer.py - Day 5b: fine-tune a Hugging Face encoder
(default: google/muril-base-cased) for binary hate-speech classification.

Plain PyTorch loop (no Trainer) to avoid API differences between transformers versions.
Model selection uses the VALIDATION set only (best epoch by macro-F1). The test set is
evaluated once at the end. Do not re-run with different settings after looking at test numbers.

Colab usage (train.csv, val.csv, test.csv and this file in the working directory):
    !python train_transformer.py --smoke        # quick pipeline check, no test evaluation
    !python train_transformer.py --seed 42      # full run
"""
import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y, prob, threshold=0.5) -> dict:
    y = np.asarray(y)
    pred = (prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_hate": float(precision),
        "recall_hate": float(recall),
        "f1_hate": float(f1),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y, prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y, prob, n_boot=1000, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pred = (prob >= 0.5).astype(int)
    idx = np.arange(len(y))
    accs, f1s = [], []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        accs.append((pred[s] == y[s]).mean())
        f1s.append(f1_score(y[s], pred[s], average="macro"))
    return {
        "accuracy_95ci": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "macro_f1_95ci": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "n_boot": n_boot,
    }


class TextDataset(Dataset):
    def __init__(self, encodings: dict, labels):
        self.enc = encodings
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


def make_collate(pad_id: int):
    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)

        def pad(key, value):
            return torch.tensor([b[key] + [value] * (max_len - len(b[key])) for b in batch], dtype=torch.long)

        out = {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0)}
        if "token_type_ids" in batch[0]:
            out["token_type_ids"] = pad("token_type_ids", 0)
        out["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        return out

    return collate


def encode(tokenizer, texts, max_length: int) -> dict:
    enc = tokenizer(list(texts), truncation=True, max_length=max_length)
    return {k: enc[k] for k in enc.keys()}


@torch.no_grad()
def predict_probs(model, loader, device, use_amp: bool) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(**inputs).logits
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/muril-base-cased")
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline; skips test evaluation")
    p.add_argument("--save-model-dir", default=None, help="optional folder to save the best model and tokenizer")
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"device: {device} | torch {torch.__version__} | transformers {transformers.__version__}")
    if device.type == "cpu":
        print("WARNING: no GPU found. Runtime > Change runtime type > T4 GPU.")
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    frames = {}
    for name in ("train", "val", "test"):
        path = data_dir / f"{name}.csv"
        if not path.exists():
            sys.exit(f"Missing {path}. Upload train.csv, val.csv and test.csv first.")
        frames[name] = pd.read_csv(path, encoding="utf-8")
        if not {"clean_text", "label"} <= set(frames[name].columns):
            sys.exit(f"{path} must contain clean_text and label columns.")
    train, val, test = frames["train"], frames["val"], frames["test"]

    epochs = args.epochs
    if args.smoke:
        train, val, epochs = train.head(256), val.head(128), 1
        print("SMOKE MODE: 256 train rows, 128 val rows, 1 epoch, no test evaluation")
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collate = make_collate(tokenizer.pad_token_id)
    train_ds = TextDataset(encode(tokenizer, train["clean_text"], args.max_length), train["label"])
    val_ds = TextDataset(encode(tokenizer, val["clean_text"], args.max_length), val["label"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)

    counts = np.bincount(train["label"].to_numpy(), minlength=2)
    class_weights = torch.tensor(len(train) / (2.0 * counts), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    print(f"class weights (Normal, Hate): {class_weights.cpu().numpy().round(3)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"macro_f1": -1.0, "epoch": 0, "state": None, "val_probs": None}
    history = []
    print(f"\nsteps per epoch: {len(train_loader)} | total steps: {total_steps}")
    print(f"{'epoch':>5} {'train_loss':>11} {'val_macro_f1':>13} {'val_acc':>8} {'val_auc':>8} {'sec':>6}")

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)

        val_probs = predict_probs(model, val_loader, device, use_amp)
        m = compute_metrics(val["label"], val_probs)
        elapsed = time.perf_counter() - start
        history.append({"epoch": epoch, "train_loss": running / seen, **{f"val_{k}": m[k] for k in ("macro_f1", "accuracy", "roc_auc")}})
        print(f"{epoch:>5} {running / seen:>11.4f} {m['macro_f1']:>13.4f} {m['accuracy']:>8.4f} {m['roc_auc']:>8.4f} {elapsed:>6.0f}")
        if m["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": m["macro_f1"], "epoch": epoch, "val_probs": val_probs,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    print(f"\nBest epoch on validation: {best['epoch']} (macro-F1 {best['macro_f1']:.4f})")

    tag = f"{args.model.split('/')[-1]}_seed{args.seed}" + ("_smoke" if args.smoke else "")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"label": val["label"].to_numpy(), "prob_hate": best["val_probs"]}).to_csv(
        out_dir / f"{tag}_val_probs.csv", index=False)

    if args.smoke:
        print(f"\nSMOKE TEST OK. Pipeline ran end to end. Outputs in {out_dir}/ (smoke numbers are meaningless).")
        return

    test_ds = TextDataset(encode(tokenizer, test["clean_text"], args.max_length), test["label"])
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    test_probs = predict_probs(model, test_loader, device, use_amp)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms_per_comment = (time.perf_counter() - start) / len(test) * 1000

    val_metrics = compute_metrics(val["label"], best["val_probs"])
    test_metrics = compute_metrics(test["label"], test_probs)
    ci = bootstrap_ci(test["label"], test_probs, seed=args.seed)

    print("\nFINAL (test evaluated once)")
    print(f"{'metric':<16} {'val':>8} {'test':>8}")
    for key in ("accuracy", "precision_hate", "recall_hate", "f1_hate", "macro_f1", "roc_auc"):
        print(f"{key:<16} {val_metrics[key]:>8.4f} {test_metrics[key]:>8.4f}")
    print(f"test confusion matrix: TN={test_metrics['tn']} FP={test_metrics['fp']} FN={test_metrics['fn']} TP={test_metrics['tp']}")
    print(f"test accuracy 95% CI  : {ci['accuracy_95ci'][0]:.3f} - {ci['accuracy_95ci'][1]:.3f}")
    print(f"test macro-F1 95% CI  : {ci['macro_f1_95ci'][0]:.3f} - {ci['macro_f1_95ci'][1]:.3f}")
    print(f"inference: {ms_per_comment:.3f} ms per comment (batch of 64, {device.type})")

    pd.DataFrame({"label": test["label"].to_numpy(), "prob_hate": test_probs}).to_csv(
        out_dir / f"{tag}_test_probs.csv", index=False)
    report = {
        "model": args.model, "seed": args.seed, "max_length": args.max_length, "epochs": epochs,
        "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio, "best_epoch": best["epoch"], "selected_on": "validation macro-F1",
        "history": history, "val_metrics": val_metrics, "test_metrics": test_metrics, "test_bootstrap": ci,
        "test_ms_per_comment": ms_per_comment,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "device": device.type + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
    }
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_dir}/: {tag}_metrics.json, {tag}_val_probs.csv, {tag}_test_probs.csv")

    if args.save_model_dir:
        save_dir = Path(args.save_model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Model and tokenizer saved to {save_dir}")


if __name__ == "__main__":
    main()