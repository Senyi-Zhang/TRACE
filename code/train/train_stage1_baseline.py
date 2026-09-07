import os, json, math, random, argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

# --------------------------
# Utils
# --------------------------
def seed_all(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

LABEL_TRUE = {"1","true","entailed","entails","support","supports","supported","yes","pos","positive","SUPPORTED"}
LABEL_FALSE= {"0","false","refuted","contradiction","contradicts","no","neg","negative","REFUTED"}
def norm_label(x):
    if isinstance(x, (int,float)):
        return int(round(float(x)))
    s = str(x).strip().lower()
    if s.isdigit(): return int(s)
    if s in {w.lower() for w in LABEL_TRUE}:  return 1
    if s in {w.lower() for w in LABEL_FALSE}: return 0
    raise ValueError(f"Unrecognized label: {x}")

# --------------------------
# Dataset
# --------------------------
class PairJsonl(Dataset):
    """
    读取 JSONL，字段支持：
    - claim: str
    - context: str   (可选；若无则从 rows 拼接)
    - rows: List[str] (可选)
    - label: {0,1} 或上述别名
    """
    def __init__(self, path: str, joiner: str="  ||  ", max_samples: int=None):
        self.recs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                claim = r.get("claim") or r.get("statement")
                if claim is None:
                    raise ValueError("Each record must include 'claim' (or 'statement').")
                if "context" in r and r["context"] is not None and len(str(r["context"]).strip())>0:
                    ctx = str(r["context"])
                else:
                    rows = r.get("rows") or []
                    if not isinstance(rows, list): rows = []
                    ctx = joiner.join([str(x) for x in rows])
                y = r.get("label", None)
                if y is None: continue
                y = norm_label(y)
                self.recs.append((claim, ctx, y))
                if max_samples and len(self.recs) >= max_samples:
                    break
        assert len(self.recs)>0, f"Empty dataset from {path}"

    def __len__(self): return len(self.recs)
    def __getitem__(self, i): return self.recs[i]

def build_collate(tokenizer, max_len: int):
    def collate(batch):
        claims, ctxs, ys = zip(*batch)
        enc = tokenizer(
            list(claims),
            list(ctxs),
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt"
        )
        y = torch.tensor(ys, dtype=torch.long)
        return enc, y
    return collate

def collate_pairjsonl(batch, tokenizer, max_len: int):
    claims, ctxs, ys = zip(*batch)
    enc = tokenizer(
        list(claims),
        list(ctxs),
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt",
    )
    y = torch.tensor(ys, dtype=torch.long)
    return enc, y

# --------------------------
# Model: DeBERTa + masked-mean pool + linear head
# --------------------------
class DebertaBaseline(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", dropout=0.1, num_labels=2):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h, num_labels)

    def masked_mean(self, hs, mask):
        # hs: [B,L,H], mask: [B,L]
        mask = mask.unsqueeze(-1).float()
        s = (hs * mask).sum(1)
        d = mask.sum(1).clamp(min=1e-6)
        return s / d

    def forward(self, enc_inputs):
        # enc_inputs: dict from tokenizer
        out = self.enc(**enc_inputs).last_hidden_state  # [B,L,H]
        pooled = self.masked_mean(out, enc_inputs["attention_mask"])  # [B,H]
        logits = self.head(self.drop(pooled))  # [B,2]
        return logits

# --------------------------
# Freeze / Unfreeze helpers
# --------------------------
def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_last_n_layers(enc: AutoModel, n: int):
    """
    只解冻 encoder 的最后 n 层（DeBERTa 结构：enc.encoder.layer[*]）
    """
    if n <= 0: return
    try:
        layers = enc.encoder.layer
    except AttributeError:
        # 不同架构的兜底
        layers = getattr(enc, "deberta", None)
        if layers is None or not hasattr(layers, "encoder"):
            raise RuntimeError("Unexpected model structure; please adapt unfreeze_last_n_layers().")
        layers = layers.encoder.layer

    L = len(layers)
    for i, block in enumerate(layers):
        if i >= L - n:
            for p in block.parameters():
                p.requires_grad = True

    # 通常也解冻最终 LayerNorm
    if hasattr(enc, "encoder") and hasattr(enc.encoder, "LayerNorm"):
        for p in enc.encoder.LayerNorm.parameters(): p.requires_grad = True

# --------------------------
# Train / Eval
# --------------------------
@torch.no_grad()
def evaluate(model, tokenizer, dataloader, device):
    model.eval()
    total = correct = 0
    for enc, y in dataloader:
        enc = {k:v.to(device) for k,v in enc.items()}
        y = y.to(device)
        logits = model(enc)
        pred = logits.argmax(-1)
        correct += (pred==y).sum().item()
        total += y.numel()
    return correct / max(1,total)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="D:\my_datasets\\tabfact\\train.jsonl")  # JSONL 路径
    ap.add_argument("--dev",   default="D:\my_datasets\\tabfact\\validation.jsonl")
    ap.add_argument("--test",  default="D:\my_datasets\\tabfact\\test.jsonl")
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)            # 小批+梯度累积
    ap.add_argument("--grad_accum", type=int, default=8)            # 8*8=64 等效 batch
    ap.add_argument("--epochs", type=int, default=3)                # 暖身很短
    ap.add_argument("--lr_head", type=float, default=2e-5)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--last_n_unfrozen", type=int, default=6)       # 解冻最后 6 层
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default="stage1_deberta_baseline.pt")
    ap.add_argument("--max_train_samples", type=int, default=None)  # 可设小子集快跑
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    train_set = PairJsonl(args.train, max_samples=args.max_train_samples)
    dev_set   = PairJsonl(args.dev)
    test_set  = PairJsonl(args.test)

    collate = partial(collate_pairjsonl, tokenizer=tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, collate_fn=collate)
    dev_loader = DataLoader(dev_set, batch_size=max(16, args.batch_size), shuffle=False,
                            num_workers=2, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=max(16, args.batch_size), shuffle=False,
                             num_workers=2, collate_fn=collate)

    model = DebertaBaseline(args.model_name, dropout=args.dropout).to(device)

    # 先冻结全部，再按需解冻最后 n 层
    freeze_all(model)
    # 解冻 head & 最后 n 层 backbone
    for p in model.head.parameters(): p.requires_grad = True
    unfreeze_last_n_layers(model.enc, args.last_n_unfrozen)

    # 参数组：backbone 与 head 不同 LR
    param_groups = [
        {"params": [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("head.")],
         "lr": args.lr_backbone},
        {"params": list(model.head.parameters()), "lr": args.lr_head},
    ]
    optim = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # scheduler
    steps_per_epoch = math.ceil(len(train_loader) / max(1,args.grad_accum))
    total_steps = args.epochs * steps_per_epoch
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type=="cuda"))

    best_dev, best_state = 0.0, None
    model.train()
    best_dev, best_state = 0.0, None
    global_step = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = len(train_loader)

        pbar = tqdm(
            enumerate(train_loader, start=1),
            total=num_batches,
            desc=f"Epoch {ep}/{args.epochs}",
            leave=False
        )

        optim.zero_grad(set_to_none=True)
        for step, (enc, y) in pbar:
            enc = {k: v.to(device) for k, v in enc.items()}
            y = y.to(device)

            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    logits = model(enc)
                    loss = F.cross_entropy(logits, y)
                # 反向 + AMP
                scaler.scale(loss).backward()
            else:
                logits = model(enc)
                loss = F.cross_entropy(logits, y)
                loss.backward()

            running_loss += loss.item()
            global_step += 1

            # 累积到一定步数再更新参数
            if step % args.grad_accum == 0:
                # AMP 情况下先 unscale 再做梯度裁剪（关键！）
                if scaler.is_enabled():
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()

                sched.step()
                optim.zero_grad(set_to_none=True)

            # 日志：每 log_every 步更新一次 tqdm 尾注
            if (step % args.log_every == 0) or (step == num_batches):
                avg_loss = running_loss / step
                # 两个 param_group：0=backbone, 1=head（你的 param_groups 定义就是这个顺序）
                lr_b = optim.param_groups[0]["lr"]
                lr_h = optim.param_groups[1]["lr"] if len(optim.param_groups) > 1 else lr_b
                postfix = {
                    "loss": f"{avg_loss:.4f}",
                    "lr_b": f"{lr_b:.2e}",
                    "lr_h": f"{lr_h:.2e}",
                }
                # 显存（可选显示）
                if device.type == "cuda":
                    mem = torch.cuda.memory_allocated() / (1024 ** 3)
                    postfix["GPU_mem_GB"] = f"{mem:.2f}"
                pbar.set_postfix(postfix)

        # epoch end eval
        train_avg_loss = running_loss / max(1, num_batches)
        dev_acc = evaluate(model, tokenizer, dev_loader, device)
        print(f"Epoch {ep:02d} | train loss {train_avg_loss:.4f} | dev acc {dev_acc:.4f}")

        if dev_acc > best_dev + 1e-4:
            best_dev = dev_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(
                {"model_name": args.model_name, "state_dict": best_state, "args": vars(args)},
                args.ckpt,
            )
            print("  (saved best)")

    # 测试
    if best_state is None:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(best_state)

    test_acc = evaluate(model, tokenizer, test_loader, device)
    print(f"Best dev acc {best_dev:.4f} | test acc {test_acc:.4f}")

if __name__ == "__main__":
    main()
