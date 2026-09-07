import os, json, math, random, argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# ==== 你自己的 NodeEncoder ====
from NodeEncoder import NodeEncoder  # 确保与下方用到的属性名一致：proj_self/proj_child/gate/norm/enc/tok

# ---------------- Utils ----------------
def seed_all(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

LABEL_TRUE  = {"1","true","entailed","entails","support","supports","supported","yes","pos","positive","SUPPORTED"}
LABEL_FALSE = {"0","false","refuted","contradiction","contradicts","no","neg","negative","REFUTED"}
def norm_label(x):
    if isinstance(x,(int,float)): return int(round(float(x)))
    s=str(x).strip().lower()
    if s.isdigit(): return int(s)
    if s in {w.lower() for w in LABEL_TRUE}:  return 1
    if s in {w.lower() for w in LABEL_FALSE}: return 0
    raise ValueError(f"Unrecognized label: {x}")

# ---------------- Dataset ----------------
class RowsJsonl(Dataset):
    def __init__(self, path: str, K: int, max_samples: int=None):
        self.K = K
        self.recs = []
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                claim = r.get("claim") or r.get("statement")
                rows  = r.get("rows") or []
                label = r.get("label")
                if claim is None or label is None: continue
                # pad / truncate rows to K
                rows = [str(x) for x in rows][:K]
                if len(rows) < K: rows += [""]*(K-len(rows))
                self.recs.append((claim, rows, norm_label(label)))
                if max_samples and len(self.recs) >= max_samples: break
        assert len(self.recs)>0, f"Empty dataset from {path}"

    def __len__(self): return len(self.recs)
    def __getitem__(self, i): return self.recs[i]

def collate(batch):
    claims, rows_list, ys = zip(*batch)  # len=B
    return list(claims), list(rows_list), torch.tensor(ys, dtype=torch.long)

# ---------------- Stage-2 Model ----------------
class Stage2Joint(nn.Module):
    """
    冻结 DeBERTa，仅训练 NodeEncoder 的 proj_self/proj_child/gate/norm + 一个线性分类头。
    行级注意力：用 self_emb 作为 query（点积软最大化）对 K 行向量聚合。
    """
    def __init__(self, node: NodeEncoder, num_labels=2, max_len=256):
        super().__init__()
        self.node = node
        self.num_labels = num_labels
        self.max_len = max_len
        h = self.node.enc.config.hidden_size
        self.head = nn.Linear(h, num_labels)

        # 冻结 encoder
        for p in self.node.enc.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _encode_pairs_batch(self, claims: List[str], rows: List[List[str]]):
        """
        返回:
          self_emb: [B,H]  (用 (claim,"") )
          child:    [B,K,H]  (用 (claim,row_j) )
        """
        device = next(self.parameters()).device
        B = len(claims); K = len(rows[0]) if rows else 0
        # self_emb
        tok_self = self.node.tok(claims, [""]*B, truncation=True, max_length=self.max_len,
                                 padding=True, return_tensors="pt").to(device)
        out_self = self.node.enc(**tok_self).last_hidden_state
        mask_s   = tok_self["attention_mask"].unsqueeze(-1).float()
        self_emb = (out_self * mask_s).sum(1) / mask_s.sum(1).clamp_min(1e-6)
        self_emb = F.normalize(self_emb, p=2, dim=-1)  # [B,H]

        if K == 0:
            child = torch.zeros(B, 0, self_emb.size(-1), device=device, dtype=self_emb.dtype)
            return self_emb, child

        # 展平 (claim, row_ij)
        flat_q = []
        flat_r = []
        for i in range(B):
            for j in range(K):
                flat_q.append(claims[i])
                flat_r.append(rows[i][j])
        tok = self.node.tok(flat_q, flat_r, truncation=True, max_length=self.max_len,
                            padding=True, return_tensors="pt").to(device)
        out  = self.node.enc(**tok).last_hidden_state
        mask = tok["attention_mask"].unsqueeze(-1).float()
        vec  = (out * mask).sum(1) / mask.sum(1).clamp_min(1e-6)  # [B*K,H]
        vec  = F.normalize(vec, p=2, dim=-1).view(B, K, -1)       # [B,K,H]
        return self_emb, vec

    def forward(self, claims: List[str], rows: List[List[str]]):
        """
        returns logits [B,2]
        """
        device = next(self.parameters()).device
        B = len(claims)
        self_emb, child = self._encode_pairs_batch(claims, rows)  # no grad through encoder
        H = self_emb.size(-1)
        K = child.size(1)

        # ---- 注意力聚合（有梯度）----
        if K > 0:
            q = F.normalize(self_emb, p=2, dim=-1).unsqueeze(1)   # [B,1,H]
            k = F.normalize(child,   p=2, dim=-1)                 # [B,K,H]
            att = torch.matmul(q, k.transpose(1,2)) / math.sqrt(H)  # [B,1,K]
            att = torch.softmax(att, dim=-1)
            ctx = torch.matmul(att, child).squeeze(1)             # [B,H]
        else:
            ctx = torch.zeros_like(self_emb)

        # ---- 门控融合（训练 proj_* / gate / norm）----
        h_self  = self.node.proj_self(self_emb)   # [B,H]
        h_child = self.node.proj_child(ctx)       # [B,H]
        g = torch.sigmoid(self.node.gate(torch.cat([h_self, h_child], dim=-1)))  # [B,1]
        fused = g * h_child + (1 - g) * h_self
        out = self.node.norm(self_emb + self.node.dropout(fused))  # [B,H]

        logits = self.head(out)  # [B,2]
        return logits

# ---------------- Train / Eval ----------------
@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    total = correct = 0
    for batch in dl:  # dl 可以是 DataLoader 或 tqdm(DataLoader)
        claims, rows, y = batch
        y = y.to(device)
        logits = model(claims, rows)
        pred = logits.argmax(-1)
        correct += (pred==y).sum().item()
        total   += y.numel()
    return correct / max(1,total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="D:\my_datasets\\tabfact\\train.jsonl")
    ap.add_argument("--dev",   default="D:\my_datasets\\tabfact\\validation.jsonl")
    ap.add_argument("--test",  default="D:\my_datasets\\tabfact\\test.jsonl")
    ap.add_argument("--stage1_ckpt", default="stage1_deberta_baseline.pt", help="stage1_deberta_baseline.pt")
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default="stage2_joint.pt")
    ap.add_argument("--val_ratio", type=float, default=0.05)  # 若未提供 dev，则从 train 切
    ap.add_argument("--max_train_samples", type=int, default=None)
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- NodeEncoder: 加载阶段一底座权重 ----
    node = NodeEncoder(model_name=args.model_name, device=device)
    # 灌入 encoder 权重（只取 'enc.' 前缀）
    sd = torch.load(args.stage1_ckpt, map_location="cpu")["state_dict"]
    enc_sd = {k[len("enc."):]: v for k,v in sd.items() if k.startswith("enc.")}
    node.enc.load_state_dict(enc_sd, strict=False)  # 忽略 head.*
    node.enc.eval()  # 冻结底座
    for p in node.enc.parameters(): p.requires_grad = False

    model = Stage2Joint(node, num_labels=2, max_len=args.max_len).to(device)

    # ---- 数据 ----
    train_full = RowsJsonl(args.train, K=args.K, max_samples=args.max_train_samples)
    # dev split
    if args.dev:
        dev_set = RowsJsonl(args.dev, K=args.K)
        train_set = train_full
    else:
        recs = train_full.recs[:]
        random.shuffle(recs)
        n_val = max(1, int(len(recs) * args.val_ratio))
        dev_recs = recs[:n_val]; train_recs = recs[n_val:]
        class RowsList(Dataset):
            def __init__(self, recs): self.recs = recs
            def __len__(self): return len(self.recs)
            def __getitem__(self,i): return self.recs[i]
        train_set = RowsList(train_recs)
        dev_set   = RowsList(dev_recs)

    test_set = RowsJsonl(args.test, K=args.K) if args.test else None

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate
    )
    dev_loader = DataLoader(
        dev_set, batch_size=max(16, args.batch_size), shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate
    )
    test_loader = (DataLoader(
        test_set, batch_size=max(16, args.batch_size), shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate
    ) if test_set else None)

    # ---- 优化器（只训门控/投影/LN + head）----
    params = []
    params += list(model.node.proj_self.parameters())
    params += list(model.node.proj_child.parameters())
    params += list(model.node.gate.parameters())
    params += list(model.node.norm.parameters())
    params += list(model.head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / max(1,args.grad_accum))
    total_steps = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=args.warmup_ratio
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(args.fp16 and device.type=="cuda"))  # NEW
# 训练时：


    # ---- 训练 ----
    print(f"[info] train size={len(train_loader.dataset)}, "
          f"dev size={len(dev_loader.dataset)}, "
          f"batch={args.batch_size}, grad_accum={args.grad_accum}, K={args.K}")

    best_dev, best_state = 0.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        bar = tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}", dynamic_ncols=True)
        running_loss = 0.0
        updates = 0

        for step, (claims, rows, y) in enumerate(bar, start=1):
            y = y.to(device)
            # autocast + loss
            with torch.amp.autocast("cuda", enabled=(args.fp16 and device.type == "cuda")):
                logits = model(claims, rows)
                loss = F.cross_entropy(logits, y)

            running_loss += loss.item()

            if args.fp16 and device.type == "cuda":
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
            else:
                loss.backward()

            did_update = False
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                if args.fp16 and device.type == "cuda":
                    scaler.step(opt)
                    scaler.update()
                    # 如果本 step 因为 NaN/Inf 被 GradScaler 跳过，就不要 scheduler.step()
                    did_update = (scaler.get_scale() < scale_before) is False
                else:
                    opt.step()
                    did_update = True

                if did_update:
                    sched.step()
                    updates += 1
                opt.zero_grad(set_to_none=True)

            # tqdm 显示
            lr_now = opt.param_groups[0]["lr"]
            bar.set_postfix(loss=f"{running_loss / step:.4f}", lr=f"{lr_now:.2e}", updates=updates)

        # 验证
        dev_bar = tqdm(dev_loader, desc="Validate", leave=False, dynamic_ncols=True)
        dev_acc = evaluate(model, dev_bar, device)  # 下面我们也改了 evaluate 支持 tqdm
        print(f"Epoch {ep} | dev acc {dev_acc:.4f}")

        if dev_acc > best_dev + 1e-4:
            best_dev = dev_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "args": vars(args), "model_name": args.model_name}, args.ckpt)
            print("  (saved best)")

    # 测试（可选）
    if best_state is None:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(best_state)

    if test_loader is not None:
        test_acc = evaluate(model, test_loader, device)
        print(f"Best dev acc {best_dev:.4f} | test acc {test_acc:.4f}")
    else:
        print(f"Best dev acc {best_dev:.4f} | (no test set provided)")
if __name__ == "__main__":
    main()
