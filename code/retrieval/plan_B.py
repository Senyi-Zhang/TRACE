# ============================================
# Cross-Encoder Reranker (Plan B) - Full Script
# Author: ChatGPT for Senyi
# ============================================
import os
from path_utils import *
from proxy import *
# ---------- 全局配置（按需修改） ----------
# 在 plan_B.py 顶部，替换原来的 OUTPUT_DIR / SBERT_CACHE_DIR 定义

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models", "cross_encoder_num_key")
SBERT_CACHE_DIR = os.path.join(OUTPUT_DIR, "sbert_cache")


PATH_TABLE_JSON = OTT_TABLE         # 大表格库，形如 {table_id: {"title":..., "headers":[...], "rows":[[...]...]}}
PATH_CLAIM_JSONL = OUR_TRAIN      # 每行: {"id":..., "claim":..., "table_uid1":..., "table_uid_2":...}

MINED_JSONL = os.path.join(os.path.dirname(__file__), "mined_pairs.jsonl")
      # 挖好正负样本后的列表（可复用）

# 召回与采样
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOPK = 200          # SBERT 粗排 top-k
HARD_M = 8          # 每个 query 取的难负例个数
RND_M = 2           # 随机负例个数（稳健性）

# 交叉编码器
BASE_MODEL_NAME = "microsoft/deberta-v3-base"     # 可改为 "roberta-base" / "longformer-base-4096" 等
MAX_LEN = 512
BATCH_SIZE = 8
LR = 2e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05
SEED = 42

# 数据切分
DEV_RATIO = 0.1     # 10% 做验证
EVAL_EVERY_EPOCH = True

# -------------------------------------------

import os
import json
import math
import random
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Any

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)

from sentence_transformers import SentenceTransformer
try:
    import faiss
except Exception as e:
    faiss = None
    print("[WARN] faiss-cpu 未安装，将降级为 numpy 暴力近邻（慢）。", e)

from sklearn.metrics import ndcg_score


# =============== 工具：加载表库与数据 ===============

def load_all_tables(path: str) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        table_db = json.load(f)
    table_ids = list(table_db.keys())
    return table_ids, table_db


def iter_claims(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# =============== SBERT 召回构建 ===============

def serialize_table_for_retrieval(tbl: Dict[str, Any], max_rows: int = 32) -> str:
    title = str(tbl.get("title", ""))[:512]
    headers = " | ".join(map(str, tbl.get("headers", [])))[:512]
    rows = tbl.get("rows", [])[:max_rows]
    row_text = " || ".join([" | ".join(map(str, r)) for r in rows])[:2000]
    return f"Title: {title} || Header: {headers} || Rows: {row_text}"


class SBERTIndex:
    def __init__(self, sbert_name: str, table_ids: List[str], table_db: Dict[str, Any]):
        print("[SBERT] 编码表格向量 ...")
        self.model_name = sbert_name                     # ← 新增：保存模型名
        self.model = SentenceTransformer(sbert_name)
        self.table_ids = table_ids
        self.table_texts = [serialize_table_for_retrieval(table_db[tid]) for tid in table_ids]
        embs = self.model.encode(
            self.table_texts, batch_size=256, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True
        )
        self.dim = embs.shape[1]
        self.embs = embs
        self.id_map = {i: tid for i, tid in enumerate(table_ids)}
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(embs)
        else:
            self.index = None

        # 指纹（可选）
        self.fingerprint = _texts_fingerprint(self.table_texts)

    def retrieve(self, query: str, topk: int) -> List[str]:
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        if self.index is not None:
            D, I = self.index.search(q, topk)
            return [self.id_map[i] for i in I[0]]
        else:
            # numpy fallback
            sims = (self.embs @ q[0])
            top_idx = np.argsort(-sims)[:topk]
            return [self.id_map[i] for i in top_idx]


# =============== 负例挖掘 ===============

def mine_pairs_to_jsonl(
    claim_jsonl: str,
    table_ids: List[str],
    sbert_index: SBERTIndex,
    out_jsonl: str,
    topk: int = TOPK,
    hard_m: int = HARD_M,
    rnd_m: int = RND_M,
):
    rng = random.Random(SEED)
    print("[MINE] 开始挖掘正/负例 ...")
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for ex in tqdm(iter_claims(claim_jsonl), desc="mining"):
            qid = ex.get("id")
            query = ex.get("claim", "")
            pos_set = set()
            for k in ["table_uid1", "table_uid_2", "table_uid2", "gold_table1", "gold_table2"]:
                if k in ex and ex[k]:
                    pos_set.add(str(ex[k]))
            # 召回
            cands = sbert_index.retrieve(query, topk=topk)
            hard_negs = [t for t in cands if t not in pos_set][:hard_m]
            # 随机负例
            rnd_pool = [tid for tid in table_ids if tid not in pos_set]
            rnd_negs = rng.sample(rnd_pool, k=min(rnd_m, len(rnd_pool))) if rnd_pool else []

            # 正例写出
            for p in pos_set:
                rec = {"qid": qid, "query": query, "table_id": p, "label": 1}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # 负例写出（标注 hard 与 random 源，可选地将 random 降权）
            for n in hard_negs:
                rec = {"qid": qid, "query": query, "table_id": n, "label": 0, "neg_src": "hard"}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for n in rnd_negs:
                rec = {"qid": qid, "query": query, "table_id": n, "label": 0, "neg_src": "rand"}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[MINE] 完成，写入 {out_jsonl}")


# =============== 高亮/标记：关键词 & 数字 ===============

STOPWORDS = set([
    "the","a","an","of","for","and","to","in","on","at","by","with","is","are","was","were",
    "from","as","that","this","it","its","be","or","if","then","than","which"
])

UNIT_PAT = r"(?:%|km|m|cm|mm|h|min|s|hr|hrs|hour|hours|sec|secs|second|seconds|kg|g|mg|\$)"
NUM_PAT  = r"\d+(?:[\.,]\d+)?(?:\s*"+UNIT_PAT+r")?"
re_num = re.compile(NUM_PAT, flags=re.IGNORECASE)
re_tok = re.compile(r"[a-zA-Z0-9%]+\b")

SPECIAL_TOKENS = ["[KEY]","[/KEY]","[NUM]","[/NUM]","[HDR]","[/HDR]","[ROW]","[/ROW]"]


def extract_numbers(text: str) -> List[Tuple[str, Tuple[int,int]]]:
    return [(m.group(), m.span()) for m in re_num.finditer(text or "")]


def query_tokens(query: str) -> List[str]:
    toks = [w.lower() for w in re_tok.findall(query or "")]
    return [w for w in toks if w not in STOPWORDS and len(w) > 1]


def simple_kw_spans(q_toks: List[str], text: str) -> List[Tuple[int,int]]:
    spans = []
    low = (text or "").lower()
    for w in q_toks:
        start = 0
        while True:
            pos = low.find(w, start)
            if pos == -1: break
            spans.append((pos, pos+len(w)))
            start = pos + len(w)
    return spans


def mark_spans(text: str, spans: List[Tuple[int,int]], open_tok: str, close_tok: str) -> str:
    if not text:
        return ""
    if not spans:
        return text
    spans = sorted(spans)
    out, cur = [], 0
    for s, e in spans:
        if s < cur:  # 重叠跳过
            continue
        s = max(0, min(len(text), s))
        e = max(0, min(len(text), e))
        out.append(text[cur:s]); out.append(open_tok); out.append(text[s:e]); out.append(close_tok); cur = e
    out.append(text[cur:])
    return "".join(out)


# =============== 将表格转成“带标记”的 cross-encoder 文本 ===============

def serialize_table_marked(query: str, tbl: Dict[str, Any], max_rows: int = 50) -> str:
    q_toks = query_tokens(query)
    q_nums = {n for n, _ in extract_numbers(query)}

    title = str(tbl.get("title", "") or "")
    headers = list(map(str, tbl.get("headers", []) or []))
    rows = tbl.get("rows", []) or []

    # 标注标题命中的关键词
    title_marked = mark_spans(title, simple_kw_spans(q_toks, title), "[KEY]", "[/KEY]")

    # 表头：关键词命中 → [HDR]…[/HDR]
    hdr_marked = []
    for h in headers:
        h_m = mark_spans(h, simple_kw_spans(q_toks, h), "[HDR]", "[/HDR]")
        hdr_marked.append(h_m)

    # 行：数字同形命中 → [NUM]…[/NUM]；若该行命中数字或关键词，则整体包 [ROW]…[/ROW]
    row_lines = []
    for r in rows[:max_rows]:
        cells, hit = [], False
        for c in map(str, r):
            cm = c
            # 数字标注
            for n, _ in extract_numbers(c):
                if n in q_nums:
                    cm = cm.replace(n, f"[NUM]{n}[/NUM]")
                    hit = True
            # 关键词标注（若没有数字命中也可继续做）
            spans = simple_kw_spans(q_toks, c)
            if spans:
                cm2 = mark_spans(cm, spans, "[KEY]", "[/KEY]")
                hit = True if cm2 != cm else hit
                cm = cm2
            cells.append(cm)
        line = " | ".join(cells)
        if hit:
            line = f"[ROW]{line}[/ROW]"
        row_lines.append(f"- {line}")

    parts = [
        f"Q: {query}",
        f"T: Title: {title_marked}",
        f"Header: | {' | '.join(hdr_marked)} |",
        "Rows:",
        *row_lines
    ]
    return "\n".join(parts)


# =============== 数据集与 DataLoader ===============

class RerankDataset(Dataset):
    def __init__(self, entries: List[Dict[str, Any]], table_db: Dict[str, Any],
                 tokenizer: AutoTokenizer, max_len: int = 512, neg_weight_rand: float = 0.5):
        """
        entries: [{"qid","query","table_id","label"(0/1), "neg_src" (optional)} ...]
        neg_weight_rand: 对随机负例的降权系数（通过 sample 权重实现）
        """
        self.entries = entries
        self.table_db = table_db
        self.tok = tokenizer
        self.max_len = max_len
        # 采样权重：随机负例降权，hard 负例正常
        self.weights = []
        for e in entries:
            if e.get("label", 0) == 1:
                self.weights.append(1.0)
            else:
                self.weights.append(neg_weight_rand if e.get("neg_src") == "rand" else 1.0)

    def __len__(self): return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        q = e["query"]
        tid = e["table_id"]
        tbl = self.table_db.get(tid, {"title":"", "headers":[], "rows":[]})

        text = serialize_table_marked(q, tbl, max_rows=50)
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",  # ← 原来是 padding=False
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        # BCE with logits → label float
        item["labels"] = torch.tensor([float(e.get("label", 0))], dtype=torch.float)
        # 可选：传权重（Trainer 不直接用，可自行在 loss 中使用；此处先返回备用）
        item["sample_weight"] = torch.tensor(self.weights[idx], dtype=torch.float)
        item["qid"] = e.get("qid", "")
        item["table_id"] = tid
        return item


# =============== 评测：nDCG@10（基于验证集） ===============

def eval_ndcg_at_10(trainer: Trainer, dataset: RerankDataset) -> float:
    """
    简单 nDCG@10：按 qid 分组，对每个 qid 排序后计算 ndcg@10（label=1 的为相关）。
    """
    trainer.model.eval()
    preds_per_qid = defaultdict(list)
    labels_per_qid = defaultdict(list)
    # 逐条推理（小验证集可接受；大验证集可用 DataLoader batched）
    for i in tqdm(range(len(dataset)), desc="eval"):
        ex = dataset[i]
        inputs = {k: v.unsqueeze(0).to(trainer.args.device) for k, v in ex.items()
                  if k in ["input_ids","attention_mask","token_type_ids"] and isinstance(ex[k], torch.Tensor)}
        with torch.no_grad():
            out = trainer.model(**inputs)
            score = out.logits.squeeze(-1).detach().cpu().item()
        qid = ex["qid"] if isinstance(ex["qid"], str) else ""
        label = int(ex["labels"].item())
        preds_per_qid[qid].append(score)
        labels_per_qid[qid].append(label)

    ndcgs = []
    for qid in preds_per_qid:
        y_true = np.asarray([labels_per_qid[qid]], dtype=float)  # shape (1, n_docs)
        y_score = np.asarray([preds_per_qid[qid]], dtype=float)
        nd = ndcg_score(y_true, y_score, k=10)
        ndcgs.append(nd)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


# =============== 读写 mined JSONL ===============

def load_entries_from_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def split_train_dev(entries: List[Dict[str, Any]], dev_ratio: float = 0.1):
    # 按 qid 切分，避免泄漏
    qids = list({e.get("qid","") for e in entries})
    rng = random.Random(SEED)
    rng.shuffle(qids)
    n_dev = max(1, int(len(qids) * dev_ratio))
    dev_qids = set(qids[:n_dev])
    train_set, dev_set = [], []
    for e in entries:
        (dev_set if e.get("qid","") in dev_qids else train_set).append(e)
    return train_set, dev_set


# ---- 兼容旧版 transformers：覆写 compute_loss ----
# ---- 兼容旧版/新版 transformers：覆写 compute_loss ----
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 取出标签与可选的样本权重；inputs 里只保留模型需要的键
        labels = inputs.pop("labels")                          # (B,) or (B,1)
        sample_weight = inputs.pop("sample_weight", None)      # (B,) or (B,1)

        outputs = model(**inputs)                              # forward
        logits = outputs.logits                                # (B,) or (B,1)

        # 形状对齐
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        labels = labels.to(logits.dtype)

        # BCE-with-logits（逐样本）
        loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )

        if sample_weight is not None:
            if sample_weight.dim() == 1:
                sample_weight = sample_weight.unsqueeze(-1)
            sample_weight = sample_weight.to(loss_raw.dtype)
            loss = (loss_raw * sample_weight).mean()
        else:
            loss = loss_raw.mean()

        return (loss, outputs) if return_outputs else loss



# =============== 训练入口 ===============

def main():
    os.makedirs(os.path.dirname(MINED_JSONL), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(SEED)

    print("[LOAD] 载入表库 ...")
    table_ids, table_db = load_all_tables(PATH_TABLE_JSON)

    # SBERT 索引与负例挖掘（若已存在 mined 文件可跳过）
    if not os.path.exists(MINED_JSONL):
        sbert_index = SBERTIndex(SBERT_MODEL_NAME, table_ids, table_db)
        mine_pairs_to_jsonl(PATH_CLAIM_JSONL, table_ids, sbert_index, MINED_JSONL,
                            topk=TOPK, hard_m=HARD_M, rnd_m=RND_M)
    else:
        print(f"[MINE] 跳过挖掘，已存在 {MINED_JSONL}")

    print("[DATA] 读取挖掘结果 ...")
    entries = load_entries_from_jsonl(MINED_JSONL)
    train_entries, dev_entries = split_train_dev(entries, DEV_RATIO)
    print(f"[DATA] train={len(train_entries)}, dev={len(dev_entries)}")

    # Tokenizer 与 模型（注册特殊 token）
    print("[MODEL] 准备 tokenizer / model ...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, num_labels=1
    )
    model.resize_token_embeddings(len(tok))

    # Dataset
    train_ds = RerankDataset(train_entries, table_db, tok, max_len=MAX_LEN, neg_weight_rand=0.5)
    dev_ds = RerankDataset(dev_entries, table_db, tok, max_len=MAX_LEN, neg_weight_rand=0.5)

    # 训练参数
    # ---- 版本兼容的 TrainingArguments 构造 ----
    from inspect import signature

    ta_kwargs = dict(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=100,
        warmup_ratio=WARMUP_RATIO,
        report_to=[],  # 不上报到 wandb 等
    )

    # 旧版本没有 save_strategy，用 save_steps 兜底
    ta_kwargs["save_steps"] = 1000
    ta_kwargs["save_total_limit"] = 2

    # 有些版本不支持 bf16 / fp16，这里按签名动态添加
    sig = signature(TrainingArguments.__init__)
    if "bf16" in sig.parameters:
        ta_kwargs["bf16"] = torch.cuda.is_available()
    elif "fp16" in sig.parameters:
        ta_kwargs["fp16"] = torch.cuda.is_available()

    # 同理，如果你的版本恰好支持 evaluation_strategy / save_strategy，再加上也行
    if "evaluation_strategy" in sig.parameters:
        ta_kwargs["evaluation_strategy"] = "no"  # 我们手动 eval
    if "save_strategy" in sig.parameters:
        ta_kwargs["save_strategy"] = "steps"

    args = TrainingArguments(**ta_kwargs)

    # 自定义 loss：BCEWithLogits（默认 AutoModelForSequenceClassification 用 MSE/CE）
    # 我们在 compute_loss 中覆盖
    def compute_loss(model, inputs, return_outputs=False):
        labels = inputs.pop("labels").view(-1, 1)  # (B,1)
        # 可选使用 sample_weight
        sample_weight = inputs.pop("sample_weight", None)
        outputs = model(**inputs)
        logits = outputs.logits  # (B,1)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        loss_raw = loss_fn(logits, labels)
        if sample_weight is not None:
            # 扩展权重到 (B,1)
            w = sample_weight.view_as(loss_raw)
            loss = (loss_raw * w).mean()
        else:
            loss = loss_raw.mean()
        return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds if EVAL_EVERY_EPOCH else None,
        data_collator=default_data_collator,
        # 不再传 tokenizer= 与 compute_loss=
    )

    print("[TRAIN] 开始训练 ...")
    trainer.train()

    if len(dev_ds) > 0:
        print("[EVAL] 计算 nDCG@10 ...")
        ndcg10 = eval_ndcg_at_10(trainer, dev_ds)
        print(f"[EVAL] nDCG@10 = {ndcg10:.4f}")

    print("[SAVE] 保存模型与 tokenizer ...")
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print("[DONE] 训练完成。")

# =========================
# Inference Utilities
# =========================
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# —— 复用你脚本里已有的这些函数/类 ——
# load_all_tables, serialize_table_for_retrieval, serialize_table_marked,
# SBERTIndex, PATH_TABLE_JSON, SBERT_MODEL_NAME, OUTPUT_DIR,
# MAX_LEN, BATCH_SIZE

def load_reranker(model_dir: str = OUTPUT_DIR, device: str = None):
    """
    加载交叉编码器重排模型与 tokenizer。
    要求：model_dir 是你 Trainer 保存的目录（包含 config.json, pytorch_model.bin, tokenizer.json 等）
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1)
    model.to(device).eval()
    return tok, model, device


def build_sbert_index(path_table_json: str = PATH_TABLE_JSON,
                      sbert_model_name: str = SBERT_MODEL_NAME) -> tuple[SBERTIndex, list[str], dict]:
    """
    一次性构建 SBERT 召回索引；返回 (index, table_ids, table_db)。
    注意：这一步会把整库表编码一遍，时间较长；建议在进程内缓存/复用。
    """
    table_ids, table_db = load_all_tables(path_table_json)
    sbert_index = SBERTIndex(sbert_model_name, table_ids, table_db)
    return sbert_index, table_ids, table_db


@torch.no_grad()
def rerank_query(query: str,
                 sbert_index: SBERTIndex,
                 table_db: dict,
                 reranker_tok: AutoTokenizer,
                 reranker_model: AutoModelForSequenceClassification,
                 topk_return: int = 10,
                 topk_recall: int = 200,
                 max_len: int = MAX_LEN,
                 batch_size: int = BATCH_SIZE,
                 device: str = None) -> list[tuple[str, float]]:
    """
    端到端推理：SBERT 取候选 topk_recall → 交叉编码器重排 → 返回前 topk_return 的 (table_id, score)

    参数：
      - query: 输入查询
      - sbert_index / table_db: 召回与表库
      - reranker_tok / reranker_model: load_reranker() 返回的 tokenizer & model
      - topk_recall: SBERT 候选池大小（例如 200/500/1000）
      - topk_return: 返回最终数量
      - max_len / batch_size: 与训练一致（训练时我们 padding="max_length"）
    """
    if device is None:
        device = next(reranker_model.parameters()).device

    # 1) SBERT 粗排获取候选表 ID
    cand_ids = sbert_index.retrieve(query, topk=topk_recall)

    # 2) 构造交叉编码器输入（带高亮标记）
    texts = [serialize_table_marked(query, table_db.get(tid, {}), max_rows=50)
             for tid in cand_ids]

    # 3) 批量编码 + 前向，得到 logits 作为重排分数
    scores = []
    for s in range(0, len(texts), batch_size):
        e = min(s + batch_size, len(texts))
        enc = reranker_tok(
            texts[s:e],
            truncation=True,
            max_length=max_len,
            padding="max_length",   # 与训练设置保持一致
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = reranker_model(**enc)
        batch_scores = out.logits.squeeze(-1).detach().cpu().tolist()
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)

    # 4) 依据分数排序并截断
    pairs = list(zip(cand_ids, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:topk_return]


def rerank_query_uids(query: str,
                      sbert_index: SBERTIndex,
                      table_db: dict,
                      reranker_tok: AutoTokenizer,
                      reranker_model: AutoModelForSequenceClassification,
                      topk_return: int = 10,
                      topk_recall: int = 200,
                      **kwargs) -> list[str]:
    """只返回 uid 列表的便捷封装。"""
    return [tid for tid, _ in rerank_query(
        query, sbert_index, table_db, reranker_tok, reranker_model,
        topk_return=topk_return, topk_recall=topk_recall, **kwargs
    )]

# =========================
# SBERT Index Persistence
# =========================
import os, json, hashlib
import numpy as np


SBERT_EMB_NPY   = "embeddings.f32.npy"
SBERT_IDS_NPY   = "table_ids.npy"
SBERT_META_JSON = "meta.json"
SBERT_FAISS_IDX = "faiss.index"


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def _texts_fingerprint(table_texts, sample=200):
    """给索引一个简短指纹，便于检查缓存是否对应同一份语料。"""
    m = hashlib.sha1()
    m.update(str(len(table_texts)).encode("utf-8"))
    for t in table_texts[:min(sample, len(table_texts))]:
        m.update(t[:1024].encode("utf-8", errors="ignore"))
    return m.hexdigest()


def save_sbert_index(sbert_index: "SBERTIndex",
                     cache_dir: str = SBERT_CACHE_DIR,
                     save_faiss: bool = True):
    """
    将 SBERTIndex 持久化到磁盘。
    - 保存归一化后的表向量 (float32) 到 .npy
    - 保存 table_ids
    - 如果安装了 faiss 并且 index 存在，保存 faiss.index
    - 保存 meta.json（包含模型名、维度、语料指纹等）
    """
    _ensure_dir(cache_dir)
    # 向量
    np.save(os.path.join(cache_dir, SBERT_EMB_NPY), sbert_index.embs.astype(np.float32))
    # id 顺序
    np.save(os.path.join(cache_dir, SBERT_IDS_NPY), np.array(sbert_index.table_ids, dtype=object))
    # faiss
    if save_faiss and (faiss is not None) and (getattr(sbert_index, "index", None) is not None):
        faiss.write_index(sbert_index.index, os.path.join(cache_dir, SBERT_FAISS_IDX))
    # meta
    meta = {
        "sbert_model": getattr(sbert_index, "model_name", "UNKNOWN"),
        "dim": int(sbert_index.dim),
        "normalized": True,
        "n_tables": int(len(sbert_index.table_ids)),
        "serializer": "serialize_table_for_retrieval",
        "fingerprint": getattr(sbert_index, "fingerprint", None),
        "has_faiss": bool((faiss is not None) and (getattr(sbert_index, "index", None) is not None)),
    }
    with open(os.path.join(cache_dir, SBERT_META_JSON), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[SBERT-CACHE] Saved to {cache_dir}")


def load_sbert_index(path_table_json: str = PATH_TABLE_JSON,
                     sbert_model_name: str = SBERT_MODEL_NAME,
                     cache_dir: str = SBERT_CACHE_DIR) -> tuple["SBERTIndex", list[str], dict]:
    """
    优先从磁盘缓存加载；不存在则现建并保存。
    返回: (sbert_index, table_ids, table_db)
    """
    table_ids, table_db = load_all_tables(path_table_json)

    cache_ok = (
        os.path.exists(os.path.join(cache_dir, SBERT_EMB_NPY))
        and os.path.exists(os.path.join(cache_dir, SBERT_IDS_NPY))
        and os.path.exists(os.path.join(cache_dir, SBERT_META_JSON))
    )

    if cache_ok:
        # 读取缓存
        meta = json.load(open(os.path.join(cache_dir, SBERT_META_JSON), "r", encoding="utf-8"))
        ids_cached = np.load(os.path.join(cache_dir, SBERT_IDS_NPY), allow_pickle=True).tolist()
        if ids_cached == table_ids:
            embs = np.load(os.path.join(cache_dir, SBERT_EMB_NPY))
            # 构造 SBERTIndex（绕过重新编码）
            idx = SBERTIndex.__new__(SBERTIndex)  # 不走 __init__
            idx.model_name = sbert_model_name
            idx.model = SentenceTransformer(sbert_model_name)  # 仅加载查询侧的编码器
            idx.table_ids = table_ids
            idx.table_texts = None  # 不再需要
            idx.embs = embs
            idx.dim = embs.shape[1]
            idx.id_map = {i: tid for i, tid in enumerate(table_ids)}
            # faiss（如可用且存在）
            faiss_path = os.path.join(cache_dir, SBERT_FAISS_IDX)
            if faiss is not None and os.path.exists(faiss_path):
                idx.index = faiss.read_index(faiss_path)
            else:
                idx.index = None
            print(f"[SBERT-CACHE] Loaded from {cache_dir}  (n={len(table_ids)}, dim={idx.dim})")
            return idx, table_ids, table_db
        else:
            print("[SBERT-CACHE] ID 顺序变化，重建索引…")

    # 无缓存或缓存失配：现建并保存
    sbert_index = SBERTIndex(sbert_model_name, table_ids, table_db)
    # 生成语料指纹并保存
    sbert_index.model_name = sbert_model_name
    sbert_index.fingerprint = _texts_fingerprint(sbert_index.table_texts)
    save_sbert_index(sbert_index, cache_dir=cache_dir, save_faiss=True)
    return sbert_index, table_ids, table_db

if __name__ == "__main__":
    main()
