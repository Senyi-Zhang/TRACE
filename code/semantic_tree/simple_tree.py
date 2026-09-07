# -*- coding: utf-8 -*-
# Copy-Span Phrase Tree (CSPT) 最小示例
# 目标：每个节点都是原句里的“原样子串 + 字符区间”，不做改写

import spacy
from typing import List, Dict

nlp = spacy.load("en_core_web_sm")

def claim_to_span_tree(text: str) -> Dict:
    doc = nlp(text)

    spans: List[Dict] = []

    def add_span(start: int, end: int):
        if end <= start:
            return
        spans.append({"start": start, "end": end, "text": text[start:end], "children": []})

    for ent in doc.ents:
        add_span(ent.start_char, ent.end_char)


    for chunk in doc.noun_chunks:
        add_span(chunk.start_char, chunk.end_char)

    PREPS = {
        # 常见基础
        "about", "above", "across", "after", "against", "along", "alongside", "amid", "amidst", "among", "amongst",
        "around", "as", "at", "before", "behind", "below", "beneath", "beside", "besides", "between", "beyond",
        "but", "by", "despite", "down", "during", "except", "excepting", "excluding", "following", "for", "from",
        "in", "including", "inside", "into", "like", "minus", "near", "of", "off", "on", "onto", "opposite", "outside",
        "over", "past", "per", "plus", "regarding", "round", "save", "since", "through", "throughout", "till", "to",
        "toward", "towards", "under", "underneath", "unlike", "until", "up", "upon", "versus", "via", "with", "within",
        "without",

        # 拓展&书面/少见
        "aboard", "above", "afore", "afterward", "against", "along", "amid", "apud", "apropos", "athwart", "bar",
        "barring", "circa", "concerning", "considering", "contre", "downstream", "failing", "given", "inside",
        "mid", "midst", "nearby", "notwithstanding", "onto", "opposite", "out", "outside", "pending", "per",
        "post", "pre", "re", "regardless", "respecting", "roundabout", "sans", "throughout", "toward", "under",
        "underneath", "underway", " underneath", "unlike", "vis", "viz", "vis-a-vis", "vis-à-vis", "worth",

        # 学术/拉丁/法律常用
        "ante", "contra", "cum", "de", "ex", "extra", "infra", "inter", "intra", "juxta", "meta", "ob", "per", "post",
        "pre", "pro", "qua", "re", "sub", "supra", "ultra", "versus", "via", "vice",

        # 时间/范围里常见（你之前白名单的超集）
        "across", "amid", "around", "during", "between", "from", "in", "over", "since", "through", "throughout",
        "till", "until", "within", "beyond", "before", "after", "past"
    }

    for i, tok in enumerate(doc):
        if tok.pos_ == "ADP" and tok.text.lower() in PREPS:
            # 从介词起，向右扩到句末或遇到逗号/句号
            j = i
            end_char = tok.idx + len(tok)
            while j + 1 < len(doc):
                j += 1
                end_char = doc[j].idx + len(doc[j])
                if doc[j].text in {",", ".", ";", "!", "?"}:
                    end_char = doc[j].idx  # 不含标点
                    break
            add_span(tok.idx, end_char)

    COORDS = {"and", "or"}
    noun_spans = sorted(
        [s for s in spans if any(c.isalpha() for c in s["text"])],
        key=lambda s: s["start"]
    )
    for i in range(len(noun_spans) - 1):
        left, right = noun_spans[i], noun_spans[i + 1]
        middle = text[left["end"]: right["start"]].lower()
        if any(conn in middle for conn in COORDS) or "both" in text[:left["start"]].lower():
            add_span(left["start"], right["end"])


    def is_pp(s: Dict) -> bool:
        return s["text"].strip().split(" ", 1)[0].lower() in PREPS
    pp_spans = [s for s in spans if is_pp(s)]
    merged = []
    for s in spans:
        if " and " in s["text"] or " or " in s["text"]:
            for pp in pp_spans:
                between = text[s["end"]: pp["start"]]
                if between.strip() == "":
                    merged.append({"start": s["start"], "end": pp["end"],
                                   "text": text[s["start"]:pp["end"]], "children": []})
    spans.extend(merged)


    root = {"start": 0, "end": len(text), "text": text, "children": []}
    nodes = [root]

    uniq = {}
    for s in spans:
        uniq[(s["start"], s["end"])] = s
    spans = list(uniq.values())

    for s in sorted(spans, key=lambda x: (x["end"] - x["start"])):

        parents = [n for n in nodes if n["start"] <= s["start"] and n["end"] >= s["end"]]

        parent = min(parents, key=lambda n: (n["end"] - n["start"])) if parents else root
        parent["children"].append(s)
        nodes.append(s)

    return root

def print_tree(node: Dict, indent: int = 0):
    span = f'[{node["start"]}:{node["end"]}] "{node["text"]}"'
    prefix = "  " * indent + ("- " if indent else "")
    print(prefix + span)
    for ch in sorted(node.get("children", []), key=lambda x: (x["start"], x["end"])):
        print_tree(ch, indent + 1)

if __name__ == "__main__":
    claim = "Shari Belafonte appeared in both television movies and feature films across the 1980s and 1990s."
    tree = claim_to_span_tree(claim)
    print_tree(tree)
