"""Word2Vec：模型读取、近义词扩展、本地训练。

模型格式（word2vec.bin）：<II 词数,维度> + float32 向量矩阵 + <I json长度> + 词表 JSON，
旁边的 word2vec_vocab_meta.json 存每个词的 count / rank。也兼容 gensim .model/.kv 与标准 word2vec .bin/.txt。

命令行：
    python w2v.py expand 大数据 人工智能 --model models\\word2vec.bin
    python w2v.py train --data data\\管理层讨论与分析 --out models\\my\\word2vec.bin
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import core  # numpy / pandas / gensim 都在函数里再导入，分词子进程不需要它们

MODEL_DIR = core.BASE_DIR / "models"
DEFAULT_MODEL = MODEL_DIR / "word2vec.bin"           # 预训练模型（从 GitHub Release 下载后放这里）
META_NAME = "word2vec_vocab_meta.json"

TRAIN_DEFAULTS = dict(vector_size=200, sg=1, window=5, negative=10, min_count=10, epochs=5)

# ---------------------------------------------------------------- 过滤与置信度

CHINESE_WORD = re.compile(r"^[一-鿿]{2,}$")
ORDINAL_NOISE = re.compile(r"^第?[一二三四五六七八九十百千万〇零]+(次|届|期|号|条|款|项)$")
BARE_ORDINAL = re.compile(r"^第[一二三四五六七八九十百千万〇零]{2,}$")
SHORT_GLUE = re.compile(r"^[一-鿿][为于在的是了不]$")
SUSPICIOUS_ORDINAL = re.compile(r"^第?[一二三四五六七八九十百千万〇零]{2,}$")
COMMON_SURNAME = re.compile(
    r"^[赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹]")
ORG_TENDENCY = re.compile(r"(公司|集团|银行|证券|控股|股份|委员|委员会|中心|研究院|协会|交易所|管理局)$")
DISPLAY_ALLOWLIST = {"扣非", "归母", "净利", "商誉", "摊销", "计提", "折现", "抵扣", "列报", "储能", "锂电",
                     "股改", "认缴", "实缴", "信披", "独董"}
CONF_LABEL = {"high": "高", "medium": "中", "low": "低"}


class Model:
    """归一化后的向量矩阵 + 词表 + 词频/rank 元数据。"""

    def __init__(self, vocab, vectors, meta, path):
        import numpy as np
        self.vocab = list(vocab)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.vectors = (vectors / norms).astype(np.float32)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        self.meta = meta
        self.path = str(path)
        self.vector_size = self.vectors.shape[1]
        self._bad = core.load_word_set(core.BAD_TOKENS_FILE)
        self._hard = None

    def __len__(self):
        return len(self.vocab)

    def is_blocked(self, w):
        return w in self._bad

    def hard_filter_mask(self):
        import numpy as np
        if self._hard is None:
            self._hard = np.array([is_hard_filtered(w, self._bad) for w in self.vocab], dtype=bool)
        return self._hard


def is_hard_filtered(word, bad):
    if word in DISPLAY_ALLOWLIST:
        return False
    return (word in bad or not CHINESE_WORD.match(word) or bool(ORDINAL_NOISE.match(word))
            or bool(BARE_ORDINAL.match(word)) or bool(SHORT_GLUE.match(word)))


def mark_confidence(word, sim, meta):
    count = meta.get("count") if meta else None
    reasons = []
    if count is not None and count < 20:
        reasons.append("低频词")
    if word not in DISPLAY_ALLOWLIST:
        if SUSPICIOUS_ORDINAL.match(word):
            reasons.append("疑似序数")
        if (2 <= len(word) <= 3 and COMMON_SURNAME.match(word)) or ORG_TENDENCY.search(word):
            reasons.append("疑似专名/机构名")
        if count is not None and count < 50 and SHORT_GLUE.match(word):
            reasons.append("低频且形态可疑")
    conf = "low" if reasons else ("medium" if count is not None and count < 100 else "high")
    return {"word": word, "similarity": sim, "count": count, "rank": meta.get("rank") if meta else None,
            "confidence": conf, "hidden": conf == "low", "reason": "；".join(reasons)}


def _log10p(x):
    import math
    return math.log10(x + 1)


def result_score(r):
    boost = 0 if r["count"] is None else _log10p(r["count"]) * 0.01
    return r["similarity"] + boost - (0.03 if r["hidden"] else 0)


def multi_seed_score(r):
    best_rank = max((1 / s["rank"] for s in r["seed_scores"]), default=0)
    coverage = _log10p(len(r["seed_scores"])) * 0.04
    boost = 0 if r["count"] is None else _log10p(r["count"]) * 5e-3
    return best_rank + r["similarity"] * 0.1 + coverage + boost - (0.03 if r["hidden"] else 0)


# ---------------------------------------------------------------- 加载

def is_packed_bin(path) -> bool:
    p = Path(path)
    if not p.exists() or p.stat().st_size < 12:
        return False
    with open(p, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        if not (0 < n <= 10_000_000 and 0 < d <= 4096):
            return False
        off = 8 + n * d * 4
        if off + 4 > p.stat().st_size:
            return False
        f.seek(off)
        (jl,) = struct.unpack("<I", f.read(4))
        return off + 4 + jl <= p.stat().st_size


def load_model(path) -> Model:
    import numpy as np
    path = Path(path)
    meta_path = path.with_name(META_NAME)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if path.suffix.lower() == ".bin" and is_packed_bin(path):
        raw = path.read_bytes()
        n, d = struct.unpack_from("<II", raw, 0)
        vec = np.frombuffer(raw, dtype=np.float32, count=n * d, offset=8).reshape(n, d)
        off = 8 + n * d * 4
        (jl,) = struct.unpack_from("<I", raw, off)
        vocab = json.loads(raw[off + 4: off + 4 + jl].decode("utf-8"))
        return Model(vocab, vec, meta, path)
    from gensim.models import KeyedVectors, Word2Vec
    low = path.suffix.lower()
    if low == ".bin":
        kv = KeyedVectors.load_word2vec_format(str(path), binary=True)
    elif low in (".txt", ".vec"):
        kv = KeyedVectors.load_word2vec_format(str(path), binary=False)
    else:
        try:
            kv = Word2Vec.load(str(path)).wv
        except Exception:
            kv = KeyedVectors.load(str(path))
    if not meta:
        for i, w in enumerate(kv.index_to_key):
            try:
                meta[w] = {"count": int(kv.get_vecattr(w, "count")), "rank": i}
            except Exception:
                meta[w] = {"rank": i}
    return Model(kv.index_to_key, kv.vectors, meta, path)


def available_models() -> list[Path]:
    if not MODEL_DIR.exists():
        return []
    out = [p for p in MODEL_DIR.rglob("*") if p.is_file() and
           (p.suffix.lower() == ".bin" or p.suffix.lower() in (".model", ".kv"))
           and "downloads" not in p.parts]
    out.sort(key=lambda p: (p != DEFAULT_MODEL, str(p)))
    return out


# ---------------------------------------------------------------- 查询 & 扩展

_jieba_ready = False


def _jieba():
    global _jieba_ready
    if not _jieba_ready:
        core.setup_jieba()
        _jieba_ready = True
    import jieba
    return jieba


def query_vector(m: Model, word):
    """词在词表里直接用；不在则用分词结果的平均向量。"""
    import numpy as np
    if m.is_blocked(word):
        return None, []
    if word in m.index:
        return m.vectors[m.index[word]], [word]
    jb = _jieba()
    toks = [t.strip() for t in list(jb.cut_for_search(word)) + list(jb.cut(word))]
    matched = list(dict.fromkeys(t for t in toks if t and t != word and not m.is_blocked(t) and t in m.index))
    if not matched:
        return None, []
    v = np.sum([m.vectors[m.index[t]] for t in matched], axis=0)
    nv = np.linalg.norm(v)
    return (v / nv if nv else v), matched


def similar(m: Model, word, topn=30, min_sim=0.0):
    import numpy as np
    qv, matched = query_vector(m, word)
    if qv is None:
        return [], matched
    sims = m.vectors @ qv
    mask = ~m.hard_filter_mask()
    for t in matched:
        mask[m.index[t]] = False
    mask &= sims >= min_sim
    idx = np.nonzero(mask)[0]
    # 排序分 displayScore = 相似度 + 词频加成(≤0.08) − 低置信扣分(0.03)，
    # 所以只需精排相似度不低于“第 topn 名相似度 − 0.12”的词，结果与全量排序相同
    if len(idx) > topn:
        kth = np.partition(sims[idx], len(idx) - topn)[len(idx) - topn]
        idx = idx[sims[idx] >= kth - 0.12]
    res = [mark_confidence(m.vocab[i], round(float(sims[i]), 4), m.meta.get(m.vocab[i])) for i in idx]
    res.sort(key=result_score, reverse=True)
    return res[:topn], matched


def expand(m: Model, seeds, topn=20, min_sim=0.70, show_low=False, show_seeds=False):
    """多种子词合并。返回 (DataFrame, 信息字典)。"""
    import pandas as pd
    seed_set = set(seeds)
    merged, errors = {}, []
    for seed in seeds:
        res, matched = similar(m, seed, max(100, topn * 5), 0)
        if not matched:
            errors.append(f"{seed}：该词及其分词结果均不在模型词表中")
            continue
        for rank, r in enumerate(res, 1):
            e = merged.get(r["word"])
            if e is None:
                merged[r["word"]] = {**r, "seed": seed, "seed_scores": [{"seed": seed, "similarity": r["similarity"], "rank": rank}],
                                     "is_seed": r["word"] in seed_set}
                continue
            e["seed_scores"].append({"seed": seed, "similarity": r["similarity"], "rank": rank})
            if r["similarity"] > e["similarity"]:
                e.update({k: r[k] for k in ("similarity", "count", "rank", "confidence", "hidden", "reason")})
            labels = [s["seed"] for s in sorted(e["seed_scores"], key=lambda s: -s["similarity"])]
            e["seed"] = f"{'、'.join(labels[:2])} 等{len(labels)}个" if len(labels) > 2 else "、".join(labels)
    allr = list(merged.values())
    allr.sort(key=multi_seed_score if len(seeds) > 1 else result_score, reverse=True)
    by_sim = [r for r in allr if r["similarity"] >= min_sim]
    hidden_low = sum(1 for r in by_sim if r["hidden"])
    by_conf = [r for r in by_sim if show_low or not r["hidden"]]
    hidden_seed = sum(1 for r in by_conf if r["is_seed"])
    final = [r for r in by_conf if show_seeds or not r["is_seed"]][:topn]
    df = pd.DataFrame([{
        "推荐词": r["word"], "相似度": r["similarity"], "词频": r["count"],
        "rank": r["rank"], "置信度": CONF_LABEL[r["confidence"]], "标注": r["reason"] or "-",
        "种子词": r["seed"],
    } for r in final])
    return df, {"errors": errors, "hidden_low": hidden_low, "hidden_seed": hidden_seed, "total": len(by_sim)}


# ---------------------------------------------------------------- 训练（语料清洗见 text_cleaning.py）

_STOP: set[str] = set()
_BAD: set[str] = set()


def _init_tok(resources):
    global _STOP, _BAD
    _, _STOP, _BAD = core.setup_jieba(resources, del_bad=True)


def _tokenize_doc(path: str) -> str:
    import jieba
    from text_cleaning import VALID_WORD, cleaned_sentences
    sents, _ = cleaned_sentences(core.read_text(path))
    lines = []
    for s in sents:
        ws = [w.strip() for w in jieba.lcut(s, HMM=False)]
        ws = [w for w in ws if w and w not in _STOP and w not in _BAD and VALID_WORD.match(w)]
        if len(ws) >= 3:
            lines.append(" ".join(ws))
    return "\n".join(lines)


def _tokenize_batch(args):
    """子进程把一批文件的分词结果直接写进自己的分片文件，只回传文件数，避免结果堆在主进程内存里。"""
    paths, shard = args
    with open(shard, "w", encoding="utf-8") as f:
        for p in paths:
            chunk = _tokenize_doc(p)
            if chunk:
                f.write(chunk)
                f.write("\n")
    return len(paths)


def build_corpus(docs, corpus_path, resources=None, workers=None, progress=None, batch=200):
    corpus_path = Path(corpus_path)
    shard_dir = corpus_path.with_name(corpus_path.stem + "_shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    paths = [d.path for d in docs]
    jobs = [(paths[i:i + batch], str(shard_dir / f"{i // batch:05d}.txt")) for i in range(0, len(paths), batch)]
    done, total = 0, len(paths)
    with ProcessPoolExecutor(workers or core.suggest_workers(), initializer=_init_tok, initargs=(resources,)) as ex:
        for n in ex.map(_tokenize_batch, jobs):
            done += n
            if progress:
                progress(done, total)
    with open(corpus_path, "wb") as out:
        for _, shard in jobs:
            with open(shard, "rb") as f:
                shutil.copyfileobj(f, out, 16 * 1024 * 1024)
    shutil.rmtree(shard_dir, ignore_errors=True)
    return corpus_path


def save_bin_format(wv, out_bin):
    """导出 word2vec.bin + word2vec_vocab_meta.json（本项目的模型格式）。"""
    import numpy as np
    out_bin = Path(out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    vocab = list(wv.index_to_key)
    vj = json.dumps(vocab, ensure_ascii=False).encode("utf-8")
    with open(out_bin, "wb") as f:
        f.write(struct.pack("<II", len(vocab), wv.vector_size))
        f.write(np.ascontiguousarray(wv.vectors, dtype=np.float32).tobytes())
        f.write(struct.pack("<I", len(vj)))
        f.write(vj)
    meta = {w: {"count": int(wv.get_vecattr(w, "count")), "rank": i} for i, w in enumerate(vocab)}
    out_bin.with_name(META_NAME).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def train(corpus_path, out_bin, workers=None, on_epoch=None, **params):
    """gensim 训练；输出 out_bin（.bin 格式）+ meta + .gensim 原生模型。on_epoch(done, total) 每轮回调。"""
    from gensim.models import Word2Vec
    from gensim.models.callbacks import CallbackAny2Vec
    p = {**TRAIN_DEFAULTS, **params}
    workers = workers or max(1, (os.cpu_count() or 2) - 2)  # 训练是线程，不额外占进程内存

    class _Epoch(CallbackAny2Vec):
        def __init__(self):
            self.n = 0

        def on_epoch_end(self, model):
            self.n += 1
            if on_epoch:
                on_epoch(self.n, p["epochs"])

    model = Word2Vec(corpus_file=str(corpus_path), workers=workers, callbacks=[_Epoch()], **p)
    save_bin_format(model.wv, out_bin)
    model.save(str(out_bin) + ".gensim")
    return model


# ---------------------------------------------------------------- CLI

def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="训练模型")
    t.add_argument("--data", nargs="+", default=[str(core.DEFAULT_DATA_DIR)])
    t.add_argument("--out", default=str(MODEL_DIR / "my_model" / "word2vec.bin"))
    t.add_argument("--corpus", default=str(MODEL_DIR / "corpus.txt"))
    t.add_argument("--reuse-corpus", action="store_true", help="已有分词语料时跳过分词")
    t.add_argument("--keep-corpus", action="store_true", help="训练完保留分词语料（约 3GB）")
    t.add_argument("--year-from", type=int)
    t.add_argument("--year-to", type=int)
    for k, v in core.default_resources().items():
        t.add_argument(f"--{k.replace('_', '-')}", default=v)
    for k, v in TRAIN_DEFAULTS.items():
        t.add_argument(f"--{k.replace('_', '-')}", type=int, default=v)
    e = sub.add_parser("expand", help="近义词扩展")
    e.add_argument("seeds", nargs="+")
    e.add_argument("--model", default=str(DEFAULT_MODEL))
    e.add_argument("--topn", type=int, default=20)
    e.add_argument("--threshold", type=float, default=0.70)
    e.add_argument("--show-low", action="store_true")
    e.add_argument("-o", "--output")
    a = ap.parse_args()

    if a.cmd == "train":
        t0 = time.time()
        res = {k: getattr(a, k) for k in core.default_resources()}
        if not (a.reuse_corpus and Path(a.corpus).exists()):
            docs = core.filter_docs(core.discover(a.data), year_from=a.year_from, year_to=a.year_to)
            print(f"分词 {len(docs)} 个文件 ...")
            build_corpus(docs, a.corpus, res, progress=lambda i, n: print(f"\r  {i}/{n}", end="", flush=True))
            print(f"\n分词完成 {time.time() - t0:.0f}s")
        params = {k: getattr(a, k) for k in TRAIN_DEFAULTS}
        print("训练中 ...", params)
        import logging  # gensim 定期打印 "EPOCH x - PROGRESS: at y% examples"
        logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
        m = train(a.corpus, a.out, **params)
        if not a.keep_corpus:
            Path(a.corpus).unlink(missing_ok=True)
        print(f"完成：词表 {len(m.wv)} 词，{m.wv.vector_size} 维，耗时 {time.time() - t0:.0f}s → {a.out}")
    else:
        m = load_model(a.model)
        df, info = expand(m, a.seeds, a.topn, a.threshold, show_low=a.show_low)
        for err in info["errors"]:
            print(err)
        print(df.to_string())
        print(f"（相似度≥阈值共 {info['total']} 个，隐藏低置信 {info['hidden_low']} 个）")
        if a.output:
            df.to_csv(a.output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    _cli()
