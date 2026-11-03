"""年报文本分析 —— 统计核心（词频 / 句频 / TF-IDF）。

词频模式：jieba 精确模式（HMM=False）分词 → 去掉空白、停用词、bad_tokens → 剩余词数为分母；
          关键词若作为完整分词出现，用分词次数，否则退回原文子串计数。
句频模式：按 。！？… 切句，分母为句子数，分子为包含关键词的句子数。
TF-IDF：ratio × idf，idf = ln((N+1)/(df+1)) + 1，N 为导出文档数，df 为 count>0 的文档数。

文件命名约定：股票代码_年份.txt（如 000001_2023.txt）。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

# 子进程会继承这些环境变量：每个进程只给 numpy/OpenBLAS 一个线程，
# 否则 20 个进程 × 22 线程的 BLAS 缓冲区会白白占掉大量内存
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# 注意：pandas 只在用到的函数里导入，子进程只需要 jieba，能省下不少内存

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "管理层讨论与分析"
RES_DIR = BASE_DIR / "resources"
STOPWORDS_FILE = RES_DIR / "stopwords.txt"
CUSTOM_DICT = RES_DIR / "custom_dict.txt"
HIGH_FREQ_DICT = RES_DIR / "high_freq_dict.txt"
BAD_TOKENS_FILE = RES_DIR / "bad_tokens.txt"
METADATA_DB = RES_DIR / "metadata.db"          # 内置公司/行业库（代码、简称、行业、年份）
COMPANY_INFO = RES_DIR / "company_info.xlsx"   # 可选：你自己的公司/行业表，优先级高于 metadata.db
STATE_DIR = BASE_DIR / "state"
RESULTS_DB = STATE_DIR / "results.db"

HIGH_FREQ_DEFAULT = 2_000_000
SENT_SPLIT_RE = re.compile(r"[。！？…]+")
FNAME_RE = re.compile(r"^(\d{6,8})_(\d{4})(?:-\d{2}-\d{2})?")

MODES = {"word": "词频模式", "sentence": "句频模式"}


def default_resources() -> dict:
    return {"stopwords": str(STOPWORDS_FILE), "custom_dict": str(CUSTOM_DICT),
            "high_freq_dict": str(HIGH_FREQ_DICT), "bad_tokens": str(BAD_TOKENS_FILE)}


# ---------------------------------------------------------------- 文件发现

@dataclass
class Doc:
    path: str
    code: str
    year: int | None


def parse_name(path: Path) -> tuple[str, int | None]:
    m = FNAME_RE.match(path.stem)
    if m:
        return m.group(1), int(m.group(2))
    return path.stem, None


def discover(folders: list[str | Path]) -> list[Doc]:
    docs, seen = [], set()
    for folder in folders:
        folder = os.path.abspath(str(folder))
        if os.path.isfile(folder):
            files = [folder]
        elif os.path.isdir(folder):
            files = (os.path.join(root, n) for root, _, names in os.walk(folder)
                     for n in names if n.lower().endswith(".txt"))
        else:
            continue
        for key in files:
            if key in seen:
                continue
            seen.add(key)
            code, year = parse_name(Path(key))
            docs.append(Doc(key, code, year))
    docs.sort(key=lambda d: (d.code, d.year or 0))
    return docs


def parse_codes(text: str) -> set[str] | None:
    """'all' / 空 → None（全部）；否则返回 6 位代码集合。"""
    text = (text or "").strip()
    if not text or text.lower() == "all":
        return None
    parts = re.split(r"[\s,，;；、]+", text)
    return {p.strip().zfill(6) for p in parts if p.strip()}


def filter_docs(docs, codes=None, year_from=None, year_to=None, allowed_codes=None):
    out = []
    for d in docs:
        if codes is not None and d.code not in codes:
            continue
        if allowed_codes is not None and d.code not in allowed_codes:
            continue
        if d.year is not None:
            if year_from and d.year < year_from:
                continue
            if year_to and d.year > year_to:
                continue
        out.append(d)
    return out


def read_text(path: str) -> str:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def read_list_file(path_or_buf, name: str = "") -> list[str]:
    """读取 TXT / XLSX / CSV：每行（第一列）一个条目。"""
    import pandas as pd
    name = name or str(path_or_buf)
    if name.lower().endswith((".xlsx", ".xls")):
        items = pd.read_excel(path_or_buf, header=None, dtype=str).iloc[:, 0].dropna().tolist()
    elif name.lower().endswith(".csv"):
        items = pd.read_csv(path_or_buf, header=None, dtype=str).iloc[:, 0].dropna().tolist()
    else:
        raw = path_or_buf.read() if hasattr(path_or_buf, "read") else Path(path_or_buf).read_bytes()
        if isinstance(raw, bytes):
            for enc in ("utf-8-sig", "gb18030"):
                try:
                    raw = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
        items = raw.splitlines()
    out = []
    for it in items:
        it = str(it).strip()
        if it and it not in out:
            out.append(it)
    return out


def split_keywords(text: str) -> list[str]:
    out = []
    for k in re.split(r"[\s,，;；、]+", text or ""):
        k = k.strip()
        if k and k not in out:
            out.append(k)
    return out


# ---------------------------------------------------------------- 公司 / 行业库

@lru_cache(maxsize=1)
def _metadata():
    """读取 metadata.db：返回 (精确表 {(代码,年份):(简称,行业)}, 兜底表 {代码:(简称,行业)}, 别名表)。"""
    exact, fallback, aliases, rows = {}, {}, {}, []
    if METADATA_DB.exists():
        con = sqlite3.connect(f"file:{METADATA_DB}?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT 代码, 简称, 行业, 年份 FROM companies ORDER BY id").fetchall()
            try:
                for old, new in con.execute("SELECT old_code, new_code FROM code_aliases"):
                    aliases.setdefault(old, set()).add(new)
                    aliases.setdefault(new, set()).add(old)
            except sqlite3.OperationalError:
                pass
        finally:
            con.close()
    for code, name, ind, year in rows:
        exact[(code, int(year))] = (name, ind)
        fallback.setdefault(code, (name, ind))
    # 用户自己的 company_info.xlsx 覆盖（列：股票代码、公司名称、行业，可选 年份）
    if COMPANY_INFO.exists():
        import pandas as pd
        df = pd.read_excel(COMPANY_INFO, dtype=str)
        keys_by_code = {}
        for k in exact:
            keys_by_code.setdefault(k[0], []).append(k)
        for _, r in df.iterrows():
            code = str(r.get("股票代码", "")).strip().zfill(6)
            val = (r.get("公司名称") if pd.notna(r.get("公司名称")) else None,
                   r.get("行业") if pd.notna(r.get("行业")) else None)
            if pd.notna(r.get("年份")) and str(r.get("年份")).strip():
                exact[(code, int(float(r["年份"])))] = val
            else:
                fallback[code] = val
                for k in keys_by_code.get(code, []):
                    exact[k] = val
    return exact, fallback, aliases, rows


def has_metadata() -> bool:
    return bool(_metadata()[3]) or COMPANY_INFO.exists()


def company_info(code: str, year: int | None) -> tuple[str | None, str | None]:
    exact, fallback, _, _ = _metadata()
    if year is not None and (code, year) in exact:
        return exact[(code, year)]
    return fallback.get(code, (None, None))


def industries() -> list[str]:
    exact, fallback, _, _ = _metadata()
    return sorted({v[1] for v in list(exact.values()) + list(fallback.values()) if v[1]})


def codes_by_industry(inds, year_from=None, year_to=None, exclude_st=False) -> set[str]:
    """行业 ∈ inds 且年份在范围内的公司代码；可剔除 ST 与 B 股(900)。"""
    exact, _, _, _ = _metadata()
    inds = set(inds or [])
    out = set()
    for (code, year), (name, ind) in exact.items():
        if inds and ind not in inds:
            continue
        if year_from and year < year_from or year_to and year > year_to:
            continue
        out.add(code)
    if exclude_st:
        bad = {c for (c, _), (n, _) in exact.items() if (n and "ST" in n) or c.startswith("900")}
        out -= bad
    return out


def resolve_codes(codes: set[str] | None) -> set[str] | None:
    """把新旧代码（如北交所 830xxx ↔ 920xxx）都纳入。"""
    if codes is None:
        return None
    aliases = _metadata()[2]
    out = set(codes)
    for c in codes:
        out |= aliases.get(c, set())
    return out


# ---------------------------------------------------------------- 单文件统计（子进程）

_WORDS: list[str] = []
_MODE = "word"
_STOP: set[str] = set()
_BAD: set[str] = set()


def load_word_set(path) -> set[str]:
    words = set()
    if not path or not Path(path).exists():
        return words
    for raw in read_text(str(path)).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        words.add(line.split()[0])
    return words


def load_jieba_dict(jieba, path, default_freq=None):
    if not path or not Path(path).exists():
        return 0
    n = 0
    for raw in read_text(str(path)).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        freq = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else default_freq
        jieba.add_word(parts[0], freq=freq)
        n += 1
    return n


def setup_jieba(resources: dict | None = None, del_bad=False):
    """加载 custom_dict（默认词频）和 high_freq_dict（默认 2e6）；返回 (jieba, 停用词, bad_tokens)。"""
    import jieba
    jieba.setLogLevel(60)
    res = {**default_resources(), **(resources or {})}
    stop = load_word_set(res.get("stopwords"))
    bad = load_word_set(res.get("bad_tokens"))
    load_jieba_dict(jieba, res.get("custom_dict"))
    load_jieba_dict(jieba, res.get("high_freq_dict"), HIGH_FREQ_DEFAULT)
    if del_bad:  # 训练时把 bad_tokens 从词典删掉；统计时只过滤不删除
        for t in bad:
            jieba.del_word(t)
    return jieba, stop, bad


def _init_worker(words, mode, resources):
    global _WORDS, _MODE, _STOP, _BAD
    _WORDS, _MODE = list(words), mode
    if mode == "word":
        _, _STOP, _BAD = setup_jieba(resources)


def count_text(text: str, words, mode: str):
    """返回 (分母, {关键词: 计数})。"""
    if mode == "word":
        import jieba
        counts_all = {}
        total = 0
        for t in jieba.cut(text, HMM=False):
            t = t.strip()
            if not t or t in _STOP or t in _BAD:
                continue
            total += 1
            counts_all[t] = counts_all.get(t, 0) + 1
        return total, {w: counts_all[w] if w in counts_all else text.count(w) for w in words}
    sents = [s for s in SENT_SPLIT_RE.split(text) if s.strip()]
    return len(sents), {w: sum(1 for s in sents if w in s) for w in words}


def _count_batch(docs: list[Doc]):
    out = []
    for d in docs:
        try:
            denom, counts = count_text(read_text(d.path), _WORDS, _MODE)
            out.append((d, denom, counts, None))
        except Exception as e:  # 单个文件失败不影响整体
            out.append((d, 0, {}, str(e)))
    return out


# ---------------------------------------------------------------- 结果缓存（断点续跑）

def _words_sig(words, mode, resources) -> str:
    res = {**default_resources(), **(resources or {})}
    stamp = [(k, v, os.path.getmtime(v) if v and os.path.exists(v) else 0) for k, v in sorted(res.items())]
    return hashlib.sha1(json.dumps([mode, list(words), stamp], ensure_ascii=False).encode()).hexdigest()[:16]


def _db():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(RESULTS_DB, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS task_results (
        key TEXT PRIMARY KEY, path TEXT, code TEXT, year INTEGER, mode TEXT, sig TEXT,
        denom INTEGER, counts_json TEXT, mtime REAL, created_at TEXT)""")
    return con


def clear_cache():
    if RESULTS_DB.exists():
        con = _db()
        con.execute("DELETE FROM task_results")
        con.commit()
        con.close()


def analyze(docs: list[Doc], words: list[str], mode="word", workers: int | None = None,
            resources: dict | None = None, progress=None, batch=16):
    """统计并返回结果列表 [{code, year, path, denom, counts}]。

    已算过的文件（同一关键词表、模式、词典、且文件未修改）直接读缓存，所以中途停止后再次开始会从断点继续。
    progress(done, total, cached) 回调；在回调里抛异常（如 Streamlit 停止）会安全中断，已完成部分保留。
    """
    if not docs or not words:
        return []
    sig = _words_sig(words, mode, resources)
    con = _db()
    results, todo = {}, []
    mtimes = {d.path: os.path.getmtime(d.path) for d in docs}
    keys = {d.path: f"{d.path}|{sig}" for d in docs}
    cached = {}
    key_list = list(keys.values())
    for i in range(0, len(key_list), 900):
        part = key_list[i:i + 900]
        q = f"SELECT key, denom, counts_json, mtime FROM task_results WHERE key IN ({','.join('?' * len(part))})"
        for k, denom, cj, mt in con.execute(q, part):
            cached[k] = (denom, cj, mt)
    for d in docs:
        c = cached.get(keys[d.path])
        if c and abs(c[2] - mtimes[d.path]) < 1e-6:
            results[d.path] = {"code": d.code, "year": d.year, "path": d.path, "denom": c[0],
                               "counts": json.loads(c[1])}
        else:
            todo.append(d)
    total, n_cached = len(docs), len(results)
    if progress:
        progress(n_cached, total, n_cached)

    def save(rows):
        now = datetime.now().isoformat(timespec="seconds")
        con.executemany(
            "INSERT OR REPLACE INTO task_results VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(keys[d.path], d.path, d.code, d.year, mode, sig, denom, json.dumps(counts, ensure_ascii=False),
              mtimes[d.path], now) for d, denom, counts, err in rows if err is None])
        con.commit()
        for d, denom, counts, err in rows:
            if err is None:
                results[d.path] = {"code": d.code, "year": d.year, "path": d.path, "denom": denom, "counts": counts}

    try:
        if todo:
            workers = workers or suggest_workers()
            if workers == 1 or len(todo) < 20:
                _init_worker(words, mode, resources)
                for i in range(0, len(todo), batch):
                    save(_count_batch(todo[i:i + batch]))
                    if progress:
                        progress(len(results), total, n_cached)
            else:
                ex = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(words, mode, resources))
                try:
                    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
                    it = iter(chunks)
                    pending = set()
                    for _ in range(workers * 3):  # 有界提交，不会一次性堆积全部任务
                        c = next(it, None)
                        if c is None:
                            break
                        pending.add(ex.submit(_count_batch, c))
                    last = 0.0
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        rows = []
                        for f in done:
                            rows.extend(f.result())
                            c = next(it, None)
                            if c is not None:
                                pending.add(ex.submit(_count_batch, c))
                        save(rows)
                        if progress and (time.time() - last > 0.3 or not pending):
                            last = time.time()
                            progress(len(results), total, n_cached)
                finally:
                    ex.shutdown(wait=False, cancel_futures=True)
    finally:
        con.close()
    return [results[d.path] for d in docs if d.path in results]


def suggest_workers(per_proc_gb: float = 0.35, cap: int | None = None) -> int:
    """按当前可用“提交内存”估算并行进程数（Windows 页面文件小时，进程太多会被系统杀掉）。"""
    n = max(1, (os.cpu_count() or 2) - 2)
    try:
        import ctypes

        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(MEMSTAT)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        avail_gb = min(m.ullAvailPhys, m.ullAvailPageFile) / 1024 ** 3
        n = min(n, max(1, int((avail_gb - 1.5) / per_proc_gb)))
    except Exception:
        pass
    return min(n, cap) if cap else n


# ---------------------------------------------------------------- 结果表 & 导出

def build_table(results, words, mode):
    """列：代码、行业、年份、分母、{词}_count…、{词}_ratio…、{词}_tfidf…，按代码、年份排序。"""
    import pandas as pd
    n = len(results)
    df_cnt = {w: sum(1 for r in results if r["counts"].get(w, 0) > 0) for w in words}
    idf = {w: math.log((n + 1) / (df_cnt[w] + 1)) + 1 for w in words}
    rows = []
    for r in sorted(results, key=lambda r: (r["code"], r["year"] or 0)):
        _, ind = company_info(r["code"], r["year"])
        row = {"代码": r["code"], "行业": ind or "", "年份": r["year"] or 0, "分母": r["denom"] or 0}
        ratios = {}
        for w in words:
            row[f"{w}_count"] = r["counts"].get(w, 0)
        for w in words:
            ratios[w] = row[f"{w}_count"] / r["denom"] if r["denom"] else 0.0
            row[f"{w}_ratio"] = ratios[w]
        for w in words:
            row[f"{w}_tfidf"] = ratios[w] * idf[w]
        rows.append(row)
    return pd.DataFrame(rows)


def export_stata_csv(df, path: str, mode: str):
    """「Stata (CSV)」格式：前 15 行为说明，第 15 行是变量名，数据从第 16 行开始。"""
    names = ["code", "industry", "year", "denom"] + list(df.columns[4:])
    header = [
        "* 词频统计结果 - Stata导入文件",
        f"* 统计模式: {'词频' if mode == 'word' else '句频'}",
        f"* 生成时间: {datetime.now():%Y/%m/%d %H:%M:%S}",
        "* ",
        "* 导入方法:",
        f'* import delimited using "{path}", clear varnames(15) rowrange(16:)',
        "* ",
        "* 变量说明:",
        "* code=公司代码 industry=行业 year=年份 denom=分母",
        "* 关键词_count=词频计数 关键词_ratio=比率(count/denom) 关键词_tfidf=ratio*idf",
        "* idf=ln((N+1)/(df+1))+1, N为导出文档数, df为包含该词的文档数",
        "* ",
        "* 》》》数据从下一行开始《《《",
        "* ",
        "* ",
    ]

    def esc(s):
        s = "" if s is None else str(s)
        return '"' + s.replace('"', '""') + '"' if any(c in s for c in ',"\n\r') else s

    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in header:
            f.write(('"' + line.replace('"', '""') + '"' if ("," in line or '"' in line) else line) + "\n")
        f.write(",".join(esc(c) for c in names) + "\n")
        for row in df.itertuples(index=False):
            f.write(",".join(esc(v) for v in row) + "\n")


def to_stata(df, path: str):
    """.dta（Stata 15+）：中文变量名直接保留；不合法时退回 v1..vN 并写进变量标签。"""
    out = df.rename(columns={"代码": "code", "行业": "industry", "年份": "year", "分母": "denom"})
    try:
        out.to_stata(path, write_index=False, version=119)
    except Exception:
        labels = {f"v{i}": c[:80] for i, c in enumerate(out.columns, 1)}
        out.columns = list(labels)
        out.to_stata(path, write_index=False, version=118, variable_labels=labels)


def export(df, path: str, fmt: str, mode: str = "word"):
    fmt = fmt.lower()
    if fmt == "xlsx":
        df.to_excel(path, index=False, sheet_name="词频统计结果")
    elif fmt == "stata_csv":
        export_stata_csv(df, path, mode)
    elif fmt == "csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    elif fmt == "dta":
        to_stata(df, path)
    else:
        raise ValueError(fmt)
