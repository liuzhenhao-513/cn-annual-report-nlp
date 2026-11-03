"""命令行批量统计（不开界面时用），口径与界面一致。

用法：
    python analysis.py                         # 用 keywords.txt + 默认数据目录，词频模式，输出 output.xlsx
    python analysis.py -o 我的结果.xlsx
    python analysis.py --mode sentence -f stata_csv -o 我的结果.csv
    python analysis.py --data D:\\我的文本 --codes 000001,600000 --year-from 2010 --year-to 2023
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import core

# 关键词：优先读取同目录 keywords.txt（每行一个），否则用下面的列表
KEYWORDS = ["数字化转型", "大数据", "人工智能", "云计算", "区块链"]
DATA_DIRS = [str(core.DEFAULT_DATA_DIR)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", nargs="+", default=DATA_DIRS, help="文本文件夹（可多个）")
    ap.add_argument("-k", "--keywords", help="关键词文件 txt/xlsx，默认 keywords.txt")
    ap.add_argument("--codes", default="all", help="股票代码，逗号分隔，或 all")
    ap.add_argument("--industry", nargs="*", help="行业（可多个），需 metadata.db 或 company_info.xlsx")
    ap.add_argument("--exclude-st", action="store_true", help="剔除 ST 与 B 股")
    ap.add_argument("--year-from", type=int)
    ap.add_argument("--year-to", type=int)
    ap.add_argument("--mode", default="word", choices=list(core.MODES))
    ap.add_argument("-f", "--format", default=None, choices=["xlsx", "stata_csv", "csv", "dta"])
    ap.add_argument("-o", "--output", default="output.xlsx")
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()

    kw_file = a.keywords or Path(__file__).with_name("keywords.txt")
    keywords = core.read_list_file(kw_file) if Path(kw_file).exists() else KEYWORDS
    fmt = a.format or {"xlsx": "xlsx", "csv": "csv", "dta": "dta"}.get(Path(a.output).suffix.lstrip(".").lower(), "xlsx")

    allowed = None
    if a.industry or a.exclude_st:
        allowed = core.codes_by_industry(a.industry, a.year_from, a.year_to, a.exclude_st)
    docs = core.filter_docs(core.discover(a.data), core.resolve_codes(core.parse_codes(a.codes)),
                            a.year_from, a.year_to, allowed)
    print(f"关键词 {len(keywords)} 个，文件 {len(docs)} 个，{core.MODES[a.mode]}")
    t0 = time.time()
    results = core.analyze(docs, keywords, a.mode, a.workers,
                           progress=lambda i, n, c: print(f"\r  {i}/{n}（缓存 {c}）", end="", flush=True))
    print(f"\n统计完成 {time.time() - t0:.0f}s")
    df = core.build_table(results, keywords, a.mode)
    core.export(df, a.output, fmt, a.mode)
    print(f"已导出 {len(df)} 行 → {a.output}")


if __name__ == "__main__":
    main()
