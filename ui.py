"""年报文本分析界面 —— 由 app.py 启动。"""
from __future__ import annotations

import io
import json
import os
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import core
import w2v

st.set_page_config(page_title="财报文本分析助手（本地版）", layout="wide")

UPLOAD_DIR = core.BASE_DIR / "uploads"
OUTPUT_DIR = core.BASE_DIR / "output"
SESSION_FILE = core.STATE_DIR / "session.json"
LAST_RESULT = core.STATE_DIR / "last_result.pkl"

ss = st.session_state


# ---------------------------------------------------------------- 会话持久化（重启后恢复任务池、词表、结果）

def load_session():
    if "pool" in ss:
        return
    ss.pool, ss.keywords, ss.mode = {}, [], "word"
    ss.result, ss.result_kw, ss.result_mode = None, [], "word"
    ss.w2v_result, ss.running = None, False
    if SESSION_FILE.exists():
        try:
            s = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
            ss.pool = {p: core.Doc(p, c, y) for p, c, y in s.get("pool", []) if os.path.exists(p)}
            ss.keywords = s.get("keywords", [])
            ss.mode = s.get("mode", "word")
            ss.result_kw, ss.result_mode = s.get("result_kw", []), s.get("result_mode", "word")
            ss.running = s.get("running", False)
        except Exception:
            pass
    if LAST_RESULT.exists():
        try:
            ss.result = pd.read_pickle(LAST_RESULT)
        except Exception:
            ss.result = None


def save_session():
    core.STATE_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(json.dumps({
        "pool": [[d.path, d.code, d.year] for d in ss.pool.values()],
        "keywords": ss.keywords, "mode": ss.mode,
        "result_kw": ss.result_kw, "result_mode": ss.result_mode, "running": ss.running,
    }, ensure_ascii=False), encoding="utf-8")


load_session()


# ---------------------------------------------------------------- 工具

@st.cache_data(show_spinner="扫描文本文件 ...")
def scan(folder: str, mtime: float):
    return core.discover([folder])


@st.cache_resource(show_spinner="加载词向量模型 ...")
def get_model(path: str, mtime: float):
    return w2v.load_model(path)


def mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


def add_keywords(words):
    n = 0
    for w in words:
        w = str(w).strip()
        if w and w not in ss.keywords:
            ss.keywords.append(w)
            n += 1
    return n


def save_uploads(files) -> Path:
    dest = UPLOAD_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(f) as z:
                for info in z.infolist():
                    name = info.filename
                    if not (info.flag_bits & 0x800):  # 非 UTF-8 标记的 zip（Windows 压缩）按 GBK 解码文件名
                        try:
                            name = name.encode("cp437").decode("gbk")
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            pass
                    if info.is_dir() or not name.lower().endswith(".txt"):
                        continue
                    with z.open(info) as src, open(dest / Path(name).name, "wb") as out:
                        shutil.copyfileobj(src, out)
        else:
            (dest / f.name).write_bytes(f.getbuffer())
    return dest


def pool_docs():
    return sorted(ss.pool.values(), key=lambda d: (d.code, d.year or 0))


def pool_summary(docs):
    rows = {}
    for d in docs:
        r = rows.setdefault(d.code, {"代码": d.code, "years": [], "info": None})
        r["years"].append(d.year)
        if r["info"] is None or d.year is not None:
            r["info"] = core.company_info(d.code, d.year)
    out = []
    for code, r in rows.items():
        ys = [y for y in r["years"] if y is not None]
        name, ind = r["info"] or (None, None)
        out.append({"选择": False, "代码": code, "公司名称": name or "", "行业": ind or "",
                    "年份范围": f"{min(ys)} - {max(ys)}" if ys else "-", "文件数": len(r["years"])})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 页头

c1, c2 = st.columns([3, 1])
c1.markdown("### 📊 财报文本分析助手 · 本地版")
c2.caption(f"公司/行业库：{'已加载（metadata.db）' if core.METADATA_DB.exists() else '无'}"
           f"{' + company_info.xlsx' if core.COMPANY_INFO.exists() else ''}  \n"
           f"数据目录：`{core.DEFAULT_DATA_DIR.name}`")

tab_main, tab_w2v = st.tabs(["主工作台", "词向量扩展"])

# ================================================================ 主工作台
with tab_main:
    left, right = st.columns([1, 2], gap="large")

    # ------------------------------------------------ 公司检索与导入
    with left:
        st.markdown("#### 1. 公司检索与导入")
        src_stock, src_custom = st.tabs(["股票代码导入", "自定义文本导入"])
        with src_stock:
            data_dir = st.text_input("年报文本目录", str(core.DEFAULT_DATA_DIR),
                                     help="文件需命名为 股票代码_年份.txt，可放在任意子文件夹里")
            codes_text = st.text_input("股票代码", "all",
                                       help="输入 all 选中全部公司；或输入代码，逗号分隔，如 000001,600000")
            code_file = st.file_uploader("或导入代码文件（TXT/XLSX，每行一个代码）",
                                         type=["txt", "xlsx", "xls", "csv"], key="code_file")
            y1, y2 = st.columns(2)
            year_from = y1.number_input("起始年份", 1990, 2100, 2001)
            year_to = y2.number_input("结束年份", 1990, 2100, 2025)
            sel_inds, exclude_st = [], False
            if core.has_metadata():
                sel_inds = st.multiselect("行业筛选（不选 = 全部行业，支持多选）", core.industries())
                exclude_st = st.checkbox("剔除 ST 与 B 股", False)

            if st.button("＋ 添加到任务池", type="primary", width="stretch", key="add_stock"):
                if not Path(data_dir).exists():
                    st.error("目录不存在")
                else:
                    codes = core.parse_codes(codes_text)
                    if code_file is not None:
                        codes = (codes or set()) | {c.zfill(6) for c in core.read_list_file(code_file, code_file.name)}
                    allowed = None
                    if sel_inds or exclude_st:
                        allowed = core.codes_by_industry(sel_inds, int(year_from), int(year_to), exclude_st)
                    all_docs = scan(data_dir, mtime(data_dir))
                    docs = core.filter_docs(all_docs, core.resolve_codes(codes), int(year_from), int(year_to), allowed)
                    before = len(ss.pool)
                    for d in docs:
                        ss.pool[d.path] = d
                    found = {d.code for d in docs}
                    msg = f"已添加 {len(ss.pool) - before} 个文件（{len(found)} 家公司）"
                    if codes:
                        invalid = sorted(c for c in codes if not (core.resolve_codes({c}) & found))
                        msg += f"；导入 {len(codes)} 个代码，有效 {len(codes) - len(invalid)}，无效 {len(invalid)}"
                        if invalid:
                            st.warning("以下代码在该年份范围内没有文本：" + "、".join(invalid[:50]))
                    st.success(msg)

        with src_custom:
            st.caption("分析自己的文本：填文件夹路径，或直接上传 TXT / ZIP。"
                       "命名为 股票代码_年份.txt 时会自动识别代码、年份和行业。")
            custom_dir = st.text_input("文本文件夹路径（含子文件夹）", "", placeholder=r"例如 D:\我的文本")
            ups = st.file_uploader("或上传 TXT / ZIP（可多选）", type=["txt", "zip"],
                                   accept_multiple_files=True, key="custom_up")
            if st.button("＋ 添加到任务池", type="primary", width="stretch", key="add_custom"):
                folders = []
                if custom_dir.strip():
                    if Path(custom_dir.strip()).exists():
                        folders.append(custom_dir.strip())
                    else:
                        st.error("文件夹不存在")
                if ups:
                    folders.append(str(save_uploads(ups)))
                docs = core.discover(folders) if folders else []
                for d in docs:
                    ss.pool[d.path] = d
                st.success(f"已添加 {len(docs)} 个文件") if docs else st.info("没有找到 .txt 文件")

        # ------------------------------------------------ 词表管理
        st.markdown(f"#### 2. 词表管理（当前共有 {len(ss.keywords)} 个词）")
        kc1, kc2 = st.columns([3, 1])
        new_kw = kc1.text_input("添加关键词", "", placeholder="输入关键词，支持逗号、顿号分隔",
                                label_visibility="collapsed")
        if kc2.button("添加", width="stretch"):
            add_keywords(core.split_keywords(new_kw))
            st.rerun()
        kw_file = st.file_uploader("导入词表（TXT/XLSX，每行一个关键词）",
                                   type=["txt", "xlsx", "xls", "csv"], key="kw_file")
        if kw_file is not None and st.button("导入该词表"):
            n = add_keywords(core.read_list_file(kw_file, kw_file.name))
            st.success(f"导入 {n} 个新词")
            st.rerun()
        if ss.keywords:
            kept = st.multiselect("当前词表（点 × 删除）", ss.keywords, default=ss.keywords)
            if kept != ss.keywords:
                ss.keywords = kept
                st.rerun()
            e1, e2 = st.columns(2)
            e1.download_button("导出词表", "\n".join(ss.keywords).encode("utf-8-sig"), "词表.txt", width="stretch")
            if e2.button("清空词表", width="stretch"):
                ss.keywords = []
                st.rerun()

        st.markdown("#### 3. 统计模式（对词表中所有词生效）")
        ss.mode = st.radio(
            "统计模式", list(core.MODES), index=list(core.MODES).index(ss.mode), label_visibility="collapsed",
            format_func=lambda m: {"word": "词频模式：统计词在全文中出现的次数，分母为 jieba Token 总数",
                                   "sentence": "句频模式：统计包含该词的句子数，分母为全文句数（按 。！？… 切分）"}[m])
        st.caption("导出结果中每个词都有 _count、_ratio、_tfidf 三列。")

    # ------------------------------------------------ 任务池 & 执行 & 导出
    with right:
        docs = pool_docs()
        n_comp = len({d.code for d in docs})
        h1, h2, h3 = st.columns([4, 1, 1])
        h1.markdown(f"#### 待处理任务池（共 {n_comp} 个公司，{len(docs)} 个文件）")
        if docs:
            summary = pool_summary(docs)
            edited = st.data_editor(summary, hide_index=True, width="stretch", height=260,
                                    disabled=[c for c in summary.columns if c != "选择"], key="pool_editor")
            chosen = set(edited.loc[edited["选择"], "代码"])
            if h2.button(f"删除选中（{len(chosen)}）", width="stretch", disabled=not chosen):
                ss.pool = {p: d for p, d in ss.pool.items() if d.code not in chosen}
                ss.pop("pool_editor", None)
                st.rerun()
        else:
            st.info("任务池为空：在左侧选择公司后点「添加到任务池」")
        if h3.button("🗑 清空", width="stretch", disabled=not docs):
            ss.pool = {}
            st.rerun()

        st.markdown("#### 执行进度")
        if ss.running:  # 上一次运行被中途停止（点了停止、刷新页面或关闭了程序）
            st.warning("上次任务已停止。已完成的部分都保存了，再次点「开始任务」会从断点继续。")
            ss.running = False
        workers = st.slider("并行进程数", 1, os.cpu_count() or 4, core.suggest_workers(),
                            help="默认值按当前可用内存自动推荐；越多越快，但每个进程约占 0.3GB 内存")
        ready = bool(docs and ss.keywords)
        b1, b2 = st.columns([3, 1])
        start = b1.button("▶ 开始任务", type="primary", disabled=not ready, width="stretch")
        b2.button("⏹ 停止任务", width="stretch", help="停止后进度会保存，再次开始从断点继续")
        if start:
            ss.running = True
            save_session()
            bar = st.progress(0.0, "准备中 ...")
            t0 = time.time()

            def cb(i, n, cached):
                el = time.time() - t0
                new = i - cached
                eta = el / new * (n - i) if new else 0
                bar.progress(i / n, f"已完成 {i} / {n} 个文件（其中 {cached} 个来自缓存）"
                                    f" · 用时 {el:.0f}s · 预计剩余 {eta:.0f}s")

            results = core.analyze(docs, list(ss.keywords), ss.mode, workers, progress=cb)
            df = core.build_table(results, list(ss.keywords), ss.mode)
            ss.result, ss.result_kw, ss.result_mode = df, list(ss.keywords), ss.mode
            ss.running = False
            core.STATE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_pickle(LAST_RESULT)
            bar.progress(1.0, f"完成：{len(df)} 个文件，用时 {time.time() - t0:.0f}s")
        if not ready:
            st.caption("需要：任务池有文件 + 词表非空")

        st.markdown("#### 结果导出")
        res = ss.result
        if res is None or res.empty:
            st.caption("所有任务完成后可导出统计结果")
        else:
            st.dataframe(res.head(200), width="stretch", height=260, hide_index=True)
            st.caption(f"共 {len(res)} 条结果（预览前 200 行）· {core.MODES[ss.result_mode]} · "
                       f"词表 {len(ss.result_kw)} 个词")
            fmt = st.radio("导出格式", ["xlsx", "stata_csv", "csv", "dta", "py"], horizontal=True,
                           format_func=lambda f: {"xlsx": "Excel (XLSX)", "stata_csv": "Stata (CSV)",
                                                  "csv": "普通 CSV", "dta": "Stata (.dta)",
                                                  "py": "Python 脚本"}[f])
            fname = st.text_input("文件名", f"词频统计结果_{datetime.now():%Y%m%d_%H%M%S}")
            save_dir = st.text_input("保存路径", str(OUTPUT_DIR))
            include_txt = False
            if fmt == "py":
                include_txt = st.checkbox("把任务池里的原始文本一起打包（文件多时压缩包会很大）")
            if st.button("导出结果", type="primary", width="stretch"):
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                if fmt == "py":
                    out = Path(save_dir) / f"{fname}.zip"
                    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
                        z.write(core.BASE_DIR / "core.py", "core.py")
                        script = (core.BASE_DIR / "analysis.py").read_text(encoding="utf-8")
                        script = script.replace(
                            'DATA_DIRS = [str(core.DEFAULT_DATA_DIR)]',
                            'DATA_DIRS = ["reports"]' if include_txt else f'DATA_DIRS = [r"{core.DEFAULT_DATA_DIR}"]')
                        script = script.replace('ap.add_argument("--mode", default="word"',
                                                f'ap.add_argument("--mode", default="{ss.result_mode}"')
                        z.writestr("analysis.py", script)
                        z.writestr("keywords.txt", "\n".join(ss.result_kw))
                        z.writestr("requirements.txt", "jieba\npandas\nopenpyxl\n")
                        for f in core.RES_DIR.iterdir():
                            if f.is_file():
                                z.write(f, f"resources/{f.name}")
                        if include_txt:
                            for d in docs:
                                z.write(d.path, f"reports/{Path(d.path).name}")
                        z.writestr("README.txt",
                                   "pip install -r requirements.txt\npython analysis.py\n"
                                   "python analysis.py -o 我的结果.xlsx\n"
                                   "python analysis.py -f stata_csv -o 我的结果.csv\n")
                else:
                    ext = {"xlsx": "xlsx", "stata_csv": "csv", "csv": "csv", "dta": "dta"}[fmt]
                    out = Path(save_dir) / f"{fname}.{ext}"
                    core.export(res, str(out), fmt, ss.result_mode)
                st.success(f"已保存：{out}")
                if fmt == "stata_csv":
                    st.code(f'import delimited using "{out}", clear varnames(15) rowrange(16:)', language="stata")


# ================================================================ 词向量扩展
with tab_w2v:
    left, right = st.columns([1, 2.4], gap="large")
    with left:
        st.markdown("#### 模型管理")
        models = w2v.available_models()
        labels = {str(p): (("预训练模型 · " if p == w2v.DEFAULT_MODEL else "") + str(p.relative_to(w2v.MODEL_DIR)))
                  for p in models}
        options = list(labels) + ["__other__"]
        choice = st.selectbox("选择模型", options, format_func=lambda k: labels.get(k, "其他路径 …"))
        model_path = st.text_input("模型文件路径", "", placeholder=r"D:\...\word2vec.bin") \
            if choice == "__other__" else choice
        model = None
        if model_path and Path(model_path).exists():
            try:
                model = get_model(model_path, mtime(model_path))
                st.success(f"状态：已加载（{len(model)} 词，{model.vector_size} 维）")
            except Exception as e:
                st.error(f"加载失败：{e}")
        else:
            st.warning("状态：未加载。请从 GitHub Release 下载预训练模型放到 models 文件夹，或在下方自行训练。")

        with st.expander("自行训练 Word2Vec 模型"):
            src = st.radio("语料来源", ["pool", "folder"], horizontal=True,
                           format_func=lambda s: {"pool": "使用任务池文本", "folder": "本地 TXT 文件夹"}[s])
            if src == "pool":
                st.caption(f"任务池当前有 {len(ss.pool)} 个文件")
            train_dir = st.text_input("文件夹", str(core.DEFAULT_DATA_DIR), disabled=src == "pool")
            p1, p2 = st.columns(2)
            vec = p1.selectbox("向量维度", [100, 200, 300], index=1)
            sg = p2.selectbox("训练算法", [1, 0], format_func=lambda v: "Skip-gram（推荐）" if v else "CBOW（更快）")
            window = p1.number_input("窗口大小", 1, 20, 5)
            negative = p2.number_input("负采样数", 1, 30, 10)
            min_count = p1.number_input("最小词频", 1, 1000, 10)
            epochs = p2.number_input("训练轮次", 1, 50, 5)
            st.caption("分词资源（默认使用内置词典）")
            res_paths = {k: st.text_input(k, v) for k, v in core.default_resources().items()}
            out_dir = st.text_input("模型保存到文件夹", str(w2v.MODEL_DIR / "my_model"))
            tb1, tb2 = st.columns(2)
            if tb1.button("导出训练包", width="stretch"):
                cmd = (f"python w2v.py train --data \"{train_dir}\" --out models/my_model/word2vec.bin "
                       f"--vector-size {vec} --sg {sg} --window {window} --negative {negative} "
                       f"--min-count {min_count} --epochs {epochs}")
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for name in ("core.py", "w2v.py", "text_cleaning.py"):
                        z.write(core.BASE_DIR / name, name)
                    for k, v in res_paths.items():
                        if v and Path(v).exists():
                            z.write(v, f"resources/{Path(v).name}")
                    z.writestr("requirements.txt", "jieba\npandas\nnumpy\ngensim>=4.4\n")
                    z.writestr("train.bat", "pip install -r requirements.txt\r\n" + cmd + "\r\npause\r\n")
                    z.writestr("README.txt", "pip install -r requirements.txt\n" + cmd +
                               "\n训练完成后，把 models/my_model 里的 word2vec.bin 和 word2vec_vocab_meta.json "
                               "导入本软件即可。\n")
                st.download_button("下载训练包 ZIP", buf.getvalue(), "word2vec训练包.zip")
            if tb2.button("在本地训练", type="primary", width="stretch"):
                tdocs = pool_docs() if src == "pool" else core.discover([train_dir])
                if not tdocs:
                    st.error("没有语料")
                else:
                    out_bin = Path(out_dir) / "word2vec.bin"
                    corpus = Path(out_dir) / "corpus.txt"
                    bar = st.progress(0.0, "分词中 ...")
                    w2v.build_corpus(tdocs, corpus, res_paths,
                                     progress=lambda i, n: bar.progress(i / n * 0.3, f"分词 {i}/{n}"))
                    bar.progress(0.3, "训练第 1 轮 ...")
                    m = w2v.train(corpus, out_bin, vector_size=vec, sg=sg, window=window, negative=negative,
                                  min_count=min_count, epochs=epochs,
                                  on_epoch=lambda i, n: bar.progress(0.3 + 0.7 * i / n, f"已完成第 {i}/{n} 轮"))
                    corpus.unlink(missing_ok=True)
                    bar.progress(1.0, f"完成：{len(m.wv)} 词 → {out_bin}")
                    st.rerun()

        st.markdown("#### 种子词搜索")
        if "seed_text" not in ss:
            ss.seed_text = "、".join(ss.keywords)
        sb1, sb2 = st.columns(2)
        if sb1.button(f"使用当前词表（{len(ss.keywords)}）", width="stretch"):
            ss.seed_text = "、".join(ss.keywords)
        seed_file = sb2.file_uploader("导入种子词", type=["txt", "xlsx", "csv"], label_visibility="collapsed")
        if seed_file is not None and st.button("把文件里的词加入种子词"):
            merged = list(dict.fromkeys(core.split_keywords(ss.seed_text) +
                                        core.read_list_file(seed_file, seed_file.name)))
            ss.seed_text = "、".join(merged)
        st.text_area("种子词（支持多个，逗号、顿号、分号或换行分隔）", key="seed_text", height=90)
        st.markdown("#### 筛选")
        threshold = st.slider("相似度 ≥", 0.0, 1.0, 0.70, 0.01)
        topn = st.number_input("显示数量", 1, 500, 20)
        show_low = st.toggle("显示低置信结果", False)
        show_seeds = st.toggle("显示种子词结果", False)
        go = st.button("🔍 搜索", type="primary", width="stretch", disabled=model is None)

    with right:
        if go and model is not None:
            seeds = core.split_keywords(ss.seed_text)
            if not seeds:
                st.warning("请先输入种子词，或点「使用当前词表」")
            else:
                ss.w2v_result = (*w2v.expand(model, seeds, int(topn), threshold, show_low, show_seeds), seeds)
        if ss.w2v_result is None:
            st.info("加载模型后输入种子词，点「搜索」获得扩展词")
        else:
            df, info, seeds = ss.w2v_result
            st.markdown(f"#### 推荐词（{len(df)} 个）")
            st.caption(f"已隐藏低置信 {info['hidden_low']} 个 · 已隐藏种子词 {info['hidden_seed']} 个 · "
                       f"相似度达到阈值的共 {info['total']} 个")
            for err in info["errors"]:
                st.caption("⚠ " + err)
            if df.empty:
                st.warning("没有找到相似度高于阈值的词，试试降低阈值")
            else:
                view = df.copy()
                view.insert(0, "选择", False)
                view.insert(view.columns.get_loc("种子词"), "相似度分布", view["相似度"])
                edited = st.data_editor(
                    view, hide_index=True, width="stretch", height=560,
                    disabled=[c for c in view.columns if c != "选择"],
                    column_config={
                        "相似度": st.column_config.NumberColumn(format="%.4f", help="与种子词的余弦相似度"),
                        "相似度分布": st.column_config.ProgressColumn(min_value=0, max_value=1, format=" "),
                        "词频": st.column_config.NumberColumn(help="训练语料中出现次数"),
                        "rank": st.column_config.NumberColumn(help="词频排名（越小越常见）"),
                        "置信度": st.column_config.TextColumn(
                            help="低：低频(<20)/疑似序数/疑似人名机构名/低频且形态可疑；中：词频<100；其余为高"),
                    })
                chosen = edited.loc[edited["选择"], "推荐词"].tolist()
                a1, a2, a3, a4 = st.columns(4)
                if a1.button("全选加入词表", width="stretch"):
                    n = add_keywords(df["推荐词"].tolist())
                    st.success(f"已加入 {n} 个词，词表现有 {len(ss.keywords)} 个")
                if a2.button(f"加入词表（{len(chosen)}）", type="primary", width="stretch", disabled=not chosen):
                    n = add_keywords(chosen)
                    st.success(f"已加入 {n} 个词，词表现有 {len(ss.keywords)} 个")
                a3.download_button("导出 CSV", df.to_csv(index=False).encode("utf-8-sig"), "扩展词.csv",
                                   width="stretch")
                py = (
                    "# 近义词扩展脚本（本地版导出）；需要与 w2v.py、core.py 放在同一目录\n"
                    "import w2v\n"
                    f"MODEL = r\"{model_path}\"\n"
                    f"SEEDS = {seeds!r}\n"
                    f"m = w2v.load_model(MODEL)\n"
                    f"df, info = w2v.expand(m, SEEDS, topn={int(topn)}, min_sim={threshold}, "
                    f"show_low={show_low}, show_seeds={show_seeds})\n"
                    "print(df.to_string())\n"
                    "df.to_csv('扩展词.csv', index=False, encoding='utf-8-sig')\n")
                a4.download_button("导出 Python", py.encode("utf-8"), "word2vec_expand.py", width="stretch")

save_session()
