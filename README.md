# cn-annual-report-nlp · 中文上市公司年报文本分析工具

> 作者：[liuzhenhao-513](https://github.com/liuzhenhao-513) · © 2026 保留所有权利

本项目是一个**在本机运行的中文上市公司年报文本分析工具**，可以分析**年报全文**，也可以只分析其中的**管理层讨论与分析（MD&A）**。
你提供一批年报文本和一份关键词表，它会计算每家公司每年的关键词词频、句频和 TF-IDF，并导出可以直接进回归的 Excel / Stata 数据。
另外还提供基于 Word2Vec 的**关键词扩展**，帮你从几个种子词出发找到语义相近的词，构建更完整的词库。

常见用途：用年报或 MD&A 文本构造企业层面的指标，比如数字化转型、ESG / 绿色发展、创新、风险披露、管理层语调等。

- 全部在本机运行，不联网，数据不出电脑
- 网页界面操作，不用写代码；也提供命令行脚本，方便复现
- 内置 2001–2025 年 A 股公司行业库，可以按行业筛选样本

---

## 功能

| 模块 | 说明 |
|---|---|
| 样本选择 | 输入股票代码、导入代码文件，或填 `all` 选全部公司；可选年份范围；**按行业筛选**（353 个行业，可多选）；剔除 ST 与 B 股；新旧代码（如北交所 830xxx ↔ 920xxx）自动对应 |
| 文本导入 | 年报全文或 MD&A 的 TXT 文件。可以填文件夹路径（含子文件夹），也可以在网页上传 TXT / ZIP。文件名为 `股票代码_年份.txt` 时，自动识别代码、年份和行业 |
| 词表管理 | 手动添加、导入 TXT / XLSX、导出、清空；词向量扩展出的词可以一键加入 |
| 文本统计 | **词频模式**（关键词次数 ÷ 分词总数）或**句频模式**（含关键词的句子数 ÷ 总句数）；每个词都输出 `count`、`ratio`、`tfidf` 三列 |
| 断点续跑 | 多进程并行；可以随时停止，再次开始时从断点继续；关掉程序后任务池、词表和结果都会保留 |
| 结果导出 | Excel、Stata (CSV，带导入命令)、普通 CSV、Stata `.dta`、Python 脚本包（离开本工具也能复现） |
| 关键词扩展 | 输入种子词，返回语义相近的词，附带相似度、词频、rank、置信度、命中的种子词；可以勾选后加入词表 |
| 训练词向量 | 用自己的语料训练 Word2Vec，参数都能调；训练出的模型可以直接在界面里切换使用 |

---

## 快速开始

### 1. 环境

- Windows 10 / 11（macOS / Linux 也能跑，启动方式见下文）
- Python 3.10 及以上（开发环境是 3.12），安装时勾选 “Add Python to PATH”

### 2. 下载代码

```bash
git clone https://github.com/liuzhenhao-513/cn-annual-report-nlp.git
```

也可以在页面上点 **Code → Download ZIP**。

### 3. 下载预训练词向量模型

到 [Releases](https://github.com/liuzhenhao-513/cn-annual-report-nlp/releases/latest) 下载下面两个文件，放进 `models/` 文件夹：

```
models/word2vec.bin
models/word2vec_vocab_meta.json
```

这一步只有“关键词扩展”需要。只做词频统计的话，可以跳过。

### 4. 准备年报文本

把年报全文或 MD&A 的 TXT 文件放进 `data/管理层讨论与分析/`，子文件夹随意（比如按年份分）；也可以放在任意位置，到界面里填路径。文件命名：

```
000001_2023.txt      ← 股票代码_年份.txt，UTF-8 或 GBK 编码都可以
600000_2022.txt
```

本仓库**不包含年报文本**，原因是体积大（全部 A 股 MD&A 约 3.6GB），而且年报版权归各上市公司。
可以从巨潮资讯网等披露平台下载 PDF，转成文本后再用。

### 5. 启动

**Windows**：双击 `启动.bat`。第一次运行会自动创建虚拟环境、安装依赖，需要几分钟。之后浏览器会自动打开 http://localhost:8501 。关掉黑色窗口，程序就退出了。

**macOS / Linux**：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 使用流程

1. **主工作台 → 公司检索与导入**：填股票代码（或 `all`）、年份范围，选行业 → 「添加到任务池」。
   如果分析的是自己整理的文本，切到「自定义文本导入」，填文件夹或上传文件。
2. **词表管理**：添加关键词，比如 `数字化转型、大数据、人工智能、云计算`。
3. **（可选）词向量扩展**：切到「词向量扩展」页，点「使用当前词表」→「搜索」，勾选合适的词 →「加入词表」。
4. **统计模式**：选词频模式或句频模式 →「开始任务」。
5. **结果导出**：选格式 →「导出结果」。用 Stata (CSV) 导出时，界面会给出导入命令：
   ```stata
   import delimited using "结果.csv", clear varnames(15) rowrange(16:)
   ```

导出列：`代码, 行业, 年份, 分母, {词}_count…, {词}_ratio…, {词}_tfidf…`

---

## 方法概要

### 文本统计（`core.py`）

| 项目 | 设置 |
|---|---|
| 分词 | jieba 精确模式，关闭 HMM（`jieba.cut(text, HMM=False)`） |
| 领域词典 | `resources/custom_dict.txt`：1,404 个财会、数字化、ESG 术语；`resources/high_freq_dict.txt`：同一批词，词频设为 2,000,000，强制作为整词切分 |
| 停用词 / 碎片词 | `resources/stopwords.txt`（313 个），`resources/bad_tokens.txt`（47 个错误切分碎片） |
| 词频模式 | 分母 = 去掉空白、停用词、碎片词后的词数。关键词作为完整的词出现时，数分词结果里的次数；否则数它在原文里出现的次数 |
| 句频模式 | 按 `。！？…` 切句。分母 = 句子数，分子 = 含关键词的句子数 |
| TF-IDF | `tfidf = ratio × [ln((N+1)/(df+1)) + 1]`，N 为当次导出的文档数，df 为出现过该词的文档数 |

### 关键词扩展（`w2v.py`）

- 模型：Word2Vec（Skip-gram + 负采样），用 gensim 训练，按余弦相似度检索。
- 种子词不在词表里时，用它分词后各子词向量之和作为查询向量。
- 过滤：只保留 ≥2 个汉字的词，并剔除序数词（如“第三次”）、粘连碎片（如“X为、X的”）、错误切分碎片。
- 置信度：低频（<20 次）、疑似序数、疑似人名或机构名的标为“低”，默认隐藏；词频 <100 的标为“中”；其余为“高”。
- 多个种子词时，按“最佳排名 + 覆盖的种子词数 + 词频”综合排序。

> 建议：词向量只用来**辅助发现候选词**，最终词表要人工逐个审核。

### 预训练模型 `annual-report-v1`

| 参数 | 取值 | 说明 |
|---|---|---|
| 训练语料 | 2001–2025 年 A 股上市公司年报全文 | 约 15.0 亿词次 |
| 向量维度 | 200 | 从模型文件核实 |
| 词表大小 | 111,944 | 从模型文件核实 |
| 最小词频 min_count | 10 | 从模型文件核实 |
| 词形 | 只保留 ≥2 个汉字的中文词 | 从模型文件核实 |
| 算法 / window / negative / epochs | Skip-gram / 5 / 10 / 5 | 训练脚本默认值。训练日志没有随模型保存 |

在论文里报告参数时，如果需要每一项都有记录可查，建议用「训练词向量」功能自己重新训练，并保存训练命令和日志。

### 训练自己的模型

界面：「词向量扩展 → 自行训练 Word2Vec 模型」，或者用命令行：

```bash
python w2v.py train --data data/管理层讨论与分析 --out models/my_model/word2vec.bin \
    --vector-size 200 --sg 1 --window 5 --negative 10 --min-count 10 --epochs 5
```

训练前的语料清洗在 `text_cleaning.py`：去页码、目录行、重复页眉、纯数字行，合并断行，按句切分。
参考耗时（22 核 CPU，2001–2025 年全部 MD&A）：分词约 8 分钟，训练约 60 分钟。

**要不要重新训练？** 词频、句频、TF-IDF 统计不依赖模型，任何中文文本都能直接分析。
只有关键词扩展用到模型：年报全文和 MD&A 可以直接用预训练模型；政府工作报告这类风格差别大的文本，建议用同类文本重新训练。
目前只支持中文文本。

---

## 命令行

```bash
# 批量统计：关键词读 keywords.txt（每行一个）
python analysis.py -k 词表.txt --year-from 2010 --year-to 2025 --mode word -f stata_csv -o 结果.csv
python analysis.py --industry 货币金融服务 --exclude-st -o 银行.xlsx

# 关键词扩展
python w2v.py expand 大数据 人工智能 云计算 --topn 30 --threshold 0.7
```

---

## 项目结构

```
cn-annual-report-nlp/
├── 启动.bat              Windows 一键启动（首次自动安装依赖）
├── app.py               启动入口
├── ui.py                网页界面（Streamlit）
├── core.py              文本统计：分词、词频/句频、TF-IDF、导出、行业库
├── w2v.py               词向量：模型读取、关键词扩展、训练
├── text_cleaning.py     训练语料清洗
├── analysis.py          命令行批量统计
├── requirements.txt
├── resources/           词典、停用词、碎片词、行业库 metadata.db
├── models/              放预训练模型（从 Releases 下载）
└── data/                放年报 / MD&A 文本（不随仓库提供）
```

运行时还会自动生成 `state/`（断点缓存）、`uploads/`（网页上传的文件）、`output/`（导出结果），这三个文件夹不提交到仓库。

---

## 版权声明

© 2026 liuzhenhao-513，保留所有权利。

本项目的源代码、词典与资源文件、行业库和预训练模型，未经作者书面许可，不得复制、修改、分发或用于商业用途。
个人学习和学术研究需要使用时，请联系作者获得许可；在论文中使用时，请注明出处。

本项目使用的第三方库按各自的许可证使用：[jieba](https://github.com/fxsjy/jieba)（MIT）、[gensim](https://github.com/piskvorky/gensim)（LGPL-2.1）、[Streamlit](https://github.com/streamlit/streamlit)（Apache-2.0）、[pandas](https://github.com/pandas-dev/pandas)（BSD-3-Clause）。
年报文本的版权归各上市公司所有。
