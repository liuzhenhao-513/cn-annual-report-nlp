"""Conservative PDF-to-text cleanup helpers for Word2Vec experiments."""

from collections import Counter
from dataclasses import asdict, dataclass
import re


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
CHINESE = re.compile(r"[\u4e00-\u9fff]")
ENGLISH_HYPHEN_WRAP = re.compile(r"([A-Za-z]{2,})-\s*\n\s*([A-Za-z]{2,})")
SPACED_CHINESE = re.compile(r"(?<![\u4e00-\u9fff])(?:[\u4e00-\u9fff][ \t]+){2,}[\u4e00-\u9fff](?![\u4e00-\u9fff])")
PAGE_NUMBER_LINE = re.compile(r"^\s*[-—–]?\s*(?:第\s*)?\d{1,4}\s*(?:页)?\s*[-—–]?\s*$", re.I)
PLACEHOLDER_LINE = re.compile(r"^\s*(?:[□√■●○◆◇※*·._…—–-]\s*){2,}$")
TOC_DOTTED_LINE = re.compile(r"[.。·…_]{5,}\s*\d{1,4}\s*$")
SENTENCE_END = re.compile(r"[。！？!?；;：:]$")
REPORT_HEADER = re.compile(r"(年度报告|半年度报告|季度报告|年报|annual report)", re.I)
PUNCT_TO_SPACE = re.compile(r"[，；：、\"“”'‘’《》【】（）()\[\]□√．－—·/]")
VALID_WORD = re.compile(r"^[\u4e00-\u9fff]{2,}$")


@dataclass
class CleaningStats:
    input_lines: int = 0
    output_lines: int = 0
    removed_blank_lines: int = 0
    removed_page_numbers: int = 0
    removed_placeholders: int = 0
    removed_toc_lines: int = 0
    removed_repeated_headers: int = 0
    removed_numeric_noise: int = 0
    repaired_spaced_chinese: int = 0
    repaired_english_hyphens: int = 0
    joined_wrapped_lines: int = 0

    def to_dict(self):
        return asdict(self)


def normalize_line(line):
    return re.sub(r"[ \t]+", " ", line).strip()


def repair_spaced_chinese(line):
    count = 0

    def replace(match):
        nonlocal count
        count += 1
        return re.sub(r"[ \t]+", "", match.group(0))

    return SPACED_CHINESE.sub(replace, line), count


def looks_numeric_noise(line):
    chinese_count = len(CHINESE.findall(line))
    digit_count = sum(char.isdigit() for char in line)
    numeric_groups = len(re.findall(r"\d[\d,.\-%]*", line))
    return (chinese_count == 0 and digit_count > 0) or (
        chinese_count <= 1 and digit_count >= 4 and numeric_groups >= 2
    )


def looks_tabular(line):
    chinese_count = len(CHINESE.findall(line))
    numeric_groups = len(re.findall(r"\d[\d,.\-%]*", line))
    return numeric_groups >= 2 or (numeric_groups >= 1 and chinese_count <= 4)


def should_join_wrapped(previous, current):
    if not previous or not current:
        return False
    if SENTENCE_END.search(previous):
        return False
    if looks_tabular(previous) or looks_tabular(current):
        return False
    previous_chinese = len(CHINESE.findall(previous))
    current_chinese = len(CHINESE.findall(current))
    return previous_chinese >= 18 and current_chinese >= 6


def repeated_report_headers(lines):
    normalized = [normalize_line(line) for line in lines if line.strip()]
    counts = Counter(line for line in normalized if len(line) <= 100 and REPORT_HEADER.search(line))
    return {line for line, count in counts.items() if count >= 3}


def clean_pdf_text(text):
    """Return conservative cleaned text plus cleanup counters."""
    stats = CleaningStats()
    text = CONTROL_CHARS.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    text, stats.repaired_english_hyphens = ENGLISH_HYPHEN_WRAP.subn(r"\1\2", text)
    raw_lines = text.split("\n")
    stats.input_lines = len(raw_lines)
    repeated_headers = repeated_report_headers(raw_lines)

    blocks = []
    current_block = []
    for raw_line in raw_lines:
        line = normalize_line(raw_line)
        if not line:
            stats.removed_blank_lines += 1
            if current_block:
                blocks.append(current_block)
                current_block = []
            continue
        if PAGE_NUMBER_LINE.match(line):
            stats.removed_page_numbers += 1
            continue
        if PLACEHOLDER_LINE.match(line):
            stats.removed_placeholders += 1
            continue
        if TOC_DOTTED_LINE.search(line):
            stats.removed_toc_lines += 1
            continue
        if line in repeated_headers:
            stats.removed_repeated_headers += 1
            continue
        if looks_numeric_noise(line):
            stats.removed_numeric_noise += 1
            continue

        line, repaired = repair_spaced_chinese(line)
        stats.repaired_spaced_chinese += repaired
        if current_block and should_join_wrapped(current_block[-1], line):
            current_block[-1] += line
            stats.joined_wrapped_lines += 1
        else:
            current_block.append(line)

    if current_block:
        blocks.append(current_block)
    output_lines = [line for block in blocks for line in block]
    stats.output_lines = len(output_lines)
    return "\n".join("\n".join(block) for block in blocks), stats


def legacy_sentences(text):
    """Reproduce the current production cleanup for comparison."""
    text = CONTROL_CHARS.sub("", text)
    text = PUNCT_TO_SPACE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return [part.strip() for part in re.split(r"[。！？\n]+", text) if len(part.strip()) >= 5]


def cleaned_sentences(text):
    cleaned, stats = clean_pdf_text(text)
    sentences = []
    for line in cleaned.splitlines():
        line = PUNCT_TO_SPACE.sub(" ", line)
        for part in re.split(r"[。！？!?]+", line):
            part = re.sub(r"\s+", " ", part).strip()
            if len(part) >= 5:
                sentences.append(part)
    return sentences, stats
