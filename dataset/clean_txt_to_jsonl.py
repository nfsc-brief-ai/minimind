#!/usr/bin/env python3
"""
Clean messy/garbled Chinese TXT files → JSONL training data.

Usage:
    python clean_txt_to_jsonl.py input1.txt input2.txt ... -o output.jsonl
    python clean_txt_to_jsonl.py corpus_dir/ -o output.jsonl

Output format: {"text": "..."} one per line, suitable for MiniMind pretrain.
"""

import re
import json
import argparse
import sys
from pathlib import Path


# ──────────────────────────────────────────────
# Encoding detection & repair
# ──────────────────────────────────────────────

CANDIDATE_ENCODINGS = ['utf-8', 'gbk', 'gb18030', 'big5', 'utf-16', 'latin-1']

def detect_and_decode(raw: bytes) -> str:
    """Try encodings in order; fall back to chardet if available."""
    # 1. Try candidate encodings
    for enc in CANDIDATE_ENCODINGS:
        try:
            text = raw.decode(enc)
            # Reject if it looks like mojibake (too many replacement chars or Latin1 garbage)
            if enc == 'utf-8' and '�' not in text:
                return text
            if enc != 'utf-8' and enc != 'latin-1':
                return text
        except (UnicodeDecodeError, LookupError):
            continue

    # 2. Try chardet
    try:
        import chardet
        result = chardet.detect(raw)
        if result['encoding'] and result['confidence'] > 0.7:
            return raw.decode(result['encoding'], errors='replace')
    except ImportError:
        pass

    # 3. Try the classic mojibake fix: was written as GBK, read as latin-1
    try:
        return raw.decode('latin-1').encode('latin-1').decode('gbk')
    except Exception:
        pass

    # 4. Last resort
    return raw.decode('utf-8', errors='replace')


def fix_mojibake(text: str) -> str:
    """Attempt to fix text that was decoded with wrong codec."""
    # Common case: UTF-8 Chinese stored, then read as latin-1 → re-encode as latin-1, decode as utf-8
    try:
        fixed = text.encode('latin-1').decode('utf-8')
        chinese_count = sum(1 for c in fixed if '一' <= c <= '鿿')
        orig_count = sum(1 for c in text if '一' <= c <= '鿿')
        if chinese_count > orig_count:
            return fixed
    except Exception:
        pass
    return text


# ──────────────────────────────────────────────
# Text cleaning
# ──────────────────────────────────────────────

# Page number patterns: "第1页", "- 3 -", "Page 5", plain numbers on their own line
RE_PAGE_NUM = re.compile(
    r'^\s*(-\s*)?\d+(\s*-\s*)?$'
    r'|^\s*第\s*\d+\s*[页码]\s*$'
    r'|^\s*[Pp]age\s*\d+\s*$',
    re.MULTILINE
)

# Header/footer repeated separators
RE_SEPARATOR = re.compile(r'^[─—=\-_＝]{4,}\s*$', re.MULTILINE)

# Legal document line numbers: "　　1." "   （一）" at line start
RE_LINE_NUM = re.compile(r'^\s*[（(]?\s*[一二三四五六七八九十百千\d]+\s*[）)、．.]\s*', re.MULTILINE)

# Broadcast transcript timestamps: "Jun. 12, 2021  " or "2021-06-12 12:34"
RE_TIMESTAMP = re.compile(
    r'^[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}\s*$'
    r'|\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(:\d{2})?',
    re.MULTILINE
)

# File header separators like ===...===【filename.txt】
RE_FILE_HEADER = re.compile(r'={10,}【[^】]*】', re.MULTILINE)

# PDF page markers: "第1 / 102页" or "Page 5 of 102"
RE_PDF_PAGE = re.compile(r'第\s*\d+\s*/\s*\d+\s*页|[Pp]age\s*\d+\s*(of\s*\d+)?')

# Multiple blank lines → single blank line
RE_MULTI_BLANK = re.compile(r'\n{3,}')

# Trailing whitespace per line
RE_TRAILING = re.compile(r'[ \t]+$', re.MULTILINE)

# Annotation brackets with case/exhibit numbers: 【证据1】【附件A】
RE_ANNOTATION = re.compile(r'[【\[](证据|附件|注释?|脚注|见附|参见)[^】\]]{0,30}[】\]]')

# Common header garbage in legal docs
RE_LEGAL_HEADER = re.compile(
    r'(案\s*号|文\s*号|发文字号|密\s*级|紧急程度|主\s*送|抄\s*送|联系人|联系电话)[：:][^\n]*\n?',
    re.MULTILINE
)


def clean_text(text: str, strip_line_numbers: bool = False) -> str:
    text = fix_mojibake(text)
    text = RE_FILE_HEADER.sub('\n\n', text)
    text = RE_PDF_PAGE.sub('', text)
    text = RE_PAGE_NUM.sub('', text)
    text = RE_SEPARATOR.sub('', text)
    text = RE_TIMESTAMP.sub('', text)
    text = RE_LEGAL_HEADER.sub('', text)
    text = RE_ANNOTATION.sub('', text)
    if strip_line_numbers:
        text = RE_LINE_NUM.sub('', text)
    text = RE_TRAILING.sub('', text)
    text = RE_MULTI_BLANK.sub('\n\n', text)
    return text.strip()


# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────

def is_chinese_enough(text: str, min_ratio: float = 0.3) -> bool:
    """Reject chunks with fewer than min_ratio CJK characters (filters garbled/English content)."""
    if not text:
        return False
    cjk = sum(1 for c in text if '一' <= c <= '鿿' or '㐀' <= c <= '䶿')
    return cjk / len(text) >= min_ratio


def split_into_chunks(text: str, min_chars: int = 50, max_chars: int = 800) -> list[str]:
    """
    Split on paragraph boundaries first; if text is a single long line
    (common with PDF/web extracts), fall back to sentence-level splitting.
    """
    # Normalize: treat single newlines as spaces if text is mostly one block
    lines = [l for l in text.split('\n') if l.strip()]
    if len(lines) <= 3:
        # Flat file (entire doc is one or few lines, spaces used as sentence separators)
        text = ' '.join(lines)
        # First try splitting by sentence-ending punctuation
        by_punct = [s.strip() for s in re.split(r'(?<=[。！？!?…])\s*', text) if s.strip()]
        if len(by_punct) > 5:
            sentences = by_punct
        else:
            # Fall back: split by space between CJK character blocks
            segments = re.split(r'(?<=[^\x00-\x7F])\s+(?=[^\x00-\x7F\s])', text)
            sentences = [s.strip() for s in segments if s.strip()]
    else:
        # Has structure — split by paragraphs, then by sentence if too long
        paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
        # Flatten single-newline paragraphs that are very long
        sentences = []
        for para in paragraphs:
            if len(para) > max_chars:
                sentences.extend(re.split(r'(?<=[。！？!?…])\s*', para))
            else:
                sentences.append(para)

    chunks = []
    buf = ''
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(buf) + len(sent) > max_chars and len(buf) >= min_chars:
            chunks.append(buf.strip())
            buf = sent
        else:
            buf = (buf + sent) if buf else sent

    if buf.strip():
        chunks.append(buf.strip())

    # Hard cap: split any chunk still over max_chars*2 by brute force
    final = []
    for c in chunks:
        if len(c) > max_chars * 2:
            for i in range(0, len(c), max_chars):
                piece = c[i:i + max_chars].strip()
                if len(piece) >= min_chars:
                    final.append(piece)
        elif len(c) >= min_chars:
            final.append(c)
    return final


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def collect_txt_files(paths: list[str]) -> list[Path]:
    files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.rglob('*.txt')))
        elif path.is_file():
            files.append(path)
        else:
            print(f'[warn] not found: {p}', file=sys.stderr)
    return files


def process_file(path: Path, min_chars: int, max_chars: int, strip_line_numbers: bool) -> list[str]:
    raw = path.read_bytes()
    text = detect_and_decode(raw)
    text = clean_text(text, strip_line_numbers=strip_line_numbers)
    return split_into_chunks(text, min_chars=min_chars, max_chars=max_chars)


def main():
    parser = argparse.ArgumentParser(description='Clean Chinese TXT files → JSONL for MiniMind pretrain')
    parser.add_argument('inputs', nargs='+', help='TXT files or directories')
    parser.add_argument('-o', '--output', default='pretrain_corpus.jsonl', help='Output JSONL path')
    parser.add_argument('--min-chars', type=int, default=50, help='Min chars per training sample')
    parser.add_argument('--max-chars', type=int, default=800, help='Max chars per training sample')
    parser.add_argument('--strip-line-numbers', action='store_true',
                        help='Strip leading legal numbering like (一) 1. etc.')
    parser.add_argument('--preview', action='store_true',
                        help='Print first 5 samples and exit (for sanity check)')
    args = parser.parse_args()

    files = collect_txt_files(args.inputs)
    if not files:
        print('No TXT files found.', file=sys.stderr)
        sys.exit(1)

    total = 0
    out_path = Path(args.output)

    with out_path.open('w', encoding='utf-8') as fout:
        for path in files:
            print(f'Processing: {path}')
            try:
                chunks = process_file(path, args.min_chars, args.max_chars, args.strip_line_numbers)
            except Exception as e:
                print(f'  [error] {e}', file=sys.stderr)
                continue

            if args.preview:
                print(f'\n=== {path.name} — {len(chunks)} chunks ===')
                for i, c in enumerate(chunks[:5]):
                    print(f'\n--- chunk {i+1} ---\n{c}')
                continue

            kept = [c for c in chunks if is_chinese_enough(c)]
            for chunk in kept:
                fout.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')
            skipped = len(chunks) - len(kept)
            total += len(kept)
            print(f'  → {len(kept)} samples' + (f' (丢弃{skipped}个非中文块)' if skipped else ''))

    if not args.preview:
        print(f'\nDone. {total} total samples → {out_path}')


if __name__ == '__main__':
    main()
