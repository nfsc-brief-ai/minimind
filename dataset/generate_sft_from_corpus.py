#!/usr/bin/env python3
"""
从 pretrain_corpus.jsonl 里的语料块，用 Claude API 自动生成 SFT 问答对。
输出格式兼容 MiniMind SFTDataset。

用法：
    python dataset/generate_sft_from_corpus.py \
        --input dataset/pretrain_corpus.jsonl \
        --output dataset/sft_politics.jsonl \
        --samples 800
"""

import json
import random
import time
import argparse
from pathlib import Path
import anthropic

SYSTEM_PROMPT = """你是一个专业的中文训练数据生成助手。
我会给你一段中文文本，请根据文本内容生成2到3个高质量的问答对。

要求：
- 问题要自然，像真实用户会问的
- 回答要忠实于原文，不要编造原文没有的内容
- 问题和回答都用中文
- 回答长度适中（50-200字）
- 输出 JSON 数组格式，每个元素包含 "q" 和 "a" 字段

只输出 JSON，不要其他说明文字。

示例输出：
[
  {"q": "郭文贵案的主要罪名是什么？", "a": "郭文贵被裁定十项重罪成立，包括敲诈勒索、证券欺诈等。"},
  {"q": "新中国联邦的核心理念是什么？", "a": "新中国联邦主张建立法治、民主和自由的新中国，推行三权分立体制。"}
]"""


def generate_qa_pairs(client: anthropic.Anthropic, text: str) -> list[dict]:
    """Call Claude to generate Q&A pairs from a text chunk."""
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            messages=[
                {
                    "role": "user",
                    "content": f"请根据以下文本生成问答对：\n\n{text}"
                }
            ],
            system=SYSTEM_PROMPT,
        )
        content = response.content[0].text.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        pairs = json.loads(content)
        return [(p["q"], p["a"]) for p in pairs if "q" in p and "a" in p]
    except Exception as e:
        print(f"  [跳过] {e}")
        return []


def to_sft_record(q: str, a: str) -> dict:
    """Convert Q&A pair to MiniMind SFT format."""
    return {
        "conversations": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/pretrain_corpus.jsonl")
    parser.add_argument("--output", default="dataset/sft_politics.jsonl")
    parser.add_argument("--samples", type=int, default=800, help="从语料中随机抽取多少块")
    parser.add_argument("--min-chars", type=int, default=150, help="跳过太短的块")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    client = anthropic.Anthropic()  # 读取 ANTHROPIC_API_KEY 环境变量

    # 读取语料
    print(f"读取语料: {args.input}")
    chunks = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            if len(text) >= args.min_chars:
                chunks.append(text)
    print(f"符合条件的块: {len(chunks)}，随机抽取 {args.samples} 个")

    random.seed(args.seed)
    selected = random.sample(chunks, min(args.samples, len(chunks)))

    out_path = Path(args.output)
    total = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for i, chunk in enumerate(selected, 1):
            print(f"[{i}/{len(selected)}] 生成中...", end=" ", flush=True)
            pairs = generate_qa_pairs(client, chunk[:600])  # 最多600字送给API
            for q, a in pairs:
                fout.write(json.dumps(to_sft_record(q, a), ensure_ascii=False) + "\n")
            total += len(pairs)
            print(f"生成 {len(pairs)} 对（累计 {total}）")
            time.sleep(0.1)  # 避免触发速率限制

    print(f"\n完成！共 {total} 条问答对 → {out_path}")


if __name__ == "__main__":
    main()
