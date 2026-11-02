import argparse
import json
import sys

from retriever import Retriever


def load(path: str = "evals/qa.jsonl") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(search, data: list[dict], k: int = 5) -> dict:
    hits, rr = 0, 0.0
    for ex in data:
        got = [d["source"] for d in search(ex["q"], k)]
        ranks = [i for i, s in enumerate(got, start=1) if s in ex["gold_sources"]]
        if ranks:
            hits += 1
            rr += 1 / ranks[0]          # 정답이 1위면 1, 2위면 0.5 ...
    n = len(data)
    return {"recall": hits / n, "mrr": rr / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="docs_1200")
    ap.add_argument("--method", default="vector", choices=["vector", "keyword", "hybrid", "rerank"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-recall", type=float, default=0.0)
    args = ap.parse_args()

    data = [ex for ex in load() if ex["gold_sources"]]   # 답 있는 질문만
    r = Retriever(args.collection)
    m = evaluate(getattr(r, args.method), data, args.k)
    print(f"{args.collection} / {args.method} / k={args.k}  "
          f"Recall@{args.k}={m['recall']:.3f}  MRR={m['mrr']:.3f}")
    if m["recall"] < args.min_recall:
        sys.exit(f"Recall {m['recall']:.3f} 이 기준 {args.min_recall} 보다 낮음")


if __name__ == "__main__":
    main()