import argparse
import json
import pathlib
import time

from pydantic import BaseModel

from evals.eval_retrieval import load
from llm import MODEL, client, cost
from rag import NO_ANSWER, answer
from retriever import Retriever


class Verdict(BaseModel):
    faithful: bool   # 답변의 모든 주장이 문서에 근거하는가
    correct: bool    # 정답과 핵심 내용이 같은가
    reason: str


JUDGE = """너는 RAG 시스템 채점자다.
faithful: 모델 답변의 모든 주장이 <documents>에 근거하면 true, 문서에 없는 내용을 지어냈으면 false.
correct: 모델 답변이 정답과 핵심 내용이 같으면 true. 표현이 달라도 의미가 같으면 true.
reason: 판단 이유를 한 문장으로."""


def judge(ex: dict, result: dict) -> Verdict:
    ctx = "\n\n".join(d["text"] for d in result["sources"])
    r = client.messages.parse(
        model=MODEL, max_tokens=2000, system=JUDGE,
        messages=[{"role": "user", "content":
                   f"<documents>\n{ctx}\n</documents>\n\n질문: {ex['q']}\n"
                   f"정답: {ex['gold_answer']}\n모델 답변: {result['answer']}"}],
        output_format=Verdict,
    )
    return r.parsed_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="docs_1200")
    ap.add_argument("--method", default="vector")
    ap.add_argument("--name", required=True, help="결과 파일 이름. 예) v0-vector")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만 (비용 절약용)")
    ap.add_argument("--min-score", type=float, default=None)
    args = ap.parse_args()

    search = getattr(Retriever(args.collection), args.method)
    data = load()[: args.limit] if args.limit else load()
    out = pathlib.Path(f"evals/results/{args.name}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    s = dict(answerable=0, faithful=0, correct=0, unanswerable=0, abstained=0, cost=0.0, latency=0.0)
    with out.open("w", encoding="utf-8") as f:
        for ex in data:
            t = time.perf_counter()
            res = answer(ex["q"], search, min_score=args.min_score)
            latency = time.perf_counter() - t
            rec = {"q": ex["q"], "type": ex.get("type"), "answer": res["answer"],
                   "sources": [d["source"] for d in res["sources"]], "latency": round(latency, 2)}
            s["latency"] += latency
            if res["usage"]:
                s["cost"] += cost(res["usage"])

            if ex["gold_sources"]:
                s["answerable"] += 1
                if res["sources"]:
                    v = judge(ex, res)
                else:
                    v = Verdict(faithful=True, correct=False, reason="임계값에 걸려 답하지 않음")
                s["faithful"] += v.faithful
                s["correct"] += v.correct
                rec.update(v.model_dump())
            else:
                s["unanswerable"] += 1
                rec["abstained"] = NO_ANSWER in res["answer"]
                s["abstained"] += rec["abstained"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    a, u, n = max(s["answerable"], 1), max(s["unanswerable"], 1), max(len(data), 1)
    print(f"[{args.name}] 충실도 {s['faithful']/a:.2f} | 정답률 {s['correct']/a:.2f} | "
          f"무응답 정확도 {s['abstained']/u:.2f} | 평균 지연 {s['latency']/n:.1f}s | "
          f"답변 비용 ${s['cost']:.3f} (채점 비용 별도)")


if __name__ == "__main__":
    main()