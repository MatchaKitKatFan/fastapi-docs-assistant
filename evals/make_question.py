import json
import random

import chromadb
from pydantic import BaseModel

from llm import MODEL, client


class QA(BaseModel):
    question: str
    answer: str


class QASet(BaseModel):
    items: list[QA]


PROMPT = """아래 FastAPI 문서 조각을 읽고, 이 조각만 보면 답할 수 있는 질문 2개와 정답을 만들어라.
- 실제 개발자가 물어볼 법한 자연스러운 한국어 질문
- 문서 문장을 그대로 베끼지 말 것
- 정답은 1~3문장

<chunk>
{chunk}
</chunk>"""

col = chromadb.PersistentClient(path="chroma").get_collection("docs_1200")
data = col.get(include=["documents", "metadatas"])
random.seed(0)
picked = random.sample(range(len(data["ids"])), 30)

with open("evals/candidates.jsonl", "w", encoding="utf-8") as f:
    for i in picked:
        r = client.messages.parse(
            model=MODEL, max_tokens=2000,
            messages=[{"role": "user", "content": PROMPT.format(chunk=data["documents"][i])}],
            output_format=QASet,
        )
        for qa in r.parsed_output.items:
            f.write(json.dumps({
                "q": qa.question, "gold_sources": [data["metadatas"][i]["source"]],
                "gold_answer": qa.answer, "type": "fact", "origin": "generated",
            }, ensure_ascii=False) + "\n")
print("evals/candidates.jsonl 생성")