from fastapi import FastAPI
from pydantic import BaseModel

from rag import answer
from retriever import Retriever

app = FastAPI(title="FastAPI Docs Assistant")
retriever = Retriever("docs_1200")


class AskReq(BaseModel):
    question: str
    k: int = 5


class Source(BaseModel):
    n: int
    source: str
    title: str
    score: float


class AskRes(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask", response_model=AskRes)
def ask_api(req: AskReq):
    r = answer(req.question, retriever.vector, req.k)
    return {
        "answer": r["answer"],
        "sources": [{"n": n, "source": d["source"], "title": d["title"], "score": round(d["score"], 3)}
                    for n, d in enumerate(r["sources"], start=1)],
    }