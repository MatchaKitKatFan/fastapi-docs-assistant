import gradio as gr

from rag import answer
from retriever import Retriever

retriever = Retriever("docs_1200")


def respond(question: str):
    r = answer(question, retriever.vector)
    sources = "\n".join(
        f"{n}. `{d['source']}` · {d['title']} (유사도 {d['score']:.2f})"
        for n, d in enumerate(r["sources"], start=1)
    )
    return r["answer"], sources or "검색된 문서 없음"


with gr.Blocks(title="FastAPI 문서 Q&A") as demo:
    gr.Markdown("## FastAPI 공식 문서 Q&A\n문서에 근거해 답하고 출처 번호를 붙입니다.")
    q = gr.Textbox(label="질문", placeholder="예) 의존성 주입에서 yield를 쓰면 정리 코드는 언제 실행되나요?")
    btn = gr.Button("질문하기", variant="primary")
    a = gr.Markdown(label="답변")
    s = gr.Markdown(label="출처")
    btn.click(respond, inputs=q, outputs=[a, s])
    q.submit(respond, inputs=q, outputs=[a, s])

if __name__ == "__main__":
    demo.launch()