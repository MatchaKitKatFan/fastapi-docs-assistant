import argparse
import pathlib
import re

DOCS_DIR = pathlib.Path("data/fastapi/docs/en/docs")


def split_sections(text: str) -> list[str]:
    """헤딩(#~###) 기준으로 섹션을 나눈다. 코드 블록 안의 # 은 무시한다."""
    sections, cur, in_code = [], [], False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"#{1,3} ", line) and cur:
            sections.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur:
        sections.append("\n".join(cur))
    return sections


def split_markdown(text: str, source: str, max_chars: int = 1200, overlap: int = 200) -> list[dict]:
    chunks = []
    for sec in split_sections(text):
        if not sec.strip():
            continue
        title = sec.strip().splitlines()[0].lstrip("#").strip()
        for start in range(0, len(sec), max_chars - overlap):
            piece = sec[start:start + max_chars].strip()
            if len(piece) > 50:                      # 너무 짧은 조각은 버림
                chunks.append({"text": piece, "source": source, "title": title})
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--collection", default="docs_1200")
    args = ap.parse_args()

    # 무거운 라이브러리는 여기서 import (테스트가 가볍게 돌도록)
    import chromadb
    from sentence_transformers import SentenceTransformer

    chunks = []
    for path in sorted(DOCS_DIR.rglob("*.md")):
        rel = path.relative_to(DOCS_DIR).as_posix()
        chunks += split_markdown(path.read_text(encoding="utf-8"), rel, args.chunk, args.overlap)
    print(f"청크 {len(chunks)}개")

    embedder = SentenceTransformer("BAAI/bge-m3")   # 처음 실행 시 약 2GB 다운로드
    embedder.max_seq_length = 1024
    vecs = embedder.encode([c["text"] for c in chunks], batch_size=16,
                           normalize_embeddings=True, show_progress_bar=True)

    db = chromadb.PersistentClient(path="chroma")
    try:
        db.delete_collection(args.collection)          # 같은 이름이 있으면 새로 만듦
    except Exception:
        pass
    col = db.create_collection(args.collection, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), 1000):
        part = chunks[i:i + 1000]
        col.add(
            ids=[f"c{j}" for j in range(i, i + len(part))],
            documents=[c["text"] for c in part],
            embeddings=vecs[i:i + 1000].tolist(),
            metadatas=[{"source": c["source"], "title": c["title"]} for c in part],
        )
    print(f"'{args.collection}' 저장 완료")


if __name__ == "__main__":
    main()