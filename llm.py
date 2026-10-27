import os

import anthropic
from dotenv import load_dotenv

load_dotenv()                     # .env 의 값을 환경변수로 읽음
client = anthropic.Anthropic()    # ANTHROPIC_API_KEY 환경변수를 자동으로 사용
MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")

# 100만 토큰당 달러 (입력, 출력). 2026-10 기준
PRICE = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
# 안전 분류기가 요청을 거절하면 다른 모델로 자동 재시도하는 서버 기능을 지원하는 모델
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}


def cost(usage) -> float:
    i, o = PRICE.get(MODEL, (0.0, 0.0))
    return (usage.input_tokens * i + usage.output_tokens * o) / 1_000_000


def ask(prompt: str, system: str = "", max_tokens: int = 4000, effort: str = "low"):
    """Claude에게 한 번 묻고 (답변 텍스트, usage)를 돌려준다."""
    kwargs = dict(
        model=MODEL,
        max_tokens=max_tokens,
        output_config={"effort": effort},   # low / medium / high: 생각을 얼마나 깊게 할지
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    if MODEL in FALLBACK_MODELS:
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
    resp = client.beta.messages.create(**kwargs)

    if resp.stop_reason == "refusal":
        raise RuntimeError("모델이 요청을 거절했습니다")
    if resp.stop_reason == "max_tokens":
        print("경고: max_tokens 에 걸려 답이 잘렸습니다")
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, resp.usage