import asyncio
import time

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel


OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"


class JudgeResult(BaseModel):
    verdict: bool
    reason: str


async def main():
    client = AsyncOpenAI(
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",
        timeout=120.0,
        max_retries=0,
    )

    patched_client = instructor.from_openai(
        client,
        mode=instructor.Mode.JSON,
    )

    start = time.perf_counter()

    result = await patched_client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are evaluating whether a statement is supported "
                    "by the supplied context."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Context: High-risk AI systems shall comply with the "
                    "requirements laid down in Section 2 of the EU AI Act.\n\n"
                    "Statement: High-risk AI systems must comply with "
                    "Section 2 requirements.\n\n"
                    "Return whether the statement is supported."
                ),
            },
        ],
        response_model=JudgeResult,
        temperature=0,
        max_tokens=128,
    )

    elapsed = time.perf_counter() - start

    print("Result:", result)
    print(f"Time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())