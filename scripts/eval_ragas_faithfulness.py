import sys
import types
import asyncio
import time

# ============================================================
# Ragas 0.4.3 compatibility patch
# ============================================================

dummy_vertex = types.ModuleType(
    "langchain_community.chat_models.vertexai"
)

dummy_vertex.ChatVertexAI = type(
    "ChatVertexAI",
    (object,),
    {}
)

sys.modules[
    "langchain_community.chat_models.vertexai"
] = dummy_vertex


# ============================================================
# Imports
# ============================================================

from ollama import AsyncClient
from ragas.metrics import Faithfulness
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.dataset_schema import SingleTurnSample


# ============================================================
# Configuration
# ============================================================

OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_HOST = "http://localhost:11434"


# ============================================================
# Native Ollama → Ragas adapter
# ============================================================

class OllamaRagasLLM(InstructorBaseRagasLLM):

    def __init__(self, model: str, host: str):
        self.model = model
        self.client = AsyncClient(host=host)
        self.is_async = True

    async def agenerate(self, prompt: str, response_model):
        response = await self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            format=response_model.model_json_schema(),
            options={
                "temperature": 0,
            },
        )

        content = response.message.content

        return response_model.model_validate_json(content)

    def generate(self, prompt: str, response_model):
        return asyncio.run(
            self.agenerate(prompt, response_model)
        )

# ============================================================
# Direct Faithfulness test
# ============================================================

async def main():

    print("Initializing native Ollama Ragas LLM...")

    llm = OllamaRagasLLM(
        model=OLLAMA_MODEL,
        host=OLLAMA_HOST,
    )

    sample = SingleTurnSample(
        user_input=(
            "What are the compliance requirements for "
            "high-risk AI systems?"
        ),

        response=(
            "High-risk AI systems must comply with the "
            "requirements set out in Section 2 of the EU AI Act. "
            "Providers must implement a risk management system "
            "and complete the relevant conformity assessment "
            "procedure."
        ),

        retrieved_contexts=[
            (
                "High-risk AI systems shall comply with the "
                "requirements laid down in this Section, taking "
                "into account their intended purpose."
            ),
            (
                "A risk management system shall be established, "
                "implemented, documented and maintained in "
                "relation to high-risk AI systems."
            ),
            (
                "Providers of high-risk AI systems shall ensure "
                "that their high-risk AI systems are compliant "
                "with the requirements set out in Section 2."
            ),
        ],
    )

    metric = Faithfulness(
        llm=llm
    )

    print("Running direct Ragas Faithfulness test...")

    start = time.perf_counter()

    score = await metric.single_turn_ascore(
        sample
    )

    elapsed = time.perf_counter() - start

    print()
    print("==============================")
    print("FAITHFULNESS TEST")
    print("==============================")
    print("Score:", score)
    print(f"Time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())