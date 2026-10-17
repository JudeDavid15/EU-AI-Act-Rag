import sys
import types
import asyncio
import multiprocessing

import pandas as pd
import requests
from datasets import Dataset
from ollama import AsyncClient


multiprocessing.set_executable(sys.executable)

print("Python:", sys.executable)



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



from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.llms.base import InstructorBaseRagasLLM

from ragas.metrics import (
    Faithfulness,
    LLMContextRecall,
    AnswerRelevancy,
)

from langchain_huggingface import HuggingFaceEmbeddings


FASTAPI_ENDPOINT = "http://localhost:8000/query"
OLLAMA_HOST = "http://localhost:11434"

OLLAMA_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"



embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
)

from data.eval_dataset import EVAL_DATASET

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
            self.agenerate(
                prompt,
                response_model,
            )
        )

def fetch_rag_responses(test_cases):

    records = []
    print("Gathering live responses from your RAG pipeline...")
    for item in test_cases:
        query = item["user_input"]
        payload = {
            "query": query,
            "top_n_candidates": 15,
            "top_n_rerank": 4,
        }
        try:
            response = requests.post(
                FASTAPI_ENDPOINT,
                json=payload,
                timeout=660,
            )
            response.raise_for_status()
            res = response.json()
            answer = res.get(
                "answer",
                ""
            )
            sources = res.get(
                "sources",
                []
            )

            sources = sorted(
                sources,
                key=lambda x: x.get(
                    "rerank_score",
                    0
                ),
                reverse=True,
            )[:2]

            contexts = [
                source["full_text"][:6000]
                for source in sources
                if source.get("full_text")
            ]

        except Exception as e:
            raise RuntimeError(
                f"RAG backend failed for '{query}': {e}"
            ) from e
        records.append(
            {
                "user_input": query,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": item["reference"],
            }
        )

    return Dataset.from_list(records)

def run_ollama_eval():
    dataset = fetch_rag_responses(
        EVAL_DATASET
    )

    print(
        "\nInitializing Local Ollama Evaluator..."
    )

    evaluator_llm = OllamaRagasLLM(
        model=OLLAMA_MODEL,
        host=OLLAMA_HOST,
    )

    metrics = [
        Faithfulness(
            llm=evaluator_llm
        ),

        LLMContextRecall(
            llm=evaluator_llm
        ),

        AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=embeddings,
            strictness=1,
        ),
    ]

    print(
        f"Evaluating metrics sequentially with "
        f"{OLLAMA_MODEL} (max_workers=1)..."
    )

    run_config = RunConfig(
        max_workers=1,
        timeout=660,
        max_retries=0,
    )

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
        raise_exceptions=True,
    )

    print("\n~~~~~~~~~~~~~ BENCHMARK RESULTS ~~~~~~~~~~~~~")

    print(results)

    results_for_csv = results.to_pandas()

    output_doc = ("ragas_ollama_results.csv")

    results_for_csv.to_csv(
        output_doc,
        index=False,
    )

    print(
        f"\nDetailed log exported to "
        f"'{output_doc}'"
    )



if __name__ == "__main__":
    run_ollama_eval()