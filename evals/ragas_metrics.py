"""RAGAS metrics — the four standard RAG quality dimensions, via the real
`ragas` library.

  faithfulness       — is every claim in the answer actually supported by
                        the retrieved context? (catches hallucination)
  answer_relevancy    — does the answer actually address the question asked?
  context_precision   — of the chunks retrieved, how many are relevant?
  context_recall       — does retrieved context contain everything the
                        reference answer needed?

ragas is built on LangChain: its LLM-as-judge and its embedding-based
answer_relevancy metric both expect LangChain-wrapped clients, so this
module is the one place in the project that touches LangChain — everything
else (planner, responder, guardrails) is framework-free by design.

This pulls in torch + sentence-transformers via ragas' dependency chain —
a deliberately heavier install than the rest of this project, accepted
here specifically to use the real, industry-standard library rather than
a hand-rolled reimplementation.
"""

from __future__ import annotations

from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from backend.config import EMBEDDING_MODEL, GEMINI_API_KEY, GROQ_API_KEY, GROQ_MODEL

_judge_llm = None
_judge_embeddings = None
_metrics = None


def _init():
    global _judge_llm, _judge_embeddings, _metrics
    if _metrics is not None:
        return

    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_groq import ChatGroq

    _judge_llm = LangchainLLMWrapper(
        ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
    )
    _judge_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model=f"models/{EMBEDDING_MODEL}", google_api_key=GEMINI_API_KEY
        )
    )
    _metrics = [
        Faithfulness(llm=_judge_llm),
        ResponseRelevancy(llm=_judge_llm, embeddings=_judge_embeddings),
        LLMContextPrecisionWithReference(llm=_judge_llm),
        LLMContextRecall(llm=_judge_llm),
    ]


def score(question: str, contexts: list[str], model_answer: str, reference_answer: str) -> dict:
    """Compute all four RAGAS metrics for one Q&A pair via the real ragas library."""
    _init()
    dataset = EvaluationDataset.from_list([
        {
            "user_input": question,
            "retrieved_contexts": contexts or [""],
            "response": model_answer,
            "reference": reference_answer,
        }
    ])
    result = evaluate(
        dataset=dataset,
        metrics=_metrics,
        llm=_judge_llm,
        embeddings=_judge_embeddings,
        show_progress=False,
        raise_exceptions=False,
    )
    row = result.to_pandas().iloc[0]

    def safe(col: str) -> float:
        val = row.get(col)
        return round(float(val), 3) if val is not None and val == val else 0.0  # NaN check

    return {
        "faithfulness": safe("faithfulness"),
        "answer_relevancy": safe("answer_relevancy"),
        "context_precision": safe("llm_context_precision_with_reference"),
        "context_recall": safe("context_recall"),
    }
