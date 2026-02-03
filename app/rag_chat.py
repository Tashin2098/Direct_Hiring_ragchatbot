import os
import pickle
import logging
from typing import List, Dict, Any

import numpy as np
from openai import OpenAI
import faiss

from app.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    EMBEDDING_MODEL,
    TOP_K_RETRIEVAL,
    STORAGE_DIR,
    MAX_CONTEXT_CHARS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("rag")

client = OpenAI(api_key=OPENAI_API_KEY)


class RAGEngine:
    """
    STAGE 5-6: Query → Retrieval → Grounded Generation
    """

    def __init__(self):
        self.index = None
        self.metadata: Dict[str, Any] = {}
        self._load_index()

    # ---------- Load index & metadata ----------

    def _load_index(self):
        index_path = os.path.join(STORAGE_DIR, "faiss_index.bin")
        metadata_path = os.path.join(STORAGE_DIR, "docs_metadata.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run indexing first: python -m app.indexing"
            )

        logger.info("📥 Loading FAISS index from %s", index_path)
        self.index = faiss.read_index(index_path)

        logger.info("📥 Loading metadata from %s", metadata_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        if self.index.ntotal == 0:
            raise RuntimeError("FAISS index is empty. Re-run indexing.")

        logger.info(
            "✅ Index loaded: %d chunks, dimension=%d",
            self.metadata["num_chunks"],
            self.metadata["embedding_dimension"],
        )

    # ---------- Stage 5a: embed query ----------

    def stage_5a_embed_query(self, query: str) -> List[float]:
        query = (query or "").strip()
        if not query:
            raise ValueError("Query cannot be empty")

        logger.info("🔍 STAGE 5a: Embedding query: %s", query)
        response = client.embeddings.create(
            input=query,
            model=EMBEDDING_MODEL,
        )
        embedding = response.data[0].embedding
        logger.info("   ✓ Generated %d-dim embedding", len(embedding))
        return embedding

    # ---------- Stage 5b: retrieve chunks ----------

    def stage_5b_retrieve_chunks(
        self, query_embedding: List[float], top_k: int | None = None
    ) -> List[Dict]:
        if top_k is None:
            top_k = TOP_K_RETRIEVAL

        logger.info("🎯 STAGE 5b: Searching FAISS index (top-%d)...", top_k)

        query_vector = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(query_vector, top_k)

        chunks = self.metadata["chunks"]
        retrieved: List[Dict] = []

        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), start=1):
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx]
            chunk_copy = dict(chunk)
            chunk_copy["distance"] = float(distance)
            chunk_copy["similarity_score"] = 1.0 / (1.0 + float(distance))
            retrieved.append(chunk_copy)

            logger.info(
                "   %d. score=%.3f source=%s len=%d",
                rank,
                chunk_copy["similarity_score"],
                chunk_copy["source"],
                len(chunk_copy["text"]),
            )

        return retrieved

    # ---------- Stage 6a: build grounded prompt ----------

    def stage_6a_build_prompt(self, query: str, retrieved_chunks: List[Dict]) -> str:
        logger.info("📝 STAGE 6a: Building RAG prompt")

        context_parts = []
        total_len = 0

        for chunk in retrieved_chunks:
            piece = f"Source: {chunk['source']}\n\n{chunk['text']}"
            if total_len + len(piece) > MAX_CONTEXT_CHARS:
                break
            context_parts.append(piece)
            total_len += len(piece)

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a friendly helpful assistant answering questions about the Direct Hiring Guide and helping users find relevant information from it. Speak naturally like you're chatting with a friend.

    IMPORTANT: Answer using only the context below. If you don't know something, just say "I don't have that info in the guide.""

    Answer naturally in short paragraphs with bullet points where helpful. Use normal spacing and keep it conversational. Also make the conversation engaging and easy to read.

    CONTEXT:
    {context}

    QUESTION: {query}

    Answer:"""

        logger.info("   ✓ Prompt length: %d chars", len(prompt))
        return prompt



    # ---------- Stage 6b: call LLM ----------

    def stage_6b_generate_answer(self, prompt: str) -> tuple[str, bool]:
        logger.info("🧠 STAGE 6b: Calling LLM %s", LLM_MODEL)
        fallback_used = False
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a friendly helpful AI assistant that answers questions. "
                            "You must answer ONLY from the provided context and "
                            "never hallucinate."
                            "Always format answers for readability:\n"
                           
                            
                            "Use simple bullet points (-) with NO bold formatting (**).\n"
                            "Add blank lines between paragraphs and lists.\n"
                            "Keep language simple and user-friendly."
                            "Be warm and approachable."
                            
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            answer = response.choices[0].message.content
            logger.info("   ✓ Answer length: %d chars", len(answer))
            return answer, fallback_used
        except Exception as e:
            logger.error("LLM error: %s", e)
            fallback_used = True
            return (
                "I’m unable to generate an answer right now due to a technical issue. "
                "Please try again in a moment.",
                fallback_used,
            )

    # ---------- Public API: full pipeline ----------

    def answer(self, query: str) -> Dict:
        logger.info("=" * 60)
        logger.info("❓ QUESTION: %s", query)
        logger.info("=" * 60)

        query_embedding = self.stage_5a_embed_query(query)
        retrieved_chunks = self.stage_5b_retrieve_chunks(query_embedding)
        prompt = self.stage_6a_build_prompt(query, retrieved_chunks)
        answer, fallback_used = self.stage_6b_generate_answer(prompt)

        citations = [
            {
                "source": c["source"],
                "snippet": (
                    c["text"][:200] + "..."
                    if len(c["text"]) > 200
                    else c["text"]
                ),
                "score": c["similarity_score"],
            }
            for c in retrieved_chunks
        ]

        logger.info("=" * 60)
        logger.info("✅ ANSWER GENERATED")
        logger.info("=" * 60)

        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "fallback_used": fallback_used,
        }


if __name__ == "__main__":
    engine = RAGEngine()
    result = engine.answer("What is direct hiring?")
    print(result["answer"])
