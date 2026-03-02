import os
import pickle
import logging
from typing import List, Dict, Any, Generator, Tuple, Optional

import numpy as np
from openai import OpenAI
import faiss

from app.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
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
    Hybrid RAG + Generative Reasoning Engine (Streaming + Role + Intent + Out-of-domain control)
    """

    def __init__(self):
        self.index = None
        self.metadata: Dict[str, Any] = {}
        self._load_index()

    # ---------------------------
    # Load FAISS index
    # ---------------------------

    def _load_index(self):
        index_path = os.path.join(STORAGE_DIR, "faiss_index.bin")
        metadata_path = os.path.join(STORAGE_DIR, "docs_metadata.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. Run indexing first."
            )

        logger.info("📥 Loading FAISS index from %s", index_path)
        self.index = faiss.read_index(index_path)

        logger.info("📥 Loading metadata from %s", metadata_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        if self.index.ntotal == 0:
            raise RuntimeError("FAISS index is empty.")

        logger.info(
            "✅ Index loaded: %d chunks, dimension=%d",
            self.metadata["num_chunks"],
            self.metadata["embedding_dimension"],
        )

    # ---------------------------
    # Stage 1: Embed Query
    # ---------------------------

    def stage_5a_embed_query(self, query: str) -> List[float]:
        query = (query or "").strip()
        if not query:
            raise ValueError("Query cannot be empty")

        response = client.embeddings.create(
            input=query,
            model=EMBEDDING_MODEL,
        )
        return response.data[0].embedding

    # ---------------------------
    # Stage 2: Retrieve Chunks
    # ---------------------------

    def stage_5b_retrieve_chunks(
        self, query_embedding: List[float], top_k: Optional[int] = None
    ) -> List[Dict]:

        if top_k is None:
            top_k = TOP_K_RETRIEVAL

        query_vector = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(query_vector, top_k)

        chunks = self.metadata["chunks"]
        retrieved: List[Dict] = []

        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(chunks):
                continue

            chunk = dict(chunks[idx])
            chunk["similarity_score"] = 1.0 / (1.0 + float(distance))
            retrieved.append(chunk)

        # Sort by similarity_score descending (safeguard)
        retrieved.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        return retrieved

    # ---------------------------
    # Domain detection (use retrieval confidence)
    # ---------------------------

    def is_in_domain(self, retrieved_chunks: List[Dict]) -> bool:
        if not retrieved_chunks:
            return False
        top_score = float(retrieved_chunks[0].get("similarity_score", 0.0))
        # Tune threshold if needed
        return top_score >= 0.60

    # ---------------------------
    # Role Detection
    # ---------------------------

    def detect_role(self, query: str) -> str:
        q = query.lower()

        if any(w in q for w in ["i am an employer", "as an employer", "i want to hire"]):
            return "employer"

        if any(w in q for w in ["i am a helper", "as a helper", "i want to work"]):
            return "helper"

        return "neutral"

    # ---------------------------
    # Intent Detection (Length Control)
    # ---------------------------

    def detect_intent(self, query: str) -> str:
        q = query.lower()

        if any(w in q for w in ["summarize", "summary", "in short", "short summary"]):
            return "summary"

        if any(w in q for w in ["brief", "short answer", "one line", "quick answer"]):
            return "brief"

        if any(w in q for w in ["detailed", "in detail", "deep", "elaborate"]):
            return "detailed"

        if any(w in q for w in ["explain", "how", "why", "what is", "how does"]):
            return "explanation"

        return "general"

    # ---------------------------
    # Stage 3: Build Hybrid Prompt
    # ---------------------------

    def stage_6a_build_prompt(self, query: str, retrieved_chunks: List[Dict]) -> str:
        logger.info("📝 Building Hybrid RAG Prompt")

        # Determine if query is in-domain
        in_domain = self.is_in_domain(retrieved_chunks)

        context_parts = []
        total_len = 0

        # Only include context if in-domain
        if in_domain:
            for chunk in retrieved_chunks:
                piece = chunk["text"]
                if total_len + len(piece) > MAX_CONTEXT_CHARS:
                    break
                context_parts.append(piece)
                total_len += len(piece)

        context = "\n\n---\n\n".join(context_parts) if context_parts else ""

        role = self.detect_role(query)
        intent = self.detect_intent(query)

        # Role instruction
        if role == "employer":
            role_instruction = "Respond primarily from an employer perspective."
        elif role == "helper":
            role_instruction = "Respond primarily from a helper perspective."
        else:
            role_instruction = (
                "Role is unclear. Provide balanced information for both employers and helpers."
            )

        # Length control instruction (only for in-domain)
        if intent == "summary":
            length_instruction = "Keep the answer very concise (2-3 short sentences maximum)."
        elif intent == "brief":
            length_instruction = "Respond in 1-2 short sentences only."
        elif intent == "explanation":
            length_instruction = "Give a clear explanation in moderate length (4-7 short lines)."
        elif intent == "detailed":
            length_instruction = "Provide a detailed answer with clear structure (steps/bullets if helpful)."
        else:
            length_instruction = "Provide a balanced medium-length answer. Avoid unnecessary extra details."

        # Out-of-domain handling instruction
        if not in_domain:
            domain_instruction = (
                "This question appears unrelated to the Direct Hiring platform. "
                "Be polite and answer ONLY in 1-2 short sentences, then redirect the user to Direct Hiring. "
                "Do NOT provide long general advice."
            )
        else:
            domain_instruction = (
                "This question appears related to Direct Hiring. Use the reference information below as helpful guidance."
            )

        prompt = f"""
You are an intelligent AI assistant for the Direct Hiring platform.

OUTPUT RULES (must follow):
- NO markdown formatting (no **bold**, no numbered lists unless asked).
- Keep the response clean and readable.

DOMAIN HANDLING:
{domain_instruction}

BEHAVIOR RULES:
1. If reference information is provided, use it to understand the platform.
2. Do NOT copy sentences directly from reference information.
3. Paraphrase clearly and naturally.
4. You may reason beyond the reference when helpful, but do not invent specific platform features.
5. If a platform detail is not clearly specified:
   - Give a reasonable helpful explanation.
   - If uncertain, say "That detail isn’t clearly outlined, but generally..."
   - Never say "Not in Guide".
6. Maintain a neutral and customer-friendly tone.
7. {role_instruction}

RESPONSE LENGTH:
{length_instruction}

REFERENCE INFORMATION (may be empty if off-topic):
{context}

USER QUESTION:
{query}

Answer:
"""
        return prompt

    # ---------------------------
    # Non-Streaming Generation
    # ---------------------------

    def stage_6b_generate_answer(self, prompt: str) -> Tuple[str, bool]:
        fallback_used = False

        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0.6,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional AI assistant for the Direct Hiring platform. "
                            "Be clear, neutral, and helpful. "
                            "Do not use markdown. "
                            "Do not copy reference text verbatim. "
                            "Follow domain handling and response length instructions."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            answer = response.choices[0].message.content or ""
            return answer, fallback_used

        except Exception as e:
            logger.error("LLM error: %s", e)
            fallback_used = True
            return (
                "I’m currently experiencing a technical issue. Please try again shortly.",
                fallback_used,
            )

    # ---------------------------
    # Streaming Generation
    # ---------------------------

    def stage_6b_generate_answer_stream(
        self, prompt: str
    ) -> Generator[Tuple[str, bool, bool], None, None]:

        fallback_used = False

        try:
            stream = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0.6,
                stream=True,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional AI assistant for the Direct Hiring platform. "
                            "Be clear, neutral, and helpful. "
                            "Do not use markdown. "
                            "Do not copy reference text verbatim. "
                            "Follow domain handling and response length instructions."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            for chunk in stream:
                delta = ""
                try:
                    delta = chunk.choices[0].delta.content or ""
                except Exception:
                    delta = ""

                if delta:
                    yield (delta, fallback_used, False)

            yield ("", fallback_used, True)

        except Exception as e:
            logger.error("Streaming error: %s", e)
            fallback_used = True
            yield (
                "I’m currently experiencing a technical issue. Please try again shortly.",
                fallback_used,
                True,
            )

    # ---------------------------
    # Build Citations
    # ---------------------------

    def _build_citations(self, retrieved_chunks: List[Dict]) -> List[Dict]:
        return [
            {
                "source": c.get("source", "unknown"),
                "snippet": (
                    (c.get("text", "")[:200] + "...")
                    if len(c.get("text", "")) > 200
                    else c.get("text", "")
                ),
                "score": float(c.get("similarity_score", 0.0)),
            }
            for c in retrieved_chunks
        ]

    # ---------------------------
    # Public API (Non-Stream)
    # ---------------------------

    def answer(self, query: str) -> Dict:
        query_embedding = self.stage_5a_embed_query(query)
        retrieved_chunks = self.stage_5b_retrieve_chunks(query_embedding)
        prompt = self.stage_6a_build_prompt(query, retrieved_chunks)
        answer, fallback_used = self.stage_6b_generate_answer(prompt)
        citations = self._build_citations(retrieved_chunks)

        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "fallback_used": fallback_used,
        }

    # ---------------------------
    # Public API (Streaming)
    # ---------------------------

    def answer_stream(self, query: str) -> Generator[Dict[str, Any], None, None]:

        query_embedding = self.stage_5a_embed_query(query)
        retrieved_chunks = self.stage_5b_retrieve_chunks(query_embedding)

        prompt = self.stage_6a_build_prompt(query, retrieved_chunks)

        # Always provide citations for transparency (even if off-topic)
        citations = self._build_citations(retrieved_chunks)

        full_text_parts: List[str] = []
        fallback_used_final = False

        yield {"event": "start"}

        for delta, fallback_used, is_done in self.stage_6b_generate_answer_stream(prompt):
            fallback_used_final = fallback_used_final or fallback_used

            if delta:
                full_text_parts.append(delta)
                yield {"event": "delta", "delta": delta}

            if is_done:
                final_answer = "".join(full_text_parts).strip()

                yield {
                    "event": "done",
                    "answer": final_answer,
                    "citations": citations,
                    "fallback_used": fallback_used_final,
                }


if __name__ == "__main__":
    engine = RAGEngine()
    print(engine.answer("How can I stay happy always?")["answer"])