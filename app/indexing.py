import os
import json
import pickle
import time
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
from PyPDF2 import PdfReader
from openai import OpenAI
import faiss

from app.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_DIR,
    STORAGE_DIR,
    EMBED_BATCH_SIZE,
    EMBED_MAX_RETRIES,
    EMBED_RETRY_BACKOFF,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("indexing")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


class PDFIndexer:
    """
    STAGE 1-4: PDFs → cleaned text → chunks → embeddings → FAISS index
    """

    def __init__(self):
        self.storage_dir = Path(STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Stage 1: load & clean PDFs ----------

    def stage_1_load_pdfs(self) -> List[Dict]:
        logger.info("📄 STAGE 1: Loading PDFs from %s", DATA_DIR)
        pdf_files = list(Path(DATA_DIR).glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in '{DATA_DIR}'")

        pdf_texts: List[Dict] = []

        for pdf_file in pdf_files:
            logger.info("   → Reading: %s", pdf_file.name)
            reader = PdfReader(pdf_file)
            pages = []
            for page in reader.pages:
                text = page.extract_text() or ""
                text = " ".join(text.split())  # normalize whitespace
                pages.append(text)
            full_text = "\n\n".join(pages)
            pdf_texts.append(
                {
                    "filename": pdf_file.name,
                    "text": full_text,
                    "num_pages": len(reader.pages),
                }
            )
            logger.info("   ✓ %s: %d pages", pdf_file.name, len(reader.pages))

        logger.info("✅ STAGE 1 COMPLETE: %d PDF(s) loaded", len(pdf_files))
        return pdf_texts

    # ---------- Stage 2: chunking (overlap, min length) ----------

    def stage_2_chunk_text(self, pdf_texts: List[Dict]) -> List[Dict]:
        logger.info("✂️  STAGE 2: Chunking text (chunk=%d, overlap=%d)",
                    CHUNK_SIZE, CHUNK_OVERLAP)
        chunks: List[Dict] = []
        chunk_id = 0

        for pdf_info in pdf_texts:
            filename = pdf_info["filename"]
            text = pdf_info["text"]

            for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk = text[i : i + CHUNK_SIZE]
                if len(chunk.strip()) < 80:
                    continue  # skip tiny/noisy chunks
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk,
                        "source": filename,
                        "start_char": i,
                        "length": len(chunk),
                    }
                )
                chunk_id += 1

        if not chunks:
            raise ValueError("No valid chunks created from PDFs")

        logger.info("   Created %d chunks total", len(chunks))
        logger.info("✅ STAGE 2 COMPLETE")
        return chunks

    # ---------- Stage 3: embeddings with batching + retries ----------

    def stage_3_create_embeddings(self, chunks: List[Dict]) -> List[List[float]]:
        logger.info("🔢 STAGE 3: Generating embeddings using %s", EMBEDDING_MODEL)
        embeddings: List[List[float]] = []

        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            texts = [c["text"] for c in batch]
            attempt = 0

            while True:
                attempt += 1
                logger.info(
                    "   Batch %d/%d (size=%d, attempt=%d)",
                    start // EMBED_BATCH_SIZE + 1,
                    (len(chunks) - 1) // EMBED_BATCH_SIZE + 1,
                    len(texts),
                    attempt,
                )
                try:
                    response = client.embeddings.create(
                        input=texts,
                        model=EMBEDDING_MODEL,
                    )
                    for d in response.data:
                        embeddings.append(d.embedding)
                    break
                except Exception as e:
                    logger.warning("   Embedding error: %s", e)
                    if attempt >= EMBED_MAX_RETRIES:
                        raise
                    sleep_for = EMBED_RETRY_BACKOFF ** attempt
                    logger.info("   Retrying in %.1f seconds...", sleep_for)
                    time.sleep(sleep_for)

        logger.info("   Generated %d embeddings", len(embeddings))
        logger.info("✅ STAGE 3 COMPLETE")
        return embeddings

    # ---------- Stage 4: build FAISS + metadata ----------

    def stage_4_build_faiss_index(
        self, embeddings: List[List[float]], chunks: List[Dict]
    ):
        logger.info("🗂️  STAGE 4: Building FAISS index...")

        embeddings_array = np.array(embeddings, dtype="float32")
        if embeddings_array.ndim != 2:
            raise ValueError("Embeddings must be 2D array")

        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)

        index_path = self.storage_dir / "faiss_index.bin"
        faiss.write_index(index, str(index_path))
        logger.info("   ✓ Saved FAISS index → %s", index_path)

        metadata = {
            "chunks": chunks,
            "num_chunks": len(chunks),
            "embedding_dimension": dimension,
            "embedding_model": EMBEDDING_MODEL,
        }
        metadata_path = self.storage_dir / "docs_metadata.pkl"
        with metadata_path.open("wb") as f:
            pickle.dump(metadata, f)
        logger.info("   ✓ Saved metadata → %s", metadata_path)

        summary = {
            "total_chunks": len(chunks),
            "embedding_dimension": dimension,
            "pdfs_indexed": len({c["source"] for c in chunks}),
            "index_file": str(index_path),
            "metadata_file": str(metadata_path),
        }
        summary_path = self.storage_dir / "index_summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        logger.info("   ✓ Saved index summary → %s", summary_path)

        logger.info("✅ STAGE 4 COMPLETE")
        return index, metadata

    # ---------- Orchestrator ----------

    def run_full_pipeline(self):
        logger.info("=" * 60)
        logger.info("🚀 STARTING RAG INDEXING PIPELINE")
        logger.info("=" * 60)

        pdf_texts = self.stage_1_load_pdfs()
        chunks = self.stage_2_chunk_text(pdf_texts)
        embeddings = self.stage_3_create_embeddings(chunks)
        index, metadata = self.stage_4_build_faiss_index(embeddings, chunks)

        logger.info("=" * 60)
        logger.info("✅ INDEXING PIPELINE COMPLETE")
        logger.info("   Total chunks: %d", len(chunks))
        logger.info("   Embedding dimension: %d", len(embeddings[0]))
        logger.info("   Index path: %s", self.storage_dir / "faiss_index.bin")
        logger.info("=" * 60)


if __name__ == "__main__":
    indexer = PDFIndexer()
    indexer.run_full_pipeline()
