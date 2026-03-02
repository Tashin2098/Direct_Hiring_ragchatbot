import json
import asyncio
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.rag_chat import RAGEngine

app = FastAPI(
    title="Direct Hiring Guide RAG Chatbot",
    description="Q&A chatbot for Direct Hiring Guide using RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in real prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    rag_engine = RAGEngine()
except FileNotFoundError as e:
    print(f"⚠️ {e}")
    rag_engine = None


class ChatRequest(BaseModel):
    message: str
    top_k: Optional[int] = None  # optional override


class Citation(BaseModel):
    source: str
    snippet: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    fallback_used: bool


@app.get("/")
def read_root():
    return {
        "service": "Direct Hiring Guide RAG Chatbot",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    if rag_engine is None:
        return {
            "status": "unhealthy",
            "message": "RAG engine not initialized. Run indexing first.",
        }
    return {"status": "healthy", "message": "RAG engine loaded and ready"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if rag_engine is None:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not initialized. Run indexing first.",
        )

    try:
        result = rag_engine.answer(request.message)
        return ChatResponse(
            answer=result["answer"],
            citations=[Citation(**c) for c in result["citations"]],
            fallback_used=result["fallback_used"],
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "INTERNAL_ERROR", "message": str(e)},
        )


@app.get("/chat_stream")
async def chat_stream(
    request: Request,
    message: str = Query(..., description="User message to answer"),
):
    """
    Streams the answer token-by-token using Server-Sent Events (SSE).

    Stops immediately if the client disconnects (EventSource closed).
    """

    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if rag_engine is None:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not initialized. Run indexing first.",
        )

    def sse_pack(obj: dict) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    async def event_generator():
        try:
            # rag_engine.answer_stream(...) is a normal (sync) generator.
            # We iterate it, but between yields we check client disconnect.
            for ev in rag_engine.answer_stream(message):
                if await request.is_disconnected():
                    # Client closed EventSource -> stop generating/streaming
                    break

                yield sse_pack(ev)

                # Give control back to event loop so disconnect can be detected quickly
                await asyncio.sleep(0)

        except Exception as e:
            # If anything crashes mid-stream, send an error event (if still connected)
            if not await request.is_disconnected():
                yield sse_pack(
                    {"event": "error", "message": str(e), "error": "INTERNAL_ERROR"}
                )
                yield sse_pack(
                    {"event": "done", "answer": "", "citations": [], "fallback_used": True}
                )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/stats")
def get_stats():
    if rag_engine is None or rag_engine.metadata is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    return {
        "total_chunks": rag_engine.metadata["num_chunks"],
        "embedding_dimension": rag_engine.metadata["embedding_dimension"],
        "embedding_model": rag_engine.metadata["embedding_model"],
    }


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("🚀 Starting RAG Chatbot Server")
    print("=" * 60)
    print("📍 Server: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)