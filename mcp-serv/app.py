import json
import sys
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path for root imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import tiktoken
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastmcp import FastMCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from pydantic import BaseModel

from chunking import chunk_documents
from classifier import classify_user_intent, classify_chunks_batch
from config import settings
from dataset_loader import index_dataset, index_multiple_datasets
from embeddings import get_embeddings, get_single_embedding
from llm import rag_answer, map_reduce_summarize
from vector_db import db


mcp = FastMCP("KnowledgeBaseServer")


@mcp.tool()
async def search_knowledge_base(
    query: str,
    filters: dict | None = None,
    top_k: int = 5,
) -> list[dict]:
    await db.ensure_collection_exists()
    query_vector = await get_single_embedding(query)
    results = await db.semantic_search(query_vector=query_vector, filters=filters, top_k=top_k)
    return [{"id": r.id, "score": round(r.score, 4), "text": r.text, "metadata": r.metadata} for r in results]


@mcp.tool()
async def get_document_metadata(document_id: str) -> dict | None:
    return await db.get_by_id(document_id)


@mcp.tool()
async def summarize_document(document_id: str) -> str:
    payload = await db.get_by_id(document_id)
    if not payload:
        return "Document not found."
    return await map_reduce_summarize([payload.get("text", "")])


@mcp.tool()
async def judge_answer(question: str, context: str, answer: str) -> dict:
    """Evaluate answer quality using LLM"""
    judge_prompt = """You are an expert evaluator for AI-generated scientific summaries.
Given the original context, a question, and the AI's answer, evaluate:

1. hallucination: "none" | "partial" | "yes"
2. coverage_score: 0-10 (did the answer cover all key facts?)
3. citation_accuracy: "accurate" | "partial" | "missing"
4. verdict: "pass" | "fail"

Return ONLY a JSON object. No markdown.

Example:
{"hallucination": "none", "coverage_score": 9, "citation_accuracy": "accurate", "verdict": "pass"}"""
    
    prompt = f"Question: {question}\n\nContext:\n{context[:3000]}\n\nAI Answer:\n{answer}"
    response = await _strong_client.chat.completions.create(
        model=settings.strong_llm_model,
        messages=[
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=150,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"hallucination": "unknown", "coverage_score": 0, "citation_accuracy": "unknown", "verdict": "fail"}


_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


_strong_client = AsyncOpenAI(
    api_key=settings.strong_llm_api_key,
    base_url=settings.strong_llm_base_url,
)


async def agent_loop(question: str, filters: dict | None) -> tuple[str, list[dict]]:
    server_params = StdioServerParameters(command="python", args=[__file__, "--mcp"])
    sources: list[dict] = []

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            tools_schema = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema,
                    },
                }
                for t in mcp_tools.tools
            ]

            messages: list[dict] = [
                {
                    "role": "system",
                    "content": (
                        "You are a precise scientific assistant with access to a knowledge base. "
                        "Always call search_knowledge_base before answering. "
                        "Use ONLY retrieved context. Cite sources as [chunk_id]. "
                        "If facts are missing, say: 'Insufficient data in the provided sources.'"
                    ),
                },
                {"role": "user", "content": question},
            ]

            for _ in range(8):
                response = await _strong_client.chat.completions.create(
                    model=settings.strong_llm_model,
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto",
                )
                choice = response.choices[0]
                msg = choice.message
                messages.append(msg.model_dump(exclude_none=True))

                if choice.finish_reason != "tool_calls":
                    return msg.content or "", sources

                for tool_call in msg.tool_calls or []:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    if filters and fn_name == "search_knowledge_base":
                        fn_args.setdefault("filters", filters)

                    tool_result = await session.call_tool(fn_name, fn_args)
                    tool_content = tool_result.content[0].text if tool_result.content else "[]"

                    if fn_name == "search_knowledge_base":
                        try:
                            sources = json.loads(tool_content)
                        except json.JSONDecodeError:
                            sources = []

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content,
                    })

    return "No answer generated.", sources


class QueryRequest(BaseModel):
    question: str
    filters: dict | None = None


class QueryResponse(BaseModel):
    answer: str
    intent: dict
    sources: list[dict]
    token_usage: dict


class IndexRequest(BaseModel):
    documents: list[dict]
    text_field: str = "text"
    auto_chunk: bool = True
    classify_chunks: bool = False
    clean_text: bool = False
    use_ner: bool = False


class SummarizeRequest(BaseModel):
    chunks: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.ensure_collection_exists()
    yield
    await db.close()


api = FastAPI(title="Knowledge Base API", lifespan=lifespan)

# Add CORS middleware for frontend access
from fastapi.middleware.cors import CORSMiddleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
from fastapi.staticfiles import StaticFiles
import os

static_dir = Path(__file__).parent.parent / "frontend"
if static_dir.exists():
    api.mount("/static", StaticFiles(directory=static_dir), name="static")

@api.get("/")
async def root():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Knowledge Base API - Frontend not found. Please create frontend/index.html"}


@api.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    intent = await classify_user_intent(request.question)

    filters = request.filters
    if intent.get("filters"):
        filters = {**(filters or {}), **intent["filters"]}

    answer, sources = await agent_loop(request.question, filters)

    return QueryResponse(
        answer=answer,
        intent=intent,
        sources=sources,
        token_usage={
            "input_tokens": count_tokens(request.question),
            "output_tokens": count_tokens(answer),
            "total_tokens": count_tokens(request.question) + count_tokens(answer),
        },
    )


@api.post("/index")
async def index(request: IndexRequest):
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    docs = request.documents
    if request.auto_chunk:
        docs = chunk_documents(
            docs,
            text_field=request.text_field,
            use_ner=request.use_ner,
            clean_text=request.clean_text,
        )

    texts = [d.get("text", "") for d in docs]
    if any(not t for t in texts):
        raise HTTPException(status_code=400, detail="Each document must have a 'text' field")

    if request.classify_chunks:
        classifications = await classify_chunks_batch(texts)
        for doc, cls in zip(docs, classifications):
            doc.update(cls)

    vectors = await get_embeddings(texts)
    await db.batch_insert(vectors, docs)

    return {"indexed": len(docs)}


@api.post("/summarize")
async def summarize(request: SummarizeRequest):
    if not request.chunks:
        raise HTTPException(status_code=400, detail="No chunks provided")
    summary = await map_reduce_summarize(request.chunks)
    return {"summary": summary}


@api.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and index a PDF, JSON, or CSV file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file type
    valid_extensions = ['.pdf', '.json', '.csv']
    file_ext = '.' + file.filename.split('.')[-1].lower()
    
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported: {', '.join(valid_extensions)}"
        )
    
    # Save temporary file
    temp_dir = Path(tempfile.gettempdir()) / "docmind_uploads"
    temp_dir.mkdir(exist_ok=True)
    
    temp_file = temp_dir / file.filename
    
    try:
        # Save uploaded file
        contents = await file.read()
        with open(temp_file, 'wb') as f:
            f.write(contents)
        
        # Index the dataset
        result = await index_dataset(
            str(temp_file),
            dataset_type=file_ext.lstrip('.'),
            classify_chunks=False,
            clean_text=True,
            use_ner=False
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()


@api.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    if "--mcp" in sys.argv:
        mcp.run(transport="stdio")
    else:
        import uvicorn
        uvicorn.run("app:api", host="0.0.0.0", port=8000, reload=True)