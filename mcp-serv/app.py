import json
import logging
import sys
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import tempfile

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

_logger = logging.getLogger(__name__)


mcp = FastMCP("KnowledgeBaseServer")


@mcp.tool()
async def search_knowledge_base(
    query: str,
    filters: dict | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Search the knowledge base using semantic similarity.

    Args:
        query: The search query text to find relevant documents.
        filters: Optional metadata filters (e.g., {"category": "Results"}).
        top_k: Maximum number of results to return (default: 5).

    Returns:
        List of matching documents with id, score, text, and metadata.
    """
    try:
        await db.ensure_collection_exists()
        query_vector = await get_single_embedding(query)
        results = await db.semantic_search(query_vector=query_vector, filters=filters, top_k=top_k)
        return [{"id": r.id, "score": round(r.score, 4), "text": r.text, "metadata": r.metadata} for r in results]
    except Exception as e:
        _logger.error(f"Search failed: {e}")
        return []


@mcp.tool()
async def get_document_metadata(document_id: str) -> dict | None:
    """
    Retrieve metadata for a specific document by ID.

    Args:
        document_id: The unique identifier of the document.

    Returns:
        Document metadata dictionary or None if not found.
    """
    try:
        return await db.get_by_id(document_id)
    except Exception as e:
        _logger.error(f"Failed to get document {document_id}: {e}")
        return None


@mcp.tool()
async def summarize_document(document_id: str) -> str:
    """
    Generate a summary of a document using Map-Reduce.

    Args:
        document_id: The unique identifier of the document to summarize.

    Returns:
        A coherent summary of the document content.
    """
    try:
        payload = await db.get_by_id(document_id)
        if not payload:
            return "Document not found."
        text = payload.get("text", "")
        if not text:
            return "Document has no text content."
        return await map_reduce_summarize([text])
    except Exception as e:
        _logger.error(f"Summarization failed for {document_id}: {e}")
        return f"Error summarizing document: {str(e)}"


@mcp.tool()
async def index_document(text: str, metadata: dict | None = None) -> dict:
    """
    Index a new document into the knowledge base.

    Args:
        text: The text content of the document.
        metadata: Optional metadata to attach to the document.

    Returns:
        Dictionary with document_id and status.
    """
    try:
        if not text or not text.strip():
            return {"status": "error", "error": "Text cannot be empty"}

        chunks = chunk_documents([{"text": text, **(metadata or {})}], clean_text=True)
        texts = [c["text"] for c in chunks]

        vectors = await get_embeddings(texts)
        await db.ensure_collection_exists()
        await db.batch_insert(vectors, chunks)

        return {"status": "success", "chunks_indexed": len(chunks)}
    except Exception as e:
        _logger.error(f"Indexing failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def index_chunked_documents(
    documents: list[dict],
    clean_text: bool = True,
    classify: bool = False,
) -> dict:
    """
    Index multiple pre-chunked documents into the knowledge base.

    Args:
        documents: List of document dicts with 'text' field.
        clean_text: Whether to clean text before indexing.
        classify: Whether to classify chunks with LLM.

    Returns:
        Dictionary with count of indexed chunks.
    """
    try:
        if not documents:
            return {"status": "error", "error": "No documents provided"}

        chunks = chunk_documents(documents, clean_text=clean_text)
        texts = [c["text"] for c in chunks]

        if classify:
            classifications = await classify_chunks_batch(texts[:10])
            for i, cls in enumerate(classifications):
                if i < len(chunks):
                    chunks[i].update(cls)

        vectors = await get_embeddings(texts)
        await db.ensure_collection_exists()
        await db.batch_insert(vectors, chunks)

        return {"status": "success", "chunks_indexed": len(chunks)}
    except Exception as e:
        _logger.error(f"Batch indexing failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def rag_query(question: str, filters: dict | None = None, top_k: int = 5) -> dict:
    """
    Perform a RAG query: search knowledge base and generate answer.

    Args:
        question: The question to answer.
        filters: Optional metadata filters for search.
        top_k: Number of context chunks to retrieve.

    Returns:
        Dictionary with answer, sources, and intent.
    """
    try:
        intent = await classify_user_intent(question)

        search_filters = filters
        if intent.get("filters"):
            search_filters = {**(filters or {}), **intent["filters"]}

        await db.ensure_collection_exists()
        query_vector = await get_single_embedding(question)
        results = await db.semantic_search(query_vector=query_vector, filters=search_filters, top_k=top_k)

        context_chunks = [{"id": r.id, "text": r.text} for r in results]

        if not context_chunks:
            return {
                "answer": "Insufficient data in the provided sources.",
                "sources": [],
                "intent": intent,
            }

        answer = await rag_answer(question, context_chunks)

        return {
            "answer": answer,
            "sources": context_chunks,
            "intent": intent,
        }
    except Exception as e:
        _logger.error(f"RAG query failed: {e}")
        return {"answer": f"Error processing query: {str(e)}", "sources": [], "intent": {}}


@mcp.tool()
async def classify_query_intent(question: str) -> dict:
    """
    Classify the intent of a user query.

    Args:
        question: The user's question text.

    Returns:
        Dictionary with intent, filters, and complexity.
    """
    try:
        return await classify_user_intent(question)
    except Exception as e:
        _logger.error(f"Intent classification failed: {e}")
        return {"intent": "GENERAL", "filters": None, "complexity": "simple"}


@mcp.tool()
async def judge_answer(question: str, context: str, answer: str) -> dict:
    """
    Evaluate answer quality using LLM-as-judge.

    Args:
        question: The original question.
        context: The context provided for answering.
        answer: The AI-generated answer to evaluate.

    Returns:
        Dictionary with hallucination, coverage_score, citation_accuracy, verdict.
    """
    judge_prompt = """You are an expert evaluator for AI-generated scientific summaries.
Given the original context, a question, and the AI's answer, evaluate:

1. hallucination: "none" | "partial" | "yes"
2. coverage_score: 0-10 (did the answer cover all key facts?)
3. citation_accuracy: "accurate" | "partial" | "missing"
4. verdict: "pass" | "fail"

Return ONLY a JSON object. No markdown.

Example:
{"hallucination": "none", "coverage_score": 9, "citation_accuracy": "accurate", "verdict": "pass"}"""

    try:
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
            _logger.warning(f"Failed to parse judge response: {raw}")
            return {"hallucination": "unknown", "coverage_score": 0, "citation_accuracy": "unknown", "verdict": "fail"}
    except Exception as e:
        _logger.error(f"Judge evaluation failed: {e}")
        return {"hallucination": "unknown", "coverage_score": 0, "citation_accuracy": "unknown", "verdict": "fail"}


@mcp.tool()
async def delete_document(document_id: str) -> dict:
    """
    Delete a document from the knowledge base.

    Args:
        document_id: The unique identifier of the document to delete.

    Returns:
        Dictionary with status indicating success or failure.
    """
    try:
        success = await db.delete_by_id(document_id)
        if success:
            return {"status": "success", "document_id": document_id}
        return {"status": "error", "error": "Document not found or delete failed"}
    except Exception as e:
        _logger.error(f"Delete failed for {document_id}: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def delete_by_filter(filters: dict) -> dict:
    """
    Delete documents matching metadata filters.

    Args:
        filters: Metadata filters to match documents for deletion.

    Returns:
        Dictionary with status of the deletion operation.
    """
    try:
        if not filters:
            return {"status": "error", "error": "Filters required for safety"}
        count = await db.delete_by_filter(filters)
        return {"status": "success", "message": "Documents deleted"}
    except Exception as e:
        _logger.error(f"Delete by filter failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def get_collection_info() -> dict:
    """
    Get information about the knowledge base collection.

    Returns:
        Dictionary with collection name, point count, and status.
    """
    try:
        return await db.get_collection_info()
    except Exception as e:
        _logger.error(f"Failed to get collection info: {e}")
        return {"status": "error", "error": str(e)}


_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


_strong_client = AsyncOpenAI(
    api_key=settings.strong_llm_api_key,
    base_url=settings.strong_llm_base_url,
)


async def _execute_mcp_tool(tool_name: str, arguments: dict) -> str:
    """Execute an MCP tool via stdio and return the result."""
    server_params = StdioServerParameters(command="python", args=[__file__, "--mcp"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            if result.content:
                return result.content[0].text
            return "null"


async def agent_loop(question: str, filters: dict | None) -> tuple[str, list[dict]]:
    """
    Agent loop that uses MCP tools for knowledge retrieval.
    Uses rag_query MCP tool for simple queries, or agent loop for complex ones.
    """
    try:
        intent = await classify_user_intent(question)

        # Use rag_query MCP tool for simple fact-finding
        if intent.get("intent") in ("FIND_FACT", "METHODOLOGY", "GREETING"):
            result_str = await _execute_mcp_tool("rag_query", {
                "question": question,
                "filters": filters,
                "top_k": 5,
            })
            try:
                result = json.loads(result_str)
                return result.get("answer", ""), result.get("sources", [])
            except json.JSONDecodeError:
                pass

        # Complex queries use the full agent loop
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
    except Exception as e:
        _logger.error(f"Agent loop failed: {e}")
        return f"Error processing request: {str(e)}", []


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
    """Upload and index a PDF, JSON, CSV, TXT, or MD file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate file type
    valid_extensions = ['.pdf', '.json', '.csv', '.txt', '.md']
    file_ext = '.' + file.filename.split('.')[-1].lower()

    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file_ext}'. Supported: {', '.join(valid_extensions)}"
        )

    # Save temporary file
    temp_dir = Path(tempfile.gettempdir()) / "docmind_uploads"
    temp_dir.mkdir(exist_ok=True)

    # Use unique filename to avoid conflicts
    import uuid
    unique_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    temp_file = temp_dir / unique_name

    try:
        # Save uploaded file
        contents = await file.read()
        with open(temp_file, 'wb') as f:
            f.write(contents)

        # Handle text files (txt, md) directly
        if file_ext in ['.txt', '.md']:
            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    text = f.read()

                if not text.strip():
                    return {"file": file.filename, "type": file_ext.lstrip('.'), "status": "error", "error": "File is empty"}

                # Chunk and index text
                documents = [{'text': text, 'source': file.filename}]
                documents = chunk_documents(documents, clean_text=True)

                if not documents:
                    return {"file": file.filename, "type": file_ext.lstrip('.'), "status": "error", "error": "No text chunks extracted"}

                texts = [d['text'] for d in documents]
                vectors = await get_embeddings(texts)
                await db.ensure_collection_exists()
                await db.batch_insert(vectors, documents)

                return {
                    "file": file.filename,
                    "type": file_ext.lstrip('.'),
                    "status": "success",
                    "chunks_indexed": len(documents)
                }
            except UnicodeDecodeError:
                return {"file": file.filename, "type": file_ext.lstrip('.'), "status": "error", "error": "Could not read file as UTF-8 text"}
            except Exception as e:
                _logger.error(f"Text file indexing failed: {e}")
                return {"file": file.filename, "type": file_ext.lstrip('.'), "status": "error", "error": str(e)}

        # Index other file types using dataset_loader
        result = await index_dataset(
            str(temp_file),
            dataset_type=file_ext.lstrip('.'),
            classify_chunks=False,
            clean_text=True,
            use_ner=False
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception:
            pass


@api.get("/health")
async def health():
    return {"status": "ok"}


# MCP HTTP endpoints for AI agents


@api.post("/mcp/tools/{tool_name}")
async def call_mcp_tool(tool_name: str, arguments: dict = None):
    """Direct HTTP endpoint to call MCP tools."""
    if not arguments:
        arguments = {}

    valid_tools = {
        "search_knowledge_base": search_knowledge_base,
        "get_document_metadata": get_document_metadata,
        "summarize_document": summarize_document,
        "index_document": index_document,
        "index_chunked_documents": index_chunked_documents,
        "rag_query": rag_query,
        "classify_query_intent": classify_query_intent,
        "judge_answer": judge_answer,
        "delete_document": delete_document,
        "delete_by_filter": delete_by_filter,
        "get_collection_info": get_collection_info,
    }

    if tool_name not in valid_tools:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found. Available: {list(valid_tools.keys())}")

    try:
        result = await valid_tools[tool_name](**arguments)
        return {"status": "success", "result": result}
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid arguments: {str(e)}")
    except Exception as e:
        _logger.error(f"Tool execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/mcp/tools")
async def list_mcp_tools():
    """List all available MCP tools with descriptions."""
    return {
        "tools": [
            {"name": "search_knowledge_base", "description": "Search knowledge base using semantic similarity", "args": ["query", "filters?", "top_k?"]},
            {"name": "get_document_metadata", "description": "Retrieve metadata for a document by ID", "args": ["document_id"]},
            {"name": "summarize_document", "description": "Generate summary using Map-Reduce", "args": ["document_id"]},
            {"name": "index_document", "description": "Index a new document into knowledge base", "args": ["text", "metadata?"]},
            {"name": "index_chunked_documents", "description": "Index multiple pre-chunked documents", "args": ["documents", "clean_text?", "classify?"]},
            {"name": "rag_query", "description": "Perform RAG query with answer generation", "args": ["question", "filters?", "top_k?"]},
            {"name": "classify_query_intent", "description": "Classify user query intent", "args": ["question"]},
            {"name": "judge_answer", "description": "Evaluate answer quality with LLM-as-judge", "args": ["question", "context", "answer"]},
            {"name": "delete_document", "description": "Delete document by ID", "args": ["document_id"]},
            {"name": "delete_by_filter", "description": "Delete documents matching filters", "args": ["filters"]},
            {"name": "get_collection_info", "description": "Get collection statistics", "args": []},
        ]
    }


if __name__ == "__main__":
    if "--mcp" in sys.argv:
        mcp.run(transport="stdio")
    else:
        import uvicorn
        uvicorn.run("app:api", host="0.0.0.0", port=8000, reload=True)