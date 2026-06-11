import json
import sys
import asyncio
from contextlib import asynccontextmanager

import tiktoken
from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from pydantic import BaseModel

from chunking import chunk_documents
from classifier import classify_user_intent, classify_chunks_batch
from config import settings
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


@api.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    if "--mcp" in sys.argv:
        mcp.run(transport="stdio")
    else:
        import uvicorn
        uvicorn.run("app:api", host="0.0.0.0", port=8000, reload=True)