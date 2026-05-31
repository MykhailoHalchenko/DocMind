import json
from contextlib import asynccontextmanager

import httpx
import tiktoken
from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

from chunking import chunk_documents
from config import settings
from embeddings import get_embeddings, get_single_embedding
from vector_db import db


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("KnowledgeBaseServer")


@mcp.tool()
async def search_knowledge_base(
    query: str,
    filters: dict | None = None,
    top_k: int = 5,
) -> list[dict]:
    await db.ensure_collection_exists()
    query_vector = await get_single_embedding(query)
    results = await db.semantic_search(
        query_vector=query_vector,
        filters=filters,
        top_k=top_k,
    )
    return [
        {
            "id": r.id,
            "score": round(r.score, 4),
            "text": r.text,
            "metadata": r.metadata,
        }
        for r in results
    ]


@mcp.tool()
async def get_document_metadata(document_id: str) -> dict | None:
    return await db.get_by_id(document_id)



tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


async def call_llm(
    messages: list[dict],
    model: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    tools: list[dict] | None = None,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()


async def classify_intent(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intent classifier. "
                "Respond with a single word: SEARCH, GENERAL, GREETING, or OTHER."
            ),
        },
        {"role": "user", "content": question},
    ]
    result = await call_llm(
        messages=messages,
        model=settings.fast_llm_model,
        api_key=settings.openai_api_key,
    )
    return result["choices"][0]["message"]["content"].strip().upper()


async def agent_answer(question: str, filters: dict | None) -> tuple[str, list[dict]]:
    server_params = StdioServerParameters(
        command="python",
        args=["app.py", "--mcp"],
    )
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
                        "You are a knowledgeable assistant with access to a knowledge base. "
                        "Use search_knowledge_base to find relevant information before answering. "
                        "Always cite your sources."
                    ),
                },
                {"role": "user", "content": question},
            ]

            while True:
                result = await call_llm(
                    messages=messages,
                    model=settings.heavy_llm_model,
                    api_key=settings.heavy_llm_api_key,
                    base_url=settings.heavy_llm_base_url,
                    tools=tools_schema,
                )

                choice = result["choices"][0]
                msg = choice["message"]
                messages.append(msg)

                if choice["finish_reason"] != "tool_calls":
                    return msg["content"], sources

                for tool_call in msg.get("tool_calls", []):
                    fn_name = tool_call["function"]["name"]
                    fn_args = json.loads(tool_call["function"]["arguments"])

                    if filters and fn_name == "search_knowledge_base":
                        fn_args.setdefault("filters", filters)

                    tool_result = await session.call_tool(fn_name, fn_args)
                    tool_content = (
                        tool_result.content[0].text if tool_result.content else "[]"
                    )

                    if fn_name == "search_knowledge_base":
                        try:
                            sources = json.loads(tool_content)
                        except json.JSONDecodeError:
                            sources = []

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": tool_content,
                        }
                    )


class QueryRequest(BaseModel):
    question: str
    filters: dict | None = None


class QueryResponse(BaseModel):
    answer: str
    intent: str
    sources: list[dict]
    token_usage: dict


class IndexRequest(BaseModel):
    documents: list[dict]
    text_field: str = "text"
    auto_chunk: bool = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.ensure_collection_exists()
    yield
    await db.close()


api = FastAPI(title="Knowledge Base API", lifespan=lifespan)


@api.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    intent = await classify_intent(request.question)
    answer, sources = await agent_answer(request.question, request.filters)

    input_tokens = count_tokens(request.question)
    output_tokens = count_tokens(answer)

    return QueryResponse(
        answer=answer,
        intent=intent,
        sources=sources,
        token_usage={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


@api.post("/index")
async def index(request: IndexRequest):
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    docs = request.documents
    if request.auto_chunk:
        docs = chunk_documents(docs, text_field=request.text_field)

    texts = [d.get("text", "") for d in docs]
    if any(not t for t in texts):
        raise HTTPException(status_code=400, detail="Each document must have a 'text' field")

    vectors = await get_embeddings(texts)
    await db.batch_insert(vectors, docs)

    return {"indexed": len(docs), "chunks": len(docs) if request.auto_chunk else None}


@api.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import sys
    if "--mcp" in sys.argv:
        mcp.run(transport="stdio")
    else:
        import uvicorn
        uvicorn.run("app:api", host="0.0.0.0", port=8000, reload=True)