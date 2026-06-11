import asyncio
import argparse
import sys
import webbrowser
from pathlib import Path
import subprocess
import time

# Add mcp-serv to path for imports
sys.path.insert(0, str(Path(__file__).parent / "mcp-serv"))

from chunking import chunk_documents
from classifier import classify_chunks_batch
import embeddings
from evaluation import print_eval_report
from pdf_parser import pdf_to_document
from vector_db import db


async def index_pdf(path: str, classify: bool = True, use_ner: bool = False) -> None:
    doc = pdf_to_document(path)
    chunks = chunk_documents([doc], clean_text=True, use_ner=use_ner)

    texts = [c["text"] for c in chunks]

    if classify:
        print(f"Classifying {len(chunks)} chunks...")
        classifications = await classify_chunks_batch(texts)
        for chunk, cls in zip(chunks, classifications):
            chunk.update(cls)

    print(f"Embedding {len(chunks)} chunks...")
    vectors = await embeddings.get_embeddings(texts)

    await db.ensure_collection_exists()
    await db.batch_insert(vectors, chunks)
    await db.close()
    print(f"Done. Indexed {len(chunks)} chunks from {path}")


async def run_eval() -> None:
    test_cases = [
        {
            "question": "What was the sample size in the study?",
            "context": "The study enrolled 120 participants divided into two groups of 60.",
            "answer": "The sample size was 120 participants [chunk_1].",
            "expected": "120 participants",
        },
        {
            "question": "What were the main findings?",
            "context": "Results showed a 35% improvement in accuracy after fine-tuning.",
            "answer": "The model improved by approximately 30% [chunk_2].",
            "expected": "35% improvement",
        },
    ]
    await print_eval_report(test_cases)


def run_server(host: str = "0.0.0.0", port: int = 8000, open_browser: bool = True) -> None:
    """Run the FastAPI server with web interface"""
    import uvicorn
    
    print(f"🚀 Starting DocMind Knowledge Base Server...")
    print(f"📚 API Server: http://localhost:{port}")
    print(f"🌐 Web Interface: http://localhost:{port}")
    print(f"Press Ctrl+C to stop\n")
    
    # Open browser after a short delay
    if open_browser:
        def open_browser_tab():
            time.sleep(2)
            webbrowser.open(f"http://localhost:{port}")
        
        import threading
        thread = threading.Thread(target=open_browser_tab, daemon=True)
        thread.start()
    
    # Run the server
    mcp_serv_dir = Path(__file__).parent / "mcp-serv"
    try:
        uvicorn.run(
            "app:api",
            host=host,
            port=port,
            reload=False,
            app_dir=str(mcp_serv_dir)
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped")


def main():
    parser = argparse.ArgumentParser(
        description="DocMind - Knowledge Base Query System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py server              # Start web server on localhost:8000
  python main.py server --port 9000  # Start on custom port
  python main.py index data.pdf      # Index a PDF file
  python main.py eval                # Run evaluation
        """
    )
    sub = parser.add_subparsers(dest="cmd")

    # Server command
    server_parser = sub.add_parser("server", help="Run the web server")
    server_parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    server_parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    server_parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")

    # Index command
    index_parser = sub.add_parser("index", help="Index a PDF/JSON/CSV file")
    index_parser.add_argument("path", type=str)
    index_parser.add_argument("--no-classify", action="store_true")
    index_parser.add_argument("--ner", action="store_true")

    # Eval command
    sub.add_parser("eval", help="Run evaluation suite")

    args = parser.parse_args()

    if args.cmd == "server":
        run_server(
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser
        )
    elif args.cmd == "index":
        asyncio.run(index_pdf(args.path, classify=not args.no_classify, use_ner=args.ner))
    elif args.cmd == "eval":
        asyncio.run(run_eval())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()