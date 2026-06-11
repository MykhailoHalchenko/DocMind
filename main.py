import asyncio
import argparse
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description="Knowledge Base CLI")
    sub = parser.add_subparsers(dest="cmd")

    index_parser = sub.add_parser("index", help="Index a PDF file")
    index_parser.add_argument("path", type=str)
    index_parser.add_argument("--no-classify", action="store_true")
    index_parser.add_argument("--ner", action="store_true")

    sub.add_parser("eval", help="Run evaluation suite")

    args = parser.parse_args()

    if args.cmd == "index":
        asyncio.run(index_pdf(args.path, classify=not args.no_classify, use_ner=args.ner))
    elif args.cmd == "eval":
        asyncio.run(run_eval())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()