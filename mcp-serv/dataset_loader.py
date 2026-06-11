import json
import csv
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_parser import pdf_to_document
from chunking import chunk_documents
from embeddings import get_embeddings
from vector_db import db


async def load_json_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse JSON dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Try to find array in common keys
        for key in ['data', 'items', 'records', 'documents']:
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    return []


async def load_csv_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse CSV dataset"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(dict(row))
    return data


async def load_pdf_document(file_path: str) -> str:
    """Load and parse PDF document"""
    doc = pdf_to_document(file_path)
    return doc.get('text', '')


async def index_dataset(
    file_path: str,
    dataset_type: str = None,
    classify_chunks: bool = False,
    clean_text: bool = True,
    use_ner: bool = False
) -> Dict[str, Any]:
    """
    Index a dataset (PDF, JSON, or CSV) into the knowledge base
    
    Args:
        file_path: Path to the file
        dataset_type: Type of dataset ('pdf', 'json', 'csv'). Auto-detect if None
        classify_chunks: Whether to classify chunks
        clean_text: Whether to clean text
        use_ner: Whether to use NER for entity recognition
    
    Returns:
        Dictionary with indexing results
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect dataset type
    if dataset_type is None:
        dataset_type = file_path.suffix.lstrip('.').lower()
    
    result = {
        "file": str(file_path),
        "type": dataset_type,
        "status": "processing",
        "chunks_indexed": 0,
        "error": None
    }
    
    try:
        documents = []
        
        if dataset_type == 'pdf':
            # Load and chunk PDF
            text = await load_pdf_document(str(file_path))
            documents = [{'text': text, 'source': file_path.name}]
            documents = chunk_documents(
                documents,
                clean_text=clean_text,
                use_ner=use_ner
            )
        
        elif dataset_type == 'json':
            # Load JSON and use as documents or metadata
            data = await load_json_dataset(str(file_path))
            for item in data:
                if isinstance(item, dict):
                    # If item has 'text' or 'content' field, use it as document
                    if 'text' in item:
                        documents.append({**item, 'source': file_path.name})
                    elif 'content' in item:
                        item['text'] = item.pop('content')
                        documents.append({**item, 'source': file_path.name})
                    else:
                        # Use JSON string as text
                        documents.append({
                            'text': json.dumps(item),
                            'metadata': item,
                            'source': file_path.name
                        })
                else:
                    documents.append({'text': str(item), 'source': file_path.name})
            
            # Chunk documents
            documents = chunk_documents(
                documents,
                clean_text=clean_text,
                use_ner=use_ner
            )
        
        elif dataset_type == 'csv':
            # Load CSV data
            data = await load_csv_dataset(str(file_path))
            for row in data:
                # Try to find a text column
                text_content = None
                text_columns = ['text', 'content', 'description', 'summary']
                
                for col in text_columns:
                    if col in row:
                        text_content = row[col]
                        break
                
                if not text_content:
                    # Use all columns as text
                    text_content = ' | '.join([f"{k}: {v}" for k, v in row.items()])
                
                documents.append({
                    'text': text_content,
                    'metadata': {k: v for k, v in row.items() if k != 'text' and k != 'content'},
                    'source': file_path.name
                })
            
            # Chunk documents
            documents = chunk_documents(
                documents,
                clean_text=clean_text,
                use_ner=use_ner
            )
        
        else:
            raise ValueError(f"Unsupported file type: {dataset_type}")
        
        # Classify if requested
        if classify_chunks:
            from classifier import classify_chunks_batch
            texts = [d.get('text', '') for d in documents]
            classifications = await classify_chunks_batch(texts)
            for doc, cls in zip(documents, classifications):
                doc.update(cls)
        
        # Generate embeddings and index
        texts = [d.get('text', '') for d in documents]
        if texts:
            vectors = await get_embeddings(texts)
            await db.ensure_collection_exists()
            await db.batch_insert(vectors, documents)
        
        result['status'] = 'success'
        result['chunks_indexed'] = len(documents)
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    
    return result


async def index_multiple_datasets(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Index multiple datasets concurrently"""
    tasks = [index_dataset(path) for path in file_paths]
    return await asyncio.gather(*tasks)
