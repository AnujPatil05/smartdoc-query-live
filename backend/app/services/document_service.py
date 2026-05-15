# backend/app/services/document_service.py
import PyPDF2
from PyPDF2.errors import PdfReadError
import io
import re
import tiktoken
from typing import List, Dict
import hashlib
import json
import asyncio
import httpx

from app.core.config import settings
from app.core.database import database
from app.core.redis import get_redis

# Gemini REST API endpoint (v1beta supports text-embedding-004)
_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
encoder = tiktoken.encoding_for_model("gpt-4")


class DocumentService:
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """Create a SHA256 hash of the text for Redis caching keys"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def vector_literal(values: List[float]) -> str:
        """Format an embedding as a pgvector literal."""
        return "[" + ",".join(str(float(value)) for value in values) + "]"
    
    @staticmethod
    async def process_document(file_content: bytes, filename: str, document_id: str) -> Dict:
        """Main document processing pipeline"""
        
        try:
            print(f"Starting processing for document {document_id}: {filename}")
            
            # Extract text from PDF
            pages = await DocumentService.extract_text_from_pdf(file_content)
            print(f"   Extracted {len(pages)} pages")
            
            # Clean text
            cleaned_pages = []
            for page in pages:
                cleaned_text = DocumentService.clean_text(page['text'])
                if cleaned_text.strip():  # Skip empty pages
                    cleaned_pages.append({
                        'page_num': page['page_num'],
                        'text': cleaned_text
                    })
            print(f"   Cleaned {len(cleaned_pages)} pages with content")
            
            # Chunk text
            all_chunks = []
            for page in cleaned_pages:
                page_chunks = DocumentService.chunk_text(
                    text=page['text'],
                    page_num=page['page_num']
                )
                for chunk in page_chunks:
                    chunk['chunk_index'] = len(all_chunks)
                    all_chunks.append(chunk)
            print(f"   Created {len(all_chunks)} chunks")
            if not all_chunks:
                raise ValueError("No extractable text was found in this PDF")
            
            # Generate embeddings
            print(f"   Generating embeddings with Gemini...")
            embedded_chunks = await DocumentService.generate_embeddings_batch(all_chunks)
            print(f"   Generated {len(embedded_chunks)} embeddings")
            
            # Store in database
            await DocumentService.store_chunks(document_id, embedded_chunks)
            print(f"   Stored chunks in database")
            
            # Update document status
            await database.execute(
                """
                UPDATE documents 
                SET processing_status = 'completed'
                WHERE id = :document_id
                """,
                {"document_id": document_id}
            )
            
            print(f"Document {document_id} processing completed: {len(embedded_chunks)} chunks")
            
            return {
                'total_chunks': len(embedded_chunks),
                'status': 'completed'
            }
            
        except Exception as e:
            print(f"Error processing document {document_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Update document status to failed
            try:
                await database.execute(
                    """
                    UPDATE documents 
                    SET processing_status = 'failed'
                    WHERE id = :document_id
                    """,
                    {"document_id": document_id}
                )
            except Exception:
                pass
            
            raise
    
    @staticmethod
    async def extract_text_from_pdf(file_content: bytes) -> List[Dict]:
        """Extract text from PDF with page numbers"""
        pdf_file = io.BytesIO(file_content)
        try:
            reader = PyPDF2.PdfReader(pdf_file)
        except PdfReadError as exc:
            raise ValueError("The uploaded file is not a readable PDF") from exc

        if reader.is_encrypted:
            try:
                decrypt_result = reader.decrypt("")
            except Exception as exc:
                raise ValueError("Encrypted PDFs are not supported") from exc
            if decrypt_result == 0:
                raise ValueError("Encrypted PDFs are not supported")
        
        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append({
                'page_num': page_num,
                'text': text
            })
        
        return pages
    
    @staticmethod
    def clean_text(raw_text: str) -> str:
        """Remove noise from PDF text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', raw_text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
        
        # Remove page numbers and headers/footers (heuristic)
        lines = text.split('\n')
        filtered_lines = []
        for i, line in enumerate(lines):
            line = line.strip()
            word_count = len(line.split())
            
            # Skip likely headers/footers
            if word_count < 3 and i < 2:
                continue
            if word_count < 3 and i > len(lines) - 3:
                continue
            
            filtered_lines.append(line)
        
        text = ' '.join(filtered_lines)
        return text.strip()
    
    @staticmethod
    def chunk_text(text: str, page_num: int) -> List[Dict]:
        """Split text into chunks with overlap"""
        chars_per_token = 4
        chunk_chars = settings.CHUNK_SIZE * chars_per_token
        overlap_chars = settings.CHUNK_OVERLAP * chars_per_token
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_chars
            chunk_text = text[start:end]
            
            # Don't create tiny chunks at the end
            if len(chunk_text) < 100 and chunks:
                chunks[-1]['text'] += ' ' + chunk_text
                break
            
            token_count = len(encoder.encode(chunk_text))
            
            chunks.append({
                'chunk_index': chunk_index,
                'text': chunk_text,
                'page_number': page_num,
                'char_count': len(chunk_text),
                'token_count': token_count
            })
            
            start = end - overlap_chars
            chunk_index += 1
        
        return chunks
    
    @staticmethod
    async def generate_embedding(text: str, max_retries: int = 3) -> List[float]:
        """Generate embedding via Gemini REST API with retry logic"""
        payload = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_DOCUMENT",
            "outputDimensionality": 768, 
        }
        params = {"key": settings.GOOGLE_API_KEY}

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        _EMBED_URL,
                        json=payload,
                        params=params,
                    )
                    if response.status_code == 429:
                        wait_time = (2 ** attempt) * 10
                        print(f"   Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    response.raise_for_status()
                    data = response.json()
                    values = data["embedding"]["values"]
                    # Truncate/pad to match DB dimension
                    dim = settings.EMBEDDING_DIMENSION
                    if len(values) > dim:
                        values = values[:dim]
                    return values
            except httpx.HTTPStatusError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)
                else:
                    raise Exception(f"Embedding API error: {e.response.status_code} {e.response.text}")

        raise Exception(f"Failed to generate embedding after {max_retries} retries")
    
    @staticmethod
    async def generate_embeddings_batch(chunks: List[Dict], batch_size: int = 20) -> List[Dict]:
        """Generate embeddings with caching"""
        results = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk['text'] for chunk in batch]
            
            # Check cache
            embeddings = []
            for text in texts:
                cached = await DocumentService.get_cached_embedding(text)
                if cached:
                    embeddings.append(cached)
                else:
                    embeddings.append(None)
            
            # Generate uncached embeddings
            uncached_indices = [j for j, emb in enumerate(embeddings) if emb is None]
            
            if uncached_indices:
                for idx in uncached_indices:
                    text = texts[idx]
                    # Generate embedding using Gemini
                    embedding = await DocumentService.generate_embedding(text)
                    embeddings[idx] = embedding
                    # Cache embedding
                    await DocumentService.cache_embedding(text, embedding)
                    # Small delay to avoid rate limits (free tier: ~100 requests/minute)
                    await asyncio.sleep(0.7)
            
            # Add embeddings to chunks
            for j, chunk in enumerate(batch):
                chunk['embedding'] = embeddings[j]
            
            results.extend(batch)
        
        return results
    
    @staticmethod
    async def get_cached_embedding(text: str):
        """Check Redis for cached embedding"""
        redis_client = await get_redis()
        if redis_client is None:
            return None

        model_name = settings.EMBEDDING_MODEL.replace("/", "-")
        cache_key = f"embedding:v2:{model_name}:768:{DocumentService._hash_text(text)}"
        cached = await redis_client.get(cache_key)
        if cached is None:
            return None

        return json.loads(cached)
    
    @staticmethod
    async def cache_embedding(text: str, embedding: List[float]):
        """Store embedding in Redis cache if Redis is available"""
        redis_client = await get_redis()
        if redis_client is None:
            return

        model_name = settings.EMBEDDING_MODEL.replace("/", "-")
        cache_key = f"embedding:v2:{model_name}:768:{DocumentService._hash_text(text)}"
        await redis_client.setex(
            cache_key,
            settings.EMBEDDING_CACHE_TTL,
            json.dumps(embedding)
        )
    
    @staticmethod
    async def store_chunks(document_id: str, chunks: List[Dict]):
        """Store chunks in database"""
        for chunk in chunks:
            await database.execute(
                """
                INSERT INTO chunks (document_id, chunk_index, content, page_number, char_count, token_count, embedding)
                VALUES (:document_id, :chunk_index, :content, :page_number, :char_count, :token_count, CAST(:embedding AS vector))
                """,
                {
                    'document_id': document_id,
                    'chunk_index': chunk['chunk_index'],
                    'content': chunk['text'],
                    'page_number': chunk['page_number'],
                    'char_count': chunk['char_count'],
                    'token_count': chunk['token_count'],
                    'embedding': DocumentService.vector_literal(chunk['embedding'])
                }
            )
