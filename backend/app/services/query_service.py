import asyncio
import hashlib
import httpx
import json
import re
from typing import Dict, List, Optional

from google import genai
from google.genai import types

from app.core.config import settings
from app.core.database import database
from app.core.redis import get_redis
from app.services.document_service import DocumentService, _EMBED_URL

# Gemini LLM client
_llm_client = genai.Client(
    api_key=settings.GOOGLE_API_KEY,
    http_options={"api_version": "v1beta"},
)


class QueryService:
    @staticmethod
    async def answer_query(
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict:
        """Run the RAG pipeline for a user question."""
        cached_answer = await QueryService.get_cached_answer(
            query,
            document_ids,
            conversation_history,
        )
        if cached_answer:
            return {**cached_answer, "cache_hit": True}

        query_embedding = await QueryService.generate_query_embedding(query)
        retrieved_chunks = await QueryService.semantic_search(
            query_embedding,
            document_ids,
            top_k,
        )

        if not retrieved_chunks:
            return {
                "answer": "I couldn't find any relevant information in the documents to answer your question.",
                "citations": [],
                "has_answer": False,
                "cache_hit": False,
            }

        answer = await QueryService.generate_answer(
            query,
            retrieved_chunks,
            conversation_history,
        )
        await QueryService.cache_answer(
            query,
            document_ids,
            conversation_history,
            answer,
        )

        return {**answer, "cache_hit": False}

    @staticmethod
    async def generate_query_embedding(query: str) -> List[float]:
        """Generate and cache an embedding for a query via REST API."""
        cached = await DocumentService.get_cached_embedding(query)
        if cached:
            return cached

        payload = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": query}]},
            "taskType": "RETRIEVAL_QUERY",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                _EMBED_URL,
                json=payload,
                params={"key": settings.GOOGLE_API_KEY},
            )
            response.raise_for_status()
            values = response.json()["embedding"]["values"]

        dim = settings.EMBEDDING_DIMENSION
        if len(values) > dim:
            values = values[:dim]

        await DocumentService.cache_embedding(query, values)
        return values

    @staticmethod
    async def semantic_search(
        query_embedding: List[float],
        document_ids: Optional[List[str]],
        top_k: int,
    ) -> List[Dict]:
        """Search pgvector for chunks nearest to the query embedding."""
        values = {
            "query_embedding": DocumentService.vector_literal(query_embedding),
            "limit": max(top_k * 2, top_k),
        }

        sql = """
            SELECT
                c.id,
                c.content,
                c.page_number,
                d.title AS document_title,
                d.id AS document_id,
                1 - (c.embedding <=> CAST(:query_embedding AS vector)) AS similarity
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.processing_status = 'completed'
            AND c.embedding IS NOT NULL
        """

        if document_ids:
            placeholders = []
            for index, document_id in enumerate(document_ids):
                key = f"document_id_{index}"
                values[key] = str(document_id)
                placeholders.append(f"CAST(:{key} AS uuid)")
            sql += f" AND d.id IN ({', '.join(placeholders)})"

        sql += """
            ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
            LIMIT :limit
        """

        results = await database.fetch_all(sql, values)
        filtered = []
        for row in results:
            similarity = float(row["similarity"] or 0)
            if similarity < settings.SIMILARITY_THRESHOLD:
                continue
            filtered.append(
                {
                    "chunk_id": str(row["id"]),
                    "content": row["content"],
                    "page_number": row["page_number"],
                    "document_title": row["document_title"],
                    "document_id": str(row["document_id"]),
                    "similarity": similarity,
                }
            )

        return filtered[:top_k]

    @staticmethod
    async def generate_answer(
        query: str,
        chunks: List[Dict],
        conversation_history: Optional[List[Dict]],
    ) -> Dict:
        """Generate a grounded answer using Gemini."""
        context = QueryService.build_context(chunks)
        response = await asyncio.to_thread(
            _llm_client.models.generate_content,
            model=settings.LLM_MODEL,
            contents=QueryService.build_user_prompt(query, context, conversation_history),
            config=types.GenerateContentConfig(
                system_instruction=QueryService.get_system_prompt(),
                temperature=0.1,
                max_output_tokens=800,
                response_mime_type="application/json",
            ),
        )
        response_text = getattr(response, "text", "") or ""

        try:
            answer_json = json.loads(QueryService._strip_json_fences(response_text))
        except json.JSONDecodeError:
            return {
                "answer": response_text or "I could not generate a response for this question.",
                "citations": [],
                "has_answer": bool(response_text),
            }

        citations = QueryService.map_citations(
            answer_json.get("citations", []),
            chunks,
        )

        return {
            "answer": answer_json.get("answer") or response_text,
            "citations": citations,
            "has_answer": bool(answer_json.get("has_answer", True)),
        }

    @staticmethod
    def build_context(chunks: List[Dict]) -> str:
        """Format retrieved chunks for the LLM prompt."""
        parts = []
        for index, chunk in enumerate(chunks, start=1):
            parts.append(
                f"""[SOURCE {index}]
Document: {chunk['document_title']}
Page: {chunk['page_number']}
Content: {chunk['content']}
---"""
            )
        return "\n\n".join(parts)

    @staticmethod
    def get_system_prompt() -> str:
        return """You are an expert document analysis assistant. Answer questions based ONLY on the provided context.
Rules:
1. Only use information explicitly stated in the context.
2. Cite sources using [SOURCE X] notation.
3. If the context lacks the answer, say: "I don't have enough information in the provided documents to answer this question."
4. Respond in JSON: {"answer": "...", "has_answer": true, "citations": [1, 2]}.
5. Be concise, precise, and include useful source-backed detail when available."""

    @staticmethod
    def build_user_prompt(query: str, context: str, history: Optional[List[Dict]]) -> str:
        parts = []

        if history:
            parts.append("Previous conversation:")
            for msg in history[-3:]:
                parts.append(f"{msg['role']}: {msg['content']}")
            parts.append("---")

        parts.append(
            f"""Context:
{context}

Question: {query}

Provide a JSON response with answer, has_answer, and citations as source numbers."""
        )
        return "\n".join(parts)

    @staticmethod
    def map_citations(indices: List[int], chunks: List[Dict]) -> List[Dict]:
        """Map LLM source numbers to chunk metadata."""
        citations = []
        seen = set()

        for raw_index in indices:
            match = re.search(r"\d+", str(raw_index))
            if not match:
                continue

            index = int(match.group(0))
            if index in seen or not 0 < index <= len(chunks):
                continue

            seen.add(index)
            chunk = chunks[index - 1]
            citations.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "document_title": chunk["document_title"],
                    "page_number": chunk["page_number"],
                    "text_preview": chunk["content"][:200] + ("..." if len(chunk["content"]) > 200 else ""),
                    "similarity_score": chunk["similarity"],
                }
            )

        return citations

    @staticmethod
    async def get_cached_answer(
        query: str,
        document_ids: Optional[List[str]],
        conversation_history: Optional[List[Dict]],
    ) -> Optional[Dict]:
        """Check Redis for a cached answer, if Redis is available."""
        redis_client = await get_redis()
        if redis_client is None:
            return None

        cached = await redis_client.get(
            QueryService.build_cache_key(query, document_ids, conversation_history)
        )
        if not cached:
            return None

        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            return None

    @staticmethod
    async def cache_answer(
        query: str,
        document_ids: Optional[List[str]],
        conversation_history: Optional[List[Dict]],
        answer: Dict,
    ) -> None:
        """Cache an answer, if Redis is available."""
        redis_client = await get_redis()
        if redis_client is None:
            return

        await redis_client.setex(
            QueryService.build_cache_key(query, document_ids, conversation_history),
            settings.QUERY_CACHE_TTL,
            json.dumps(answer),
        )

    @staticmethod
    def build_cache_key(
        query: str,
        document_ids: Optional[List[str]],
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """Generate a cache key scoped to the query, documents, and recent context."""
        doc_str = ",".join(sorted(str(doc_id) for doc_id in document_ids)) if document_ids else "all"
        history_payload = [
            {
                "role": item.get("role"),
                "content": item.get("content"),
            }
            for item in (conversation_history or [])[-3:]
        ]
        content = json.dumps(
            {
                "query": query.strip(),
                "documents": doc_str,
                "history": history_payload,
            },
            sort_keys=True,
        )
        hash_key = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return f"query_cache:{hash_key}"

    @staticmethod
    def _strip_json_fences(value: str) -> str:
        value = value.strip()
        if value.startswith("```"):
            value = re.sub(r"^```(?:json)?", "", value, flags=re.IGNORECASE).strip()
            value = re.sub(r"```$", "", value).strip()
        return value
