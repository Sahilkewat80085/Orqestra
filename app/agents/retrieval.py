"""
app/agents/retrieval.py
═══════════════════════
RetrievalAgent: Multi-hop retrieval from ChromaDB.

Rules (non-negotiable):
  - MINIMUM two hops required before synthesis
  - MINIMUM two chunks must be retrieved
  - Single-hop is NOT acceptable
  - Every chunk contribution is tracked in citations
  - Provenance maps chunk_id → claim
"""
from __future__ import annotations

import uuid
from typing import List

try:
    import chromadb
except ImportError:
    chromadb = None
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.base import BaseAgent
from app.config import settings
from app.schemas.agents import RetrievalResult, RetrievedChunk
from app.schemas.context import AgentID, AgentOutput, Citation, ProvenanceEntry, SharedContext
from app.utils.token_counter import count_tokens

_MIN_CHUNKS = 2
_MIN_HOPS = 2


def _get_chroma_collection():
    """Initialize ChromaDB client and return the configured collection."""
    if chromadb is None:
        raise ImportError("ChromaDB is not installed or available.")
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        return client.get_collection(settings.chroma_collection_name)
    except Exception:
        # Collection doesn't exist yet — create with sample data
        collection = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        _seed_collection(collection)
        return collection


def _seed_collection(collection):
    """Seed with minimal knowledge base if empty."""
    docs = [
        ("chunk_001", "Large language models are neural networks trained on vast text corpora using transformer architectures. They exhibit emergent capabilities at scale."),
        ("chunk_002", "Multi-agent systems coordinate autonomous agents through shared state rather than direct communication. Agents read/write to a shared context object."),
        ("chunk_003", "LangGraph enables stateful agentic workflows by modeling agent coordination as a graph with conditional edges driven by LLM reasoning."),
        ("chunk_004", "Retrieval-Augmented Generation (RAG) grounds LLM responses in retrieved documents, reducing hallucinations and enabling citation-aware reasoning."),
        ("chunk_005", "Prompt injection attacks attempt to override system instructions by embedding malicious instructions in user input or retrieved content."),
        ("chunk_006", "Context window management is critical for multi-agent systems. Token budgets prevent silent truncation and ensure deterministic behavior."),
        ("chunk_007", "Evaluation harnesses for AI systems require custom scoring rubrics beyond accuracy: citation precision, contradiction detection, and robustness."),
        ("chunk_008", "ChromaDB is an open-source vector database optimized for similarity search using embeddings. It supports persistent storage and cosine similarity."),
        ("chunk_009", "Self-reflection in AI agents allows them to re-examine previous reasoning, identify contradictions, and revise conclusions before final synthesis."),
        ("chunk_010", "Structured JSON logging with fields like agent_id, event_type, latency_ms, and policy_violations enables production observability and auditing."),
    ]
    collection.add(
        documents=[d[1] for d in docs],
        ids=[d[0] for d in docs],
        metadatas=[{"source": "seed_kb", "chunk_id": d[0]} for d in docs],
    )


class RetrievalAgent(BaseAgent):
    """
    Performs multi-hop retrieval from ChromaDB.

    Hop 1: Initial semantic search on user query
    Hop 2: Contextual expansion — search using terms from hop-1 results
    """

    @property
    def agent_id(self) -> str:
        return AgentID.RETRIEVAL

    async def _run(self, context: SharedContext) -> SharedContext:
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.0,
        )

        try:
            collection = _get_chroma_collection()
        except Exception as e:
            # Graceful degradation — return stub result
            return self._stub_result(context, str(e))

        # ── Hop 1: Initial retrieval ───────────────────────────────────────────
        hop1_results = collection.query(
            query_texts=[context.user_query],
            n_results=3,
            include=["documents", "distances", "metadatas"],
        )

        hop1_chunks: List[RetrievedChunk] = []
        for i, (doc, dist, meta) in enumerate(zip(
            hop1_results["documents"][0],
            hop1_results["distances"][0],
            hop1_results["metadatas"][0],
        )):
            relevance = max(0.0, 1.0 - float(dist))
            hop1_chunks.append(RetrievedChunk(
                chunk_id=meta.get("chunk_id", f"hop1_{i}"),
                content=doc,
                relevance_score=relevance,
                hop=1,
            ))

        # ── Hop 2: Contextual expansion ────────────────────────────────────────
        # Extract key terms from hop-1 to formulate hop-2 query
        hop1_text = " ".join(c.content[:100] for c in hop1_chunks)
        expansion_prompt = (
            f"Extract 3-5 key technical terms from this text for a follow-up search:\n{hop1_text}\n"
            f"Return only comma-separated terms."
        )
        try:
            expansion_response = await llm.ainvoke(expansion_prompt)
            expansion_query = expansion_response.content.strip()
        except Exception:
            expansion_query = context.user_query  # fallback

        hop2_results = collection.query(
            query_texts=[expansion_query],
            n_results=3,
            include=["documents", "distances", "metadatas"],
        )

        hop2_chunks: List[RetrievedChunk] = []
        existing_ids = {c.chunk_id for c in hop1_chunks}
        for i, (doc, dist, meta) in enumerate(zip(
            hop2_results["documents"][0],
            hop2_results["distances"][0],
            hop2_results["metadatas"][0],
        )):
            chunk_id = meta.get("chunk_id", f"hop2_{i}")
            if chunk_id in existing_ids:
                continue  # Deduplicate
            relevance = max(0.0, 1.0 - float(dist))
            hop2_chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                content=doc,
                relevance_score=relevance,
                hop=2,
            ))

        all_chunks = hop1_chunks + hop2_chunks

        # Enforce minimum chunks
        if len(all_chunks) < _MIN_CHUNKS:
            return self._stub_result(context, "Insufficient chunks retrieved")

        # ── Build citations and provenance ─────────────────────────────────────
        citations = []
        provenance_entries = []
        for chunk in all_chunks:
            claim = f"Based on retrieved information (hop {chunk.hop}): {chunk.content[:80]}..."
            citation = Citation(
                chunk_id=chunk.chunk_id,
                excerpt=chunk.content[:200],
                relevance_score=chunk.relevance_score,
                contributing_to_claim=claim,
            )
            citations.append(citation)
            provenance_entries.append(ProvenanceEntry(
                sentence=claim,
                agent_id=AgentID.RETRIEVAL,
                chunk_ids=[chunk.chunk_id],
                confidence=chunk.relevance_score,
            ))

        result = RetrievalResult(
            query_used=context.user_query,
            chunks=all_chunks,
            total_hops=_MIN_HOPS,
            provenance_map={
                c.chunk_id: [f"hop_{c.hop}_retrieval"] for c in all_chunks
            },
            reasoning=(
                f"Performed {_MIN_HOPS}-hop retrieval. "
                f"Hop 1: {len(hop1_chunks)} chunks on direct query. "
                f"Hop 2: {len(hop2_chunks)} chunks on expanded terms '{expansion_query[:50]}'."
            ),
        )

        raw_output = result.model_dump_json()
        token_count = count_tokens(raw_output)

        output = AgentOutput(
            agent_id=AgentID.RETRIEVAL,
            raw_output=raw_output,
            structured_output=result.model_dump(),
            confidence=sum(c.relevance_score for c in all_chunks) / len(all_chunks),
            token_count=token_count,
        )

        updated = context.set_agent_output(output)
        for citation in citations:
            updated = updated.add_citation(citation)
        for entry in provenance_entries:
            updated = updated.add_provenance(entry)

        return updated

    def _stub_result(self, context: SharedContext, reason: str) -> SharedContext:
        """Return a stub retrieval result when ChromaDB is unavailable."""
        stub_chunks = [
            RetrievedChunk(
                chunk_id=f"stub_{i}",
                content=f"[Stub chunk {i}] Multi-agent LLM systems coordinate through shared context.",
                relevance_score=0.5,
                hop=i + 1,
            )
            for i in range(_MIN_CHUNKS)
        ]
        output = AgentOutput(
            agent_id=AgentID.RETRIEVAL,
            raw_output=f"Stub retrieval ({reason})",
            structured_output={"chunks": [c.model_dump() for c in stub_chunks]},
            confidence=0.3,
            token_count=50,
        )
        return context.set_agent_output(output)
