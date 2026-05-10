#!/usr/bin/env python3
"""
scripts/export_trace.py
═══════════════════════
Export a full execution trace as JSON for auditing or diffing.
Usage:
  python scripts/export_trace.py <query_id> [output_file.json]
"""
import asyncio
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


async def export(query_id: str, output_path: str | None = None):
    from app.database.session import AsyncSessionFactory
    from app.database.repositories.trace_repository import TraceRepository

    async with AsyncSessionFactory() as session:
        repo = TraceRepository(session)
        trace = await repo.get_by_query_id(query_id)

    if not trace:
        print(f"No trace found for query_id: {query_id}", file=sys.stderr)
        sys.exit(1)

    data = {
        "query_id": trace.query_id,
        "user_query": trace.user_query,
        "status": trace.status,
        "final_answer": trace.final_answer,
        "routing_decision": trace.routing_decision,
        "agent_outputs": trace.agent_outputs,
        "tool_calls": trace.tool_calls,
        "policy_violations": trace.policy_violations,
        "total_tokens_used": trace.total_tokens_used,
        "created_at": trace.created_at.isoformat() if trace.created_at else None,
        "completed_at": trace.completed_at.isoformat() if trace.completed_at else None,
    }

    output = json.dumps(data, indent=2, default=str)
    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
        print(f"Trace exported to {output_path}")
    else:
        print(output)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_trace.py <query_id> [output.json]")
        sys.exit(1)
    asyncio.run(export(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
