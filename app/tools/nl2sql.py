"""
app/tools/nl2sql.py
════════════════════
Natural Language to SQL Tool.

Failure contract:
  - validation error (unsafe SQL): NOT retry eligible
  - empty result: retry eligible with simplified query
  - timeout: retry eligible with backoff
  - malformed LLM output: retry eligible once
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import sqlparse
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.schemas.context import ToolName
from app.schemas.tools import (
    NL2SQLInput, NL2SQLOutput, RetryEligibility, ToolFailureContract, ToolStatus,
)
from app.tools.base import BaseTool

# Allowlist of safe SQL statement types
_ALLOWED_SQL_TYPES = {"SELECT"}
# Patterns that indicate SQL injection attempts
_INJECTION_PATTERNS = [
    r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)",
    r"--\s*$",
    r"/\*.*\*/",
    r"xp_cmdshell",
    r"EXEC\s*\(",
]

_SCHEMA_HINT = """
Tables available:
  - execution_traces(id, query_id, user_query, status, final_answer, created_at)
  - tool_call_logs(id, trace_id, tool_name, agent_id, status, latency_ms, timestamp)
  - eval_runs(id, triggered_by, total_cases, passed_cases, avg_total_score, started_at)
  - eval_case_records(id, run_id, case_id, category, total_score, passed, timestamp)
  - prompt_versions(id, agent_id, version, status, created_at)
"""


class NL2SQLTool(BaseTool):
    """
    Converts natural language queries to validated SQL and executes them
    against the local PostgreSQL database.
    """

    def __init__(self, db_session: AsyncSession | None = None):
        self._db = db_session

    @property
    def tool_name(self) -> ToolName:
        return ToolName.NL2SQL

    @property
    def failure_contract(self) -> ToolFailureContract:
        return ToolFailureContract(
            tool_name="nl2sql",
            timeout_secs=15,
            timeout_retry_eligible=True,
            malformed_input_retry_eligible=True,
            empty_result_retry_eligible=True,
            rate_limit_retry_eligible=False,
            max_retries=settings.tool_max_retries,
            fallback_strategy="skip",
        )

    def _validate_input(self, input_data: NL2SQLInput) -> str | None:
        if not input_data.natural_language_query.strip():
            return "Natural language query must not be empty"
        return None

    def _validate_sql(self, sql: str) -> str | None:
        """Validate generated SQL for safety. Returns error string or None."""
        try:
            parsed = sqlparse.parse(sql)
        except Exception as e:
            return f"SQL parse error: {e}"

        if not parsed:
            return "Generated SQL is empty"

        stmt_type = parsed[0].get_type()
        if stmt_type not in _ALLOWED_SQL_TYPES:
            return f"Only SELECT statements are permitted, got: {stmt_type}"

        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, sql, re.IGNORECASE | re.DOTALL):
                return f"SQL contains disallowed pattern: {pattern}"

        return None

    async def _generate_sql(self, nl_query: str, schema_hint: str) -> str:
        """Use Gemini to generate SQL from natural language."""
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.0,
        )
        prompt = (
            f"Generate a safe PostgreSQL SELECT query for the following question.\n"
            f"Schema:\n{schema_hint}\n\n"
            f"Question: {nl_query}\n\n"
            f"Return ONLY the SQL query, no explanation, no markdown."
        )
        response = await llm.ainvoke(prompt)
        sql = response.content.strip()
        # Strip markdown code fences if present
        sql = re.sub(r"^```(?:sql)?\n?", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\n?```$", "", sql)
        return sql.strip()

    async def _execute(self, input_data: NL2SQLInput) -> NL2SQLOutput:
        schema_hint = input_data.target_schema or _SCHEMA_HINT

        # Generate SQL
        try:
            sql = await self._generate_sql(input_data.natural_language_query, schema_hint)
        except Exception as e:
            return NL2SQLOutput(
                tool_name="nl2sql",
                status=ToolStatus.MALFORMED_OUTPUT,
                retry_eligible=RetryEligibility.ELIGIBLE,
                error_message=f"SQL generation failed: {e}",
            )

        # Validate SQL
        sql_error = self._validate_sql(sql)
        if sql_error:
            return NL2SQLOutput(
                tool_name="nl2sql",
                status=ToolStatus.VALIDATION_ERROR,
                retry_eligible=RetryEligibility.NOT_ELIGIBLE,
                generated_sql=sql,
                error_message=sql_error,
            )

        # Execute SQL
        if self._db is None:
            # Return SQL without execution if no DB session provided
            return NL2SQLOutput(
                tool_name="nl2sql",
                status=ToolStatus.SUCCESS,
                generated_sql=sql,
                validated=True,
                sql_validated=True,
                raw_output={"sql": sql, "note": "no_db_session"},
            )

        try:
            result = await self._db.execute(
                text(sql + f" LIMIT {input_data.max_rows}")
            )
            rows: List[Dict[str, Any]] = [dict(row._mapping) for row in result]
            columns = list(result.keys()) if result.keys() else []
        except Exception as e:
            return NL2SQLOutput(
                tool_name="nl2sql",
                status=ToolStatus.EXECUTION_ERROR,
                retry_eligible=RetryEligibility.ELIGIBLE,
                generated_sql=sql,
                validated=True,
                error_message=f"SQL execution error: {e}",
            )

        return NL2SQLOutput(
            tool_name="nl2sql",
            status=ToolStatus.SUCCESS if rows else ToolStatus.EMPTY_RESULT,
            retry_eligible=RetryEligibility.ELIGIBLE if not rows else RetryEligibility.NOT_ELIGIBLE,
            generated_sql=sql,
            validated=True,
            sql_validated=True,
            result_rows=rows,
            row_count=len(rows),
            columns=columns,
            raw_output={"sql": sql, "rows": rows},
        )
