"""
app/evals/runner.py
════════════════════
EvalRunner: Executes eval cases through the full agent pipeline.
Captures all tool calls, agent outputs, and context snapshots per case.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import EvalCaseRecord, EvalRunRecord
from app.evals.cases import EVAL_CASES
from app.evals.scorer import OrquestraScorer
from app.logging.logger import get_logger
from app.orchestrator.graph import graph
from app.schemas.context import SharedContext
from app.schemas.evals import EvalCaseResult

log = get_logger("evals.runner")

_CASES_BY_ID = {c.case_id: c for c in EVAL_CASES}


class EvalRunner:
    def __init__(self, db: AsyncSession, scorer: OrquestraScorer):
        self._db = db
        self._scorer = scorer

    async def run_cases(
        self,
        case_ids: list[str] | None = None,
        triggered_by: str = "manual",
    ) -> str:
        """
        Run evaluation for specified cases (or all if None).
        Returns the run_id.
        """
        if case_ids is not None:
            cases = [_CASES_BY_ID[cid] for cid in case_ids if cid in _CASES_BY_ID]
        else:
            cases = EVAL_CASES

        run = EvalRunRecord(
            triggered_by=triggered_by,
            case_ids=[c.case_id for c in cases],
            total_cases=len(cases),
        )
        self._db.add(run)
        await self._db.flush()

        results: list[EvalCaseResult] = []
        records: list[EvalCaseRecord] = []
        passed = 0

        for case in cases:
            log.info("eval_case_started", case_id=case.case_id)
            start = time.perf_counter()

            # Run through full pipeline
            final_answer: str | None = None
            tool_calls_used: list[str] = []
            agent_outputs_snapshot: dict = {}
            error_msg: str | None = None

            try:
                ctx = SharedContext(user_query=case.query)
                state = await graph.ainvoke({"context": ctx.model_dump(mode="json")})
                final_ctx = SharedContext.model_validate(state["context"])

                final_answer = final_ctx.final_answer
                tool_calls_used = list({tc.tool_name for tc in final_ctx.tool_calls})
                agent_outputs_snapshot = {
                    k: v.model_dump(mode="json")
                    for k, v in final_ctx.agent_outputs.items()
                }
            except Exception as e:
                error_msg = str(e)

            elapsed_ms = (time.perf_counter() - start) * 1000

            temp_result = EvalCaseResult(
                case_id=case.case_id,
                query=case.query,
                category=case.category,
                final_answer=final_answer,
                tool_calls_used=tool_calls_used,
                agent_outputs_snapshot=agent_outputs_snapshot,
                error=error_msg,
                execution_time_ms=elapsed_ms,
            )

            # Score the result
            score = self._scorer.score(case, temp_result)
            result = temp_result.model_copy(update={"score": score})

            if score.passed:
                passed += 1

            results.append(result)

            # Buffer case record for batch insertion
            records.append(
                EvalCaseRecord(
                    run_id=run.id,
                    case_id=case.case_id,
                    category=case.category,
                    query=case.query,
                    final_answer=result.final_answer,
                    score=score.model_dump(mode="json"),
                    total_score=score.total_score,
                    passed=score.passed,
                    agent_outputs_snapshot=result.agent_outputs_snapshot,
                    tool_calls_used=result.tool_calls_used,
                    execution_time_ms=result.execution_time_ms,
                    error=result.error,
                )
            )
            log.info(
                "eval_case_completed",
                case_id=case.case_id,
                score=score.total_score,
                passed=score.passed,
            )

        if records:
            self._db.add_all(records)

        avg_score = (
            sum(r.score.total_score for r in results if r.score) / len(results)
            if results
            else 0.0
        )

        run.passed_cases = passed
        run.avg_total_score = avg_score
        run.completed_at = datetime.now(timezone.utc)
        run.summary = {
            "total_cases": len(cases),
            "passed": passed,
            "failed": len(cases) - passed,
            "avg_score": round(avg_score, 3),
        }

        await self._db.commit()
        log.info("eval_run_completed", run_id=run.id, passed=passed, avg_score=avg_score)
        return run.id

