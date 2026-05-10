"""
app/api/routes/evals.py
═══════════════════════
Endpoint 3: GET /api/v1/evals/latest — Latest eval summary
Endpoint 5: POST /api/v1/evals/rerun — Targeted re-eval
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import EvalRunRecord
from app.database.session import get_db
from app.evals.cases import EVAL_CASES, get_case_by_id
from app.evals.runner import EvalRunner
from app.evals.scorer import OrquestraScorer

router = APIRouter(prefix="/api/v1/evals", tags=["Evaluations"])


class RerunRequest(BaseModel):
    case_ids: List[str]
    triggered_by: str = "manual"


@router.get(
    "/latest",
    summary="Retrieve latest evaluation summary",
    description=(
        "Returns the most recent evaluation run with per-case scores across "
        "all six dimensions (correctness, citation_accuracy, contradiction_handling, "
        "tool_efficiency, context_compliance, critique_agreement). "
        "Each score includes a written justification."
    ),
)
async def get_latest_evals(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(EvalRunRecord)
        .order_by(EvalRunRecord.started_at.desc())
        .limit(1)
    )
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "NO_EVAL_RUNS",
                "message": "No evaluation runs found. Run POST /api/v1/evals/rerun to start one.",
            },
        )

    return {
        "run_id": run.id,
        "triggered_by": run.triggered_by,
        "total_cases": run.total_cases,
        "passed_cases": run.passed_cases,
        "avg_total_score": run.avg_total_score,
        "summary": run.summary,
        "started_at": run.started_at.isoformat(),
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }


@router.post(
    "/rerun",
    summary="Trigger targeted re-evaluation",
    description=(
        "Re-runs evaluation for specific case IDs. "
        "Used after a prompt rewrite is approved to validate improvement. "
        "Results are stored and comparable against previous runs."
    ),
)
async def rerun_evals(
    request: RerunRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    # Validate case IDs
    invalid = [cid for cid in request.case_ids if not get_case_by_id(cid)]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_CASE_IDS",
                "message": f"Unknown case IDs: {invalid}",
                "detail": f"Valid IDs: {[c.case_id for c in EVAL_CASES]}",
            },
        )

    runner = EvalRunner(db=db, scorer=OrquestraScorer())
    background_tasks.add_task(
        runner.run_cases,
        request.case_ids,
        request.triggered_by,
    )

    return {
        "status": "accepted",
        "message": f"Re-evaluation started for {len(request.case_ids)} case(s).",
        "case_ids": request.case_ids,
    }
