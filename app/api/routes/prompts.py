"""
app/api/routes/prompts.py
══════════════════════════
Endpoint 4: POST /api/v1/prompts/approve — Approve or reject prompt rewrite
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import PromptVersionRecord
from app.database.session import get_db
from app.logging.logger import get_logger

router = APIRouter(prefix="/api/v1/prompts", tags=["Prompts"])
log = get_logger("api.prompts")


class ApprovalRequest(BaseModel):
    version_id: str
    action: str          # "approve" or "reject"
    approved_by: str     # Human reviewer identifier
    reason: Optional[str] = None


from typing import Optional


@router.post(
    "/approve",
    summary="Approve or reject a pending prompt rewrite",
    description=(
        "Approves or rejects a MetaAgent-proposed prompt rewrite. "
        "Approved rewrites become active on next agent invocation. "
        "Rejected rewrites are archived with the rejection reason. "
        "ALL changes are fully auditable — nothing is auto-applied."
    ),
)
async def approve_prompt(
    request: ApprovalRequest,
    db: AsyncSession = Depends(get_db),
):
    if request.action not in ("approve", "reject"):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_ACTION",
                "message": "action must be 'approve' or 'reject'",
            },
        )

    result = await db.execute(
        select(PromptVersionRecord).where(
            PromptVersionRecord.id == request.version_id
        )
    )
    record = result.scalar_one_or_none()

    if not record:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "PROMPT_VERSION_NOT_FOUND",
                "message": f"No prompt version found with id: {request.version_id}",
            },
        )

    if record.status != "pending":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "PROMPT_NOT_PENDING",
                "message": f"Prompt is in status '{record.status}', not 'pending'.",
            },
        )

    from datetime import datetime

    if request.action == "approve":
        # Deprecate previous active version for this agent
        await db.execute(
            update(PromptVersionRecord)
            .where(
                PromptVersionRecord.agent_id == record.agent_id,
                PromptVersionRecord.status == "active",
            )
            .values(status="deprecated")
        )
        record.status = "active"
        record.approved_by = request.approved_by
        record.approved_at = datetime.utcnow()

        log.info(
            "prompt_approved",
            version_id=request.version_id,
            agent_id=record.agent_id,
            approved_by=request.approved_by,
        )
        message = f"Prompt version {request.version_id} approved and activated."

    else:
        record.status = "rejected"
        record.approved_by = request.approved_by
        if request.reason:
            record.rewrite_reasoning = f"REJECTED: {request.reason}"

        log.info(
            "prompt_rejected",
            version_id=request.version_id,
            agent_id=record.agent_id,
            rejected_by=request.approved_by,
        )
        message = f"Prompt version {request.version_id} rejected."

    await db.commit()

    return {
        "version_id": request.version_id,
        "agent_id": record.agent_id,
        "status": record.status,
        "message": message,
        "actioned_by": request.approved_by,
    }


@router.get(
    "/pending",
    summary="List pending prompt rewrites awaiting approval",
)
async def list_pending_prompts(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(PromptVersionRecord).where(PromptVersionRecord.status == "pending")
    )
    records = result.scalars().all()

    return {
        "pending_count": len(records),
        "rewrites": [
            {
                "version_id": r.id,
                "agent_id": r.agent_id,
                "version": r.version,
                "rewrite_reasoning": r.rewrite_reasoning,
                "failed_case_ids": r.failed_case_ids,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ],
    }
