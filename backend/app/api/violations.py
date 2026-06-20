"""Violation API routes."""
from fastapi import APIRouter, Request, Query, HTTPException
from typing import Optional
from app.db import repository as repo

router = APIRouter()


@router.get("/violations")
async def list_violations(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    violation_type: Optional[str] = None,
    camera_id: Optional[int] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    """List violations with filtering and pagination."""
    db = request.app.state.db
    offset = (page - 1) * per_page
    violations, total = await repo.list_violations(
        db,
        violation_type=violation_type,
        camera_id=camera_id,
        status=status,
        date_from=date_from,
        date_to=date_to,
        limit=per_page,
        offset=offset,
    )
    return {
        "data": violations,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": (total + per_page - 1) // per_page,
        },
    }


@router.get("/violations/stats")
async def get_stats(request: Request):
    """Get violation statistics."""
    db = request.app.state.db
    stats = await repo.violation_stats(db)
    return stats


@router.get("/violations/{violation_id}")
async def get_violation(request: Request, violation_id: int):
    """Get a single violation by row ID."""
    db = request.app.state.db
    violation = await repo.get_violation(db, violation_id)
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")
    return violation


@router.patch("/violations/{violation_id}")
async def update_violation(request: Request, violation_id: str, body: dict):
    """Update violation status by UUID."""
    db = request.app.state.db
    new_status = body.get("status")
    if new_status not in ("pending", "confirmed", "dismissed"):
        raise HTTPException(status_code=400, detail="Invalid status")
    result = await repo.update_violation_status(db, violation_id, new_status)
    if not result:
        raise HTTPException(status_code=404, detail="Violation not found")
    return {"violation_id": violation_id, "status": new_status}


@router.get("/violations/{violation_id}/certificate")
async def generate_certificate(request: Request, violation_id: int):
    """Generate BSA Section 63(4)(c) certificate."""
    db = request.app.state.db
    violation = await repo.get_violation(db, violation_id)
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")

    certificate = f"""==========================================================================================
            CERTIFICATE UNDER SECTION 63(4)(c) OF THE BHARATIYA SAKSHYA ADHINIYAM, 2023
==========================================================================================

PART A: TO BE COMPLETED BY THE SYSTEM OPERATOR
------------------------------------------------------------------------------------------
I, [Name of Officer], do hereby solemnly affirm and sincerely state as follows:

1. I am lawfully entitled to control the operations of the automated traffic enforcement
   computer network, incorporating Edge Camera Node ID: {violation.get('camera_id', 'N/A')}.
2. The digital device/system was under lawful control for regularly creating, storing, and
   processing information for traffic safety enforcement.
3. The computer system and edge processing nodes were operating properly.
4. The cryptographic SHA-256 hash value of the evidentiary image is:
   {violation.get('sha256_hash', 'N/A')}

Violation Details:
- Violation ID: {violation.get('violation_id', 'N/A')}
- Violation Type: {violation.get('violation_type', 'N/A')}
- Violation Time: {violation.get('violation_time', 'N/A')}
- Vehicle: {violation.get('vehicle_class', 'N/A')}
- License Plate: {violation.get('license_plate', 'N/A')}
- Confidence: {violation.get('confidence', 'N/A')}
- Camera: {violation.get('camera_id', 'N/A')}

Name & Signature of Party:                             Date & Time:
------------------------------------------------------------------------------------------

PART B: TO BE COMPLETED BY THE DIGITAL FORENSICS EXPERT
------------------------------------------------------------------------------------------
I, [Expert Name], do hereby certify that:

1. I have technically verified the output of the digital record source.
2. I have mathematically validated that the SHA-256 hash of the generated record matches
   the recorded value: {violation.get('sha256_hash', 'N/A')}
3. System diagnostic logs show no unauthorized alteration during the timestamp period.

Name, Designation & Signature:                          Date & Place:
=========================================================================================="""

    return {"certificate": certificate, "violation_id": violation_id}
