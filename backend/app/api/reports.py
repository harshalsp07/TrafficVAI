"""Reports and Analytics API router."""
from fastapi import APIRouter, Request, Query, Response, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import io
from app.services.report_service import ReportService

router = APIRouter()
report_service = ReportService()

@router.get("/reports/summary")
async def get_summary(request: Request):
    """Get aggregated traffic violation analytics for dashboard visualization."""
    db = request.app.state.db
    try:
        stats = await report_service.get_summary_stats(db)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch summary statistics: {str(e)}")

@router.get("/reports/export/csv")
async def export_csv(
    request: Request,
    violation_type: Optional[str] = None,
    camera_id: Optional[int] = None,
    status: Optional[str] = None
):
    """Export traffic violations matching filters as a downloadable CSV file."""
    db = request.app.state.db
    filters = {}
    if violation_type:
        filters["violation_type"] = violation_type
    if camera_id:
        filters["camera_id"] = camera_id
    if status:
        filters["status"] = status

    try:
        csv_data = await report_service.export_csv(db, filters)
        
        # Stream response back
        stream = io.StringIO(csv_data)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=traffic_violations_report.csv"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export CSV report: {str(e)}")

@router.get("/reports/export/pdf")
async def export_pdf(request: Request, date_range: Optional[str] = None):
    """Export a comprehensive PDF report with graphs, trends, and legal certificates."""
    db = request.app.state.db
    try:
        pdf_bytes = await report_service.generate_pdf(db, date_range)
        
        # Return PDF file response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=traffic_violations_report.pdf",
                "Cache-Control": "no-cache"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {str(e)}")
