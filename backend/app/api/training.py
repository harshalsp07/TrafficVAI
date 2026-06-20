"""Training API routes."""
import uuid
from fastapi import APIRouter, Request, HTTPException
from app.models.training import TrainingConfig
from app.db import repository as repo

router = APIRouter()


@router.post("/training/start")
async def start_training(request: Request, config: TrainingConfig):
    """Launch a new training job."""
    db = request.app.state.db
    run_id = str(uuid.uuid4())[:8]

    run_data = {
        "run_id": run_id,
        "model_name": config.model_name,
        "dataset": config.dataset,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "status": "queued",
        "current_epoch": 0,
        "best_map": 0.0,
        "loss": 0.0,
    }

    await repo.create_training_run(db, **run_data)

    try:
        from app.services.training_service import TrainingService
        service = TrainingService()
        service.start_training(config.model_dump(), run_id)
        await repo.update_training_run(db, run_id, status="running")
    except Exception as e:
        await repo.update_training_run(db, run_id, status="failed")
        return {"run_id": run_id, "status": "failed", "error": str(e)}

    return {"run_id": run_id, "status": "running"}


@router.get("/training/runs")
async def list_runs(request: Request):
    """List all training runs."""
    db = request.app.state.db
    runs = await repo.list_training_runs(db)
    return {"data": runs}


@router.get("/training/runs/{run_id}")
async def get_run(request: Request, run_id: str):
    """Get training run details."""
    db = request.app.state.db
    run = await repo.get_training_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    return run


@router.post("/training/runs/{run_id}/stop")
async def stop_run(request: Request, run_id: str):
    """Stop a running training job."""
    db = request.app.state.db

    try:
        from app.services.training_service import TrainingService
        service = TrainingService()
        service.stop_training(run_id)
    except Exception:
        pass

    await repo.update_training_run(db, run_id, status="stopped")
    return {"run_id": run_id, "status": "stopped"}


@router.get("/training/models")
async def list_models(request: Request):
    """List available model weights."""
    from app.services.training_service import TrainingService
    service = TrainingService()
    models = service.list_models()
    return {"data": models}


@router.post("/training/deploy")
async def deploy_model(request: Request, body: dict):
    """Set active model for inference."""
    model_path = body.get("model_path")
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path is required")

    from app.services.training_service import TrainingService
    service = TrainingService()
    success = service.deploy_model(model_path)

    if success:
        return {"message": f"Model {model_path} deployed successfully"}
    raise HTTPException(status_code=400, detail="Failed to deploy model")
