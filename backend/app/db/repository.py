"""CRUD repository – every database interaction goes through here.

All methods accept an ``aiosqlite.Connection`` as the first argument so the
caller can control the transaction boundary.  Parameterised queries (``?``)
are used everywhere to prevent SQL injection.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import aiosqlite


# ═══════════════════════════════════════════════════════════════════════
#  Cameras
# ═══════════════════════════════════════════════════════════════════════

async def list_cameras(db: aiosqlite.Connection) -> list[dict[str, Any]]:
    """Return all cameras ordered by creation date."""
    cursor = await db.execute("SELECT * FROM cameras ORDER BY created_at DESC")
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def get_camera(db: aiosqlite.Connection, camera_id: int) -> dict[str, Any] | None:
    """Return a single camera by *camera_id*, or ``None``."""
    cursor = await db.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None


async def create_camera(
    db: aiosqlite.Connection,
    *,
    name: str,
    rtsp_url: str,
    location_lat: float | None = None,
    location_lon: float | None = None,
    status: str = "inactive",
    zone_config: dict | None = None,
    calibration_data: dict | None = None,
) -> dict[str, Any]:
    """Insert a new camera row and return it."""
    cursor = await db.execute(
        """INSERT INTO cameras
               (name, rtsp_url, location_lat, location_lon, status,
                zone_config, calibration_data)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            name,
            rtsp_url,
            location_lat,
            location_lon,
            status,
            json.dumps(zone_config) if zone_config else None,
            json.dumps(calibration_data) if calibration_data else None,
        ),
    )
    await db.commit()
    return await get_camera(db, cursor.lastrowid)  # type: ignore[arg-type]


async def update_camera(
    db: aiosqlite.Connection,
    camera_id: int,
    **fields: Any,
) -> dict[str, Any] | None:
    """Update arbitrary columns on a camera.  JSON fields are serialised."""
    if not fields:
        return await get_camera(db, camera_id)

    set_parts: list[str] = []
    values: list[Any] = []
    for key, val in fields.items():
        set_parts.append(f"{key} = ?")
        if key in ("zone_config", "calibration_data") and isinstance(val, dict):
            values.append(json.dumps(val))
        else:
            values.append(val)
    values.append(camera_id)

    await db.execute(
        f"UPDATE cameras SET {', '.join(set_parts)} WHERE id = ?",  # noqa: S608
        values,
    )
    await db.commit()
    return await get_camera(db, camera_id)


async def delete_camera(db: aiosqlite.Connection, camera_id: int) -> bool:
    """Delete a camera by id.  Returns ``True`` if a row was removed."""
    cursor = await db.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
    await db.commit()
    return cursor.rowcount > 0


# ═══════════════════════════════════════════════════════════════════════
#  Violations
# ═══════════════════════════════════════════════════════════════════════

async def create_violation(db: aiosqlite.Connection, **data: Any) -> dict[str, Any]:
    """Insert a new traffic violation."""
    cols = ", ".join(data.keys())
    placeholders = ", ".join("?" for _ in data)
    cursor = await db.execute(
        f"INSERT INTO traffic_violations ({cols}) VALUES ({placeholders})",  # noqa: S608
        list(data.values()),
    )
    await db.commit()
    return await get_violation(db, cursor.lastrowid)  # type: ignore[arg-type]


async def get_violation(db: aiosqlite.Connection, row_id: int) -> dict[str, Any] | None:
    """Get a single violation by primary-key *row_id*."""
    cursor = await db.execute(
        "SELECT * FROM traffic_violations WHERE id = ?", (row_id,)
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def get_violation_by_uuid(
    db: aiosqlite.Connection, violation_id: str
) -> dict[str, Any] | None:
    """Get a single violation by its UUID ``violation_id``."""
    cursor = await db.execute(
        "SELECT * FROM traffic_violations WHERE violation_id = ?", (violation_id,)
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def check_recent_duplicate(
    db: aiosqlite.Connection,
    camera_id: int,
    violation_type: str,
    license_plate: str,
    window_minutes: int = 5
) -> bool:
    """Check if there is a duplicate violation of the same type, camera, and plate recently."""
    if not license_plate or license_plate == "UNKNOWN":
        return False
        
    query = """
        SELECT COUNT(*) FROM traffic_violations 
        WHERE camera_id = ? 
          AND violation_type = ? 
          AND license_plate = ? 
          AND datetime(violation_time) >= datetime('now', ?)
    """
    time_offset = f"-{window_minutes} minutes"
    cursor = await db.execute(query, (camera_id, violation_type, time_offset))
    row = await cursor.fetchone()
    return row[0] > 0 if row else False


async def get_recent_hashes(
    db: aiosqlite.Connection,
    camera_id: int,
    violation_type: str,
    window_minutes: int = 5
) -> list[str]:
    """Retrieve list of vehicle perceptual hashes for similar violations recently."""
    query = """
        SELECT vehicle_hash FROM traffic_violations 
        WHERE camera_id = ? 
          AND violation_type = ? 
          AND vehicle_hash IS NOT NULL
          AND datetime(violation_time) >= datetime('now', ?)
    """
    time_offset = f"-{window_minutes} minutes"
    cursor = await db.execute(query, (camera_id, violation_type, time_offset))
    rows = await cursor.fetchall()
    return [row[0] for row in rows] if rows else []


async def list_violations(
    db: aiosqlite.Connection,
    *,
    violation_type: str | None = None,
    camera_id: int | None = None,
    status: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Return paginated, filtered violations and the total matching count."""
    where_parts: list[str] = []
    params: list[Any] = []

    if violation_type:
        where_parts.append("violation_type = ?")
        params.append(violation_type)
    if camera_id is not None:
        where_parts.append("camera_id = ?")
        params.append(camera_id)
    if status:
        where_parts.append("status = ?")
        params.append(status)
    if date_from:
        where_parts.append("violation_time >= ?")
        params.append(date_from)
    if date_to:
        where_parts.append("violation_time <= ?")
        params.append(date_to)

    where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

    count_cursor = await db.execute(
        f"SELECT COUNT(*) FROM traffic_violations{where_clause}",  # noqa: S608
        params,
    )
    total = (await count_cursor.fetchone())[0]

    params_with_pagination = params + [limit, offset]
    cursor = await db.execute(
        f"SELECT * FROM traffic_violations{where_clause} "  # noqa: S608
        "ORDER BY violation_time DESC LIMIT ? OFFSET ?",
        params_with_pagination,
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows], total


async def update_violation_status(
    db: aiosqlite.Connection,
    violation_id: str,
    status: str,
) -> dict[str, Any] | None:
    """Set violation status to *confirmed* or *dismissed*."""
    await db.execute(
        "UPDATE traffic_violations SET status = ? WHERE violation_id = ?",
        (status, violation_id),
    )
    await db.commit()
    return await get_violation_by_uuid(db, violation_id)


async def violation_stats(
    db: aiosqlite.Connection,
) -> dict[str, Any]:
    """Aggregate violation statistics for the dashboard."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Today's count
    cur = await db.execute(
        "SELECT COUNT(*) FROM traffic_violations WHERE violation_time >= ?",
        (today,),
    )
    today_count = (await cur.fetchone())[0]

    # By type
    cur = await db.execute(
        "SELECT violation_type, COUNT(*) AS cnt "
        "FROM traffic_violations GROUP BY violation_type ORDER BY cnt DESC"
    )
    by_type = [{"type": r["violation_type"], "count": r["cnt"]} for r in await cur.fetchall()]

    # By camera
    cur = await db.execute(
        "SELECT camera_id, COUNT(*) AS cnt "
        "FROM traffic_violations GROUP BY camera_id ORDER BY cnt DESC"
    )
    by_camera = [{"camera_id": r["camera_id"], "count": r["cnt"]} for r in await cur.fetchall()]

    # By status
    cur = await db.execute(
        "SELECT status, COUNT(*) AS cnt "
        "FROM traffic_violations GROUP BY status"
    )
    by_status = {r["status"]: r["cnt"] for r in await cur.fetchall()}

    return {
        "today_count": today_count,
        "by_type": by_type,
        "by_camera": by_camera,
        "by_status": by_status,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Vehicle Trajectories
# ═══════════════════════════════════════════════════════════════════════

async def create_trajectory(db: aiosqlite.Connection, **data: Any) -> int:
    """Insert a trajectory point and return its row id."""
    cols = ", ".join(data.keys())
    placeholders = ", ".join("?" for _ in data)
    cursor = await db.execute(
        f"INSERT INTO vehicle_trajectories ({cols}) VALUES ({placeholders})",  # noqa: S608
        list(data.values()),
    )
    await db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def list_trajectories(
    db: aiosqlite.Connection,
    camera_id: int | None = None,
    track_id: int | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Fetch trajectory points with optional camera / track filters."""
    where_parts: list[str] = []
    params: list[Any] = []
    if camera_id is not None:
        where_parts.append("camera_id = ?")
        params.append(camera_id)
    if track_id is not None:
        where_parts.append("track_id = ?")
        params.append(track_id)

    where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
    params.append(limit)

    cursor = await db.execute(
        f"SELECT * FROM vehicle_trajectories{where_clause} "  # noqa: S608
        "ORDER BY detection_time DESC LIMIT ?",
        params,
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════
#  Training Runs
# ═══════════════════════════════════════════════════════════════════════

async def create_training_run(db: aiosqlite.Connection, **data: Any) -> dict[str, Any]:
    """Insert a new training run record."""
    if "config" in data and isinstance(data["config"], dict):
        data["config"] = json.dumps(data["config"])
    cols = ", ".join(data.keys())
    placeholders = ", ".join("?" for _ in data)
    cursor = await db.execute(
        f"INSERT INTO training_runs ({cols}) VALUES ({placeholders})",  # noqa: S608
        list(data.values()),
    )
    await db.commit()
    return await get_training_run_by_rowid(db, cursor.lastrowid)  # type: ignore[arg-type]


async def get_training_run_by_rowid(
    db: aiosqlite.Connection, row_id: int
) -> dict[str, Any] | None:
    """Fetch a training run by primary key."""
    cursor = await db.execute("SELECT * FROM training_runs WHERE id = ?", (row_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None


async def get_training_run(
    db: aiosqlite.Connection, run_id: str
) -> dict[str, Any] | None:
    """Fetch a training run by its UUID *run_id*."""
    cursor = await db.execute(
        "SELECT * FROM training_runs WHERE run_id = ?", (run_id,)
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def list_training_runs(db: aiosqlite.Connection) -> list[dict[str, Any]]:
    """Return all training runs, newest first."""
    cursor = await db.execute(
        "SELECT * FROM training_runs ORDER BY created_at DESC"
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def update_training_run(
    db: aiosqlite.Connection, run_id: str, **fields: Any
) -> dict[str, Any] | None:
    """Update arbitrary fields on a training run."""
    if not fields:
        return await get_training_run(db, run_id)
    if "config" in fields and isinstance(fields["config"], dict):
        fields["config"] = json.dumps(fields["config"])

    set_parts = [f"{k} = ?" for k in fields]
    values = list(fields.values()) + [run_id]
    await db.execute(
        f"UPDATE training_runs SET {', '.join(set_parts)} WHERE run_id = ?",  # noqa: S608
        values,
    )
    await db.commit()
    return await get_training_run(db, run_id)


# ═══════════════════════════════════════════════════════════════════════
#  Traffic Density
# ═══════════════════════════════════════════════════════════════════════

async def upsert_density(
    db: aiosqlite.Connection,
    *,
    window_hour: str,
    camera_id: int,
    lane_id: str | None,
    total_vehicles: int,
    average_speed: float | None,
) -> int:
    """Insert or replace a traffic density record."""
    cursor = await db.execute(
        """INSERT INTO traffic_density_hourly
               (window_hour, camera_id, lane_id, total_vehicles, average_speed)
           VALUES (?, ?, ?, ?, ?)""",
        (window_hour, camera_id, lane_id, total_vehicles, average_speed),
    )
    await db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_density(
    db: aiosqlite.Connection,
    camera_id: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """Query traffic density rows with optional filters."""
    where_parts: list[str] = []
    params: list[Any] = []
    if camera_id is not None:
        where_parts.append("camera_id = ?")
        params.append(camera_id)
    if date_from:
        where_parts.append("window_hour >= ?")
        params.append(date_from)
    if date_to:
        where_parts.append("window_hour <= ?")
        params.append(date_to)

    where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
    cursor = await db.execute(
        f"SELECT * FROM traffic_density_hourly{where_clause} "  # noqa: S608
        "ORDER BY window_hour DESC LIMIT 500",
        params,
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════
#  Analytics helpers
# ═══════════════════════════════════════════════════════════════════════

async def violation_trends(
    db: aiosqlite.Connection,
    *,
    granularity: str = "hourly",
    camera_id: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """Time-series counts of violations grouped by hour or day."""
    if granularity == "daily":
        time_expr = "date(violation_time)"
    else:
        time_expr = "strftime('%Y-%m-%dT%H:00', violation_time)"

    where_parts: list[str] = []
    params: list[Any] = []
    if camera_id is not None:
        where_parts.append("camera_id = ?")
        params.append(camera_id)
    if date_from:
        where_parts.append("violation_time >= ?")
        params.append(date_from)
    if date_to:
        where_parts.append("violation_time <= ?")
        params.append(date_to)

    where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

    cursor = await db.execute(
        f"SELECT {time_expr} AS bucket, COUNT(*) AS count "  # noqa: S608
        f"FROM traffic_violations{where_clause} "
        f"GROUP BY bucket ORDER BY bucket",
        params,
    )
    rows = await cursor.fetchall()
    return [{"time": r["bucket"], "count": r["count"]} for r in rows]


async def violation_heatmap(db: aiosqlite.Connection) -> list[dict[str, Any]]:
    """Return locations for map overlay."""
    cursor = await db.execute(
        """SELECT tv.violation_id, tv.violation_type, tv.violation_time,
                  c.location_lat, c.location_lon, c.name AS camera_name
           FROM traffic_violations tv
           JOIN cameras c ON c.id = tv.camera_id
           WHERE c.location_lat IS NOT NULL AND c.location_lon IS NOT NULL
           ORDER BY tv.violation_time DESC LIMIT 500"""
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def violation_distribution(
    db: aiosqlite.Connection,
    group_by: str = "violation_type",
) -> list[dict[str, Any]]:
    """Count violations grouped by a column (violation_type or vehicle_class)."""
    allowed = {"violation_type", "vehicle_class"}
    col = group_by if group_by in allowed else "violation_type"

    cursor = await db.execute(
        f"SELECT {col} AS label, COUNT(*) AS count "  # noqa: S608
        f"FROM traffic_violations GROUP BY {col} ORDER BY count DESC"
    )
    rows = await cursor.fetchall()
    return [{"label": r["label"], "count": r["count"]} for r in rows]


async def speed_distribution(
    db: aiosqlite.Connection,
    camera_id: int | None = None,
) -> list[dict[str, Any]]:
    """Return speed data from trajectories for histogram / distribution charts."""
    where_parts = ["speed IS NOT NULL"]
    params: list[Any] = []
    if camera_id is not None:
        where_parts.append("camera_id = ?")
        params.append(camera_id)

    where_clause = " WHERE " + " AND ".join(where_parts)
    cursor = await db.execute(
        f"SELECT speed, vehicle_class, camera_id "  # noqa: S608
        f"FROM vehicle_trajectories{where_clause} "
        "ORDER BY detection_time DESC LIMIT 2000",
        params,
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def peak_hours(db: aiosqlite.Connection) -> list[dict[str, Any]]:
    """Return violation counts aggregated by hour-of-day (0-23)."""
    cursor = await db.execute(
        "SELECT CAST(strftime('%H', violation_time) AS INTEGER) AS hour, "
        "COUNT(*) AS count "
        "FROM traffic_violations GROUP BY hour ORDER BY hour"
    )
    rows = await cursor.fetchall()
    return [{"hour": r["hour"], "count": r["count"]} for r in rows]


# ═══════════════════════════════════════════════════════════════════════
#  System Settings
# ═══════════════════════════════════════════════════════════════════════

async def get_setting(db: aiosqlite.Connection, key: str, default: Any = None) -> str | None:
    """Retrieve a single configuration setting by key."""
    cursor = await db.execute("SELECT value FROM system_settings WHERE key = ?", (key,))
    row = await cursor.fetchone()
    return row[0] if row else default


async def set_setting(db: aiosqlite.Connection, key: str, value: str) -> None:
    """Insert or replace a configuration setting."""
    await db.execute("INSERT OR REPLACE INTO system_settings (key, value) VALUES (?, ?)", (key, value))
    await db.commit()


async def list_settings(db: aiosqlite.Connection) -> dict[str, str]:
    """Retrieve all configuration settings as a key-value dictionary."""
    cursor = await db.execute("SELECT key, value FROM system_settings")
    rows = await cursor.fetchall()
    return {row[0]: row[1] for row in rows}
