"""Async SQLite connection management and schema bootstrapping."""

from __future__ import annotations

import aiosqlite

from app.config import settings

# Module-level connection holder.  Initialised by ``connect()`` during the
# application lifespan startup and closed by ``disconnect()`` on shutdown.
_db: aiosqlite.Connection | None = None


# ── DDL statements ──────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cameras (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    rtsp_url        TEXT    NOT NULL,
    location_lat    REAL,
    location_lon    REAL,
    status          TEXT    NOT NULL DEFAULT 'inactive',
    zone_config     TEXT,               -- JSON blob
    calibration_data TEXT,              -- JSON blob
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS vehicle_trajectories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_time  TEXT    NOT NULL,
    camera_id       INTEGER NOT NULL REFERENCES cameras(id),
    track_id        INTEGER NOT NULL,
    vehicle_class   TEXT,
    centroid_x      REAL    NOT NULL,
    centroid_y      REAL    NOT NULL,
    speed           REAL,
    lane_id         TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS traffic_violations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    violation_id        TEXT    NOT NULL UNIQUE,       -- UUID v4
    violation_time      TEXT    NOT NULL,
    camera_id           INTEGER NOT NULL REFERENCES cameras(id),
    track_id            INTEGER,
    vehicle_class       TEXT,
    license_plate       TEXT,
    plate_confidence    REAL,
    violation_type      TEXT    NOT NULL,
    confidence          REAL    NOT NULL,
    evidence_image_path TEXT,
    evidence_crop_path  TEXT,
    sha256_hash         TEXT,
    status              TEXT    NOT NULL DEFAULT 'pending',
    vehicle_hash        TEXT,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS training_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL UNIQUE,
    model_name      TEXT    NOT NULL,
    dataset         TEXT    NOT NULL,
    epochs          INTEGER NOT NULL,
    batch_size      INTEGER NOT NULL DEFAULT 16,
    status          TEXT    NOT NULL DEFAULT 'queued',
    current_epoch   INTEGER NOT NULL DEFAULT 0,
    best_map        REAL,
    loss            REAL,
    started_at      TEXT,
    completed_at    TEXT,
    config          TEXT,               -- JSON blob
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS traffic_density_hourly (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    window_hour     TEXT    NOT NULL,
    camera_id       INTEGER NOT NULL REFERENCES cameras(id),
    lane_id         TEXT,
    total_vehicles  INTEGER NOT NULL DEFAULT 0,
    average_speed   REAL
);

CREATE TABLE IF NOT EXISTS system_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_violations_type
    ON traffic_violations(violation_type);
CREATE INDEX IF NOT EXISTS idx_violations_camera
    ON traffic_violations(camera_id);
CREATE INDEX IF NOT EXISTS idx_violations_time
    ON traffic_violations(violation_time);
CREATE INDEX IF NOT EXISTS idx_violations_status
    ON traffic_violations(status);
CREATE INDEX IF NOT EXISTS idx_violations_vehicle_hash
    ON traffic_violations(vehicle_hash);
CREATE INDEX IF NOT EXISTS idx_trajectories_camera
    ON vehicle_trajectories(camera_id);
CREATE INDEX IF NOT EXISTS idx_density_camera_hour
    ON traffic_density_hourly(camera_id, window_hour);
"""


async def connect() -> aiosqlite.Connection:
    """Open (or return the existing) database connection and ensure schema."""
    global _db  # noqa: PLW0603
    if _db is not None:
        return _db

    settings.ensure_directories()
    _db = await aiosqlite.connect(settings.sqlite_path)
    _db.row_factory = aiosqlite.Row
    await _db.execute("PRAGMA journal_mode=WAL;")
    await _db.execute("PRAGMA foreign_keys=ON;")
    await _db.executescript(_SCHEMA_SQL)
    
    # Migration: Add vehicle_hash column to traffic_violations if it doesn't exist
    try:
        async with _db.execute("PRAGMA table_info(traffic_violations)") as cursor:
            columns = [row[1] for row in await cursor.fetchall()]
            if "vehicle_hash" not in columns:
                await _db.execute("ALTER TABLE traffic_violations ADD COLUMN vehicle_hash TEXT;")
                await _db.execute("CREATE INDEX IF NOT EXISTS idx_violations_vehicle_hash ON traffic_violations(vehicle_hash);")
                await _db.commit()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Migration failed: {e}")
        
    await _db.commit()
    return _db


async def disconnect() -> None:
    """Close the database connection if open."""
    global _db  # noqa: PLW0603
    if _db is not None:
        await _db.close()
        _db = None


async def get_db() -> aiosqlite.Connection:
    """Return the active database connection.

    Raises ``RuntimeError`` if called before ``connect()``.
    """
    if _db is None:
        raise RuntimeError("Database not initialised – call connect() first")
    return _db
