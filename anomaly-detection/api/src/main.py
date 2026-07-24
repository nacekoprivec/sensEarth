import os
from sqlalchemy import text
from fastapi import Depends, FastAPI
import argparse
from .component.router import router
from fastapi.middleware.cors import CORSMiddleware
from .database import *
import main
import asyncio
import pandas as pd
from sqlalchemy.orm import Session
from datetime import *
import traceback
from .component.exceptions import create_exception_handlers
from .component.service import deactivate_stale_sensors
from .logger import logger

from monitoring.client import emit_component_registration, emit_heartbeat

# How often the middleware re-checks for sensors that have been silent too long.
# Default: once per hour (deactivation threshold itself is 7 days — see SENSOR_INACTIVITY_DAYS).
SENSOR_DEACTIVATION_CHECK_INTERVAL_SECONDS = int(
    os.getenv("SENSOR_DEACTIVATION_CHECK_INTERVAL_SECONDS", str(60 * 60))
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
create_exception_handlers(app)


async def stale_sensor_deactivation_loop() -> None:
    """
    Periodically deactivate sensors that have not been seen recently.
    """
    while True:
        try:
            db = SessionLocal()
            try:
                deactivate_stale_sensors(db)
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Stale sensor deactivation failed: {e}")

        await asyncio.sleep(SENSOR_DEACTIVATION_CHECK_INTERVAL_SECONDS)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the Anomaly Detection API...")

    try:
        logger.debug("Emitting component registration to monitoring system")
        emit_component_registration(name="middleware", instance_id="default", component_type="middleware")
        logger.debug("Component registration sent to monitoring system")

        logger.debug( "Emitting initial heartbeat to monitoring system")
        emit_heartbeat(name="middleware", instance_id="default", status="OK")

        logger.debug("Emitting component registration for database to monitoring system")
        emit_component_registration(name="database", instance_id="default", component_type="database")

    except Exception as e:
        logger.warning(f"Error during component registration: {e}")

    # First run happens immediately, then every SENSOR_DEACTIVATION_CHECK_INTERVAL_SECONDS.
    asyncio.create_task(stale_sensor_deactivation_loop())
    logger.info(
        "Started stale sensor deactivation loop "
        f"(interval={SENSOR_DEACTIVATION_CHECK_INTERVAL_SECONDS}s)"
    )

@app.get("/test-db")
def test(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1;"))
        return {"status": "connected"}
    except Exception as e:
        return {"status": "error", "details": str(e)}





