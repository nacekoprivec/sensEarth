import datetime
import traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Union


# Custom Exceptions (extend HTTPException)
class NotFoundException(HTTPException):
    def __init__(self, resource: str, resource_id: Union[str, int] = None):
        detail = f"{resource} not found."
        if resource_id is not None:
            detail = f"{resource} with id {resource_id} not found."
        super().__init__(status_code=404, detail=detail)

class BadRequestException(HTTPException):
    def __init__(self, detail: str = "Invalid request."):
        super().__init__(status_code=400, detail=detail)

class InternalServerException(HTTPException):
    def __init__(self, detail: str = "Internal server error."):
        super().__init__(status_code=500, detail=detail)

class ConfigFileException(HTTPException):
    def __init__(self, conf_name: str, error: str = "Invalid config file."):
        super().__init__(status_code=400, detail=f"Config '{conf_name}': {error}")

class JSONDecodeException(HTTPException):
    def __init__(self, conf_name: str, error: str = "contains invalid JSON."):
        detail = f"Config '{conf_name}' {error}"
        super().__init__(status_code=400, detail=detail)

class ProcessingException(HTTPException):
    def __init__(self, detail: str = "An error occurred during anomaly detection."):
        super().__init__(status_code=500, detail=detail)

class DetectorNotActiveException(HTTPException):
    def __init__(self, detector_id: int):
        super().__init__(status_code=400, detail=f"Anomaly detector {detector_id} is not active.")

class DetectorNotFoundException(HTTPException):
    def __init__(self, detector_id: int):
        super().__init__(status_code=404, detail=f"Anomaly detector with ID {detector_id} not found.")


# Register detailed exception handlers
def create_exception_handlers(app: FastAPI):
    @app.exception_handler(HTTPException)
    async def custom_http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.detail,
                    "status_code": exc.status_code,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "path": str(request.url),
                    "traceback": None  
                }
            }
        )

    @app.exception_handler(Exception)
    async def custom_general_exception_handler(request: Request, exc: Exception):
        tb = traceback.format_exc()  
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "status_code": 500,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "path": str(request.url),
                    "traceback": tb
                }
            }
        )
