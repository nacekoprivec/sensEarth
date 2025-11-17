from fastapi import Depends, FastAPI
import argparse
from .component.router import router
from fastapi.middleware.cors import CORSMiddleware
from .database import *
from .component.models import AnomalyDetector, Log
import main
import asyncio
import pandas as pd
from sqlalchemy.orm import Session
from datetime import *
import traceback
from .component.exceptions import create_exception_handlers

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



