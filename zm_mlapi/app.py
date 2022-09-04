import logging
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Union
from platform import python_version

import cv2
import numpy as np
import pydantic
import uvicorn
import sqlalchemy as sa
from fastapi import (
    FastAPI,
    HTTPException,
    File,
    UploadFile,
    Depends,
    __version__ as fastapi_version,
)
from fastapi.responses import RedirectResponse
from starlette.requests import Request

from zm_mlapi.schemas import (
    Settings,
    ModelType,
    GlobalConfig,
    ModelProcessor,
    ModelFrameWork, ModelSequence,
)
__db_driver__ = "mysql+pymysql://"

# mysql+pymysql://<username>:<password>@<host>/<dbname>[?<options>]
__version__ = "0.0.1"
__version_type__ = "dev"

logger = logging.getLogger("zm_mlapi")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d->[%(message)s]",
    "%m/%d/%y %H:%M:%S",
)
# 08/28/22 11:17:10.794009 zm_mlapi[170] DBG1 pyzm_utils:1567->
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logger.info(
    f"ZM_MLAPI: {__version__} (type: {__version_type__}) [Python: {python_version()} - OpenCV: {cv2.__version__} - Numpy: {np.__version__} - FastAPI: {fastapi_version} - Pydantic: {pydantic.VERSION}]"
)

app = FastAPI()
g = GlobalConfig()


@lru_cache()
async def read_cached_settings(env_file):
    return Settings(_env_file=env_file)


async def read_settings(env_file):
    return Settings(_env_file=env_file)


def get_global_config() -> GlobalConfig:
    return g

@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def docs():
    return RedirectResponse(url="/docs")


@app.get("/available_models")
async def available_models():
    return {"models": g.available_models}


@app.get("/available_models/{model_uuid}")
async def available_model(model_uuid: str):
    model_uuid = model_uuid.lower().strip()

    if model_uuid not in g.available_models:
        raise HTTPException(status_code=404, detail="Model not found")

    return g.available_models[model_uuid]


# upload an image for inference
from fastapi import Path as FastPath, Query


@app.post("models/create_sequence", description="Create a new model sequence")
async def create_live_model(sequence: ModelSequence):
    model_uuid = str(sequence.based_on).lower().strip()

    if model_uuid not in g.available_models:
        raise HTTPException(status_code=404, detail="Available model UUID not found")

    model = g.available_models[model_uuid]


    return sequence




@app.post("/detect/{model_uuid}", summary="Run detection on an image")
async def object_detection(
    model_uuid: str = FastPath(description="A valid model UUID"),
    image: UploadFile = File(..., description="Image to run the ML model on"),
    min_conf: float = Query(
        0.2,
        le=1.0,
        ge=0.0,
        description="Minimum confidence to return a detection expressed as a float between 0.0 and 1.0",
    ),
    nms_threshold: float = Query(
        0.4,
        le=1.0,
        ge=0.0,
        description="Non Max Suppression (NMS) threshold to apply to the model results expressed as a float between 0.0 and 1.0",
    ),
):
    model_uuid = model_uuid.lower().strip()
    if model_uuid not in g.available_models:
        raise HTTPException(status_code=404, detail="Model UUID not found")


class MLAPI:
    available_models: list
    cached_settings: Settings
    env_file: Path
    server: uvicorn.Server

    def __init__(self, env_file: Union[str, Path], run_server: bool = False):
        """
        Initialize the FastAPI MLAPI server object, read a supplied environment file, and start the server if requested.
        :param env_file: The settings file to read in the Bash ENVIRONMENT style.
        :param run_server: Start the server after initialization.
        """

        if not isinstance(env_file, (str, Path)):
            raise ValueError(
                f"The supplied ENVIRONMENT file must be a str or pathlib.Path object, not {type(env_file)}"
            )
        # test that the env file exists and is a file
        self.env_file = Path(env_file)
        if not self.env_file.exists():
            raise FileNotFoundError(f"'{self.env_file.as_posix()}' does not exist")
        elif not self.env_file.is_file():
            raise ValueError(f"'{self.env_file.as_posix()}' is not a file")
        logger.info(f"reading settings from '{self.env_file.as_posix()}'")
        self.cached_settings = g.settings = self.read_settings()
        if run_server:
            self.start_server()

    def read_settings(self):
        if self.env_file.exists():
            g.settings = self.cached_settings = Settings(_env_file=self.env_file)
            logger.debug(f"{self.cached_settings = }")
            self.available_models = (
                g.available_models
            ) = self.cached_settings.available_models
        else:
            raise FileNotFoundError(f"'{self.env_file.as_posix()}' does not exist")

        return self.cached_settings

    def restart_server(self):
        self.read_settings()
        self.start_server()

    def start_server(self):
        logger.info("running server")
        config = uvicorn.Config(
            "zm_mlapi.app:app",
            host=self.cached_settings.host,
            port=self.cached_settings.port,
            reload=self.cached_settings.reload,
        )
        lifetime = time.perf_counter()
        self.server = uvicorn.Server(config)
        self.server.run()
        lifetime = time.perf_counter() - lifetime
        logger.debug(f"server running for {lifetime:.2f} seconds")
