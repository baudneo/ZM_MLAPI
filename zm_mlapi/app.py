import inspect
import logging
import sys
import time
from functools import lru_cache
from pathlib import Path
from platform import python_version
from typing import Union, Optional, Type, Tuple, List, Dict

import cv2
import numpy as np
import pydantic
import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    File,
    UploadFile,
    __version__ as fastapi_version,
    Form,
    Depends,
)
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from zm_mlapi.imports import (
    Settings,
    GlobalConfig,
    ModelOptions,
    APIDetector,
    AvailableModel,
    DetectionResult,
)

# mysql+pymysql://<username>:<password>@<host>/<dbname>[?<options>]
__version__ = "0.0.1"
__version_type__ = "dev"

logger = logging.getLogger("zm_mlapi")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s",
    "%m/%d/%y %H:%M:%S",
)
# 08/28/22 11:17:10.794009 zm_mlapi[170] DBG1 pyzm_utils:1567->
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logger.info(
    f"ZM_MLAPI: {__version__} (type: {__version_type__}) [Python: {python_version()} - "
    f"OpenCV: {cv2.__version__} - Numpy: {np.__version__} - FastAPI: {fastapi_version} - "
    f"Pydantic: {pydantic.VERSION}]"
)

app = FastAPI()
g = GlobalConfig()


# Allow Form() to contain JSON, Nested JSON is not allowed though
def as_form(cls: Type[BaseModel]):
    new_parameters = []

    for field_name, model_field in cls.__fields__.items():
        model_field: ModelField  # type: ignore

        new_parameters.append(
            inspect.Parameter(
                model_field.alias,
                inspect.Parameter.POSITIONAL_ONLY,
                default=Form(...)
                if not model_field.required
                else Form(model_field.default),
                annotation=model_field.outer_type_,
            )
        )

    async def as_form_func(**data):
        return cls(**data)

    sig = inspect.signature(as_form_func)
    sig = sig.replace(parameters=new_parameters)
    as_form_func.__signature__ = sig  # type: ignore
    setattr(cls, "as_form", as_form_func)
    return cls


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
    return {"models": get_global_config().available_models}


# upload an image for inference
@app.post(
    "/detect/{model_uuid}",
    summary="Run detection on an image",
    response_model=DetectionResult,
)
async def object_detection(
    model_uuid: str,
    model_options: ModelOptions = Depends(),
    image: UploadFile = File(..., description="Image to run the ML model on"),
):
    logger.info(f"Running detection on image '{image.filename}'")
    model_uuid = model_uuid.lower().strip()
    model: Optional[AvailableModel] = None
    for _model in g.available_models:
        if str(_model.id) == model_uuid:
            logger.debug(f"Found model: {_model.id} NAME: {_model.name}")
            model = _model
            break

    if not model:
        raise HTTPException(status_code=404, detail="Model UUID not found")
    # Check if a detector is already configured using the MODEL and PROCESSOR type, if so, use it with
    # the new image and options
    # If not, create a new detector and use it

    detectors = get_global_config().detectors
    detector: Optional[APIDetector] = None
    # check if a detector already exists for this model and processor
    for detector in detectors:
        if (
            str(detector.model_config.id) == model_uuid
            and detector.model_options.processor == model_options.processor
        ):
            logger.info(f"Using existing detector for model {model_uuid} -> {detector}")
            detector.set_model_options(model_options)
            break
    else:
        # create a new detector and append to the list
        detector = APIDetector(model, model_options)
        detectors.append(detector)
        logger.info(f"Created new detector for model {model_uuid} -> {detector}")
    # now we have a detector, use it
    frame = load_image_into_numpy_array(await image.read())
    detections: Dict = detector.detect(frame)
    logger.info(f"detections -> {detections}")
    return detections


def load_image_into_numpy_array(data):
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


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
            self.cached_settings = Settings(_env_file=self.env_file)
            g.settings = self.cached_settings
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
        logger.info(f"{str(get_global_config().available_models[0].id)}")
        uvicorn_log_config = {
            "version": 1,
            "disable_existing_loggers": True,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d->[%(message)s]",
                    "use_colors": None,
                },
                "access": {
                    "()": "uvicorn.logging.AccessFormatter",
                    "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "access": {
                    "formatter": "access",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "DEBUG"},
                "uvicorn.error": {
                    "level": "DEBUG",
                    "handlers": ["default"],
                    "propagate": True,
                },
                "uvicorn.access": {
                    "handlers": ["access"],
                    "level": "DEBUG",
                    "propagate": False,
                },
                "zm_mlapi": {"handlers": ["default"], "level": "DEBUG"},
            },
        }
        uvicorn.LOGGING_CONFIG = uvicorn_log_config
        config = uvicorn.Config(
            "zm_mlapi.app:app",
            host=self.cached_settings.host,
            port=self.cached_settings.port,
            reload=self.cached_settings.reload,
            debug=self.cached_settings.debug,
            # log_config=uvicorn_log_config,
            log_level="debug",
        )
        lifetime = time.perf_counter()
        self.server = uvicorn.Server(config)
        self.server.run()
        lifetime = time.perf_counter() - lifetime
        logger.debug(f"server running for {lifetime:.2f} seconds")
