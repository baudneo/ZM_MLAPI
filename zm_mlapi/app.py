import collections
import inspect
import logging
import sys
import time
from pathlib import Path
from platform import python_version
from typing import Union, Optional, Type, Dict

import cv2
import numpy as np
import portalocker as portalocker
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
    ModelProcessor,
    Settings,
    GlobalConfig,
    ModelOptions,
    FaceModelOptions,
    ALPRModelOptions,
    APIDetector,
    MLModelConfig,
    DetectionResult,
)

__version__ = "0.0.1"
__version_type__ = "dev"

logger = logging.getLogger("zm_mlapi")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s",
    "%m/%d/%y %H:%M:%S",
)
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logger.info(
    f"ZM_MLAPI: {__version__} (type: {__version_type__}) [Python: {python_version()} - "
    f"OpenCV: {cv2.__version__} - Numpy: {np.__version__} - FastAPI: {fastapi_version} - "
    f"Pydantic: {pydantic.VERSION}]"
)

app = FastAPI(debug=True)
g = GlobalConfig()




# Allow Form() to contain JSON, Nested JSON is not allowed though - Transforms into Query()
def as_form(cls: Type[BaseModel]):
    logger.info(f"as_form: {cls}")
    new_parameters = []

    for field_name, model_field in cls.__fields__.items():
        model_field: ModelField  # type: ignore
        logger.info(f"as_form: {field_name} -> {model_field}")

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


def get_global_config() -> GlobalConfig:
    return g


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def docs():
    return RedirectResponse(url="/docs")


@app.get("/available_models", summary="Get a list of available models")
async def available_models():
    return {"models": get_global_config().available_models}


def get_detector(
    model_uuid: str,
    model: MLModelConfig,
    model_options: Union[ModelOptions, FaceModelOptions, ALPRModelOptions],
):
    detectors = get_global_config().detectors
    detector: Optional[APIDetector] = None
    # check if a detector already exists for this model and processor
    for detector in detectors:
        if (
            str(detector.model_config.id) == model_uuid
            and detector.model_options.processor == model_options.processor
        ):
            logger.info(
                f"Using existing {detector.model_processor} detector for model {model_uuid} -> {detector}"
            )
            # set model_options on existing detector, if they are different the model will be reloaded into memory
            # TODO: only reload on certain keys being different
            detector.set_model_options(model_options)
            break
    else:
        # create a new detector and append to the list
        detector = APIDetector(model, model_options)
        detectors.append(detector)
        logger.info(f"Created new detector for model {model_uuid} -> {detector}")
    return detector


def get_available_model(model_uuid: str) -> MLModelConfig:
    available_models = get_global_config().available_models
    for model in available_models:
        if str(model.id) == model_uuid:
            return model
    raise HTTPException(status_code=404, detail=f"Model {model_uuid} not found")

@app.post("/detector/create", summary="Create a new detector (ML pipeline sequence)")
async def create_detector(
    model_uuid: str = Form(...),
    model_options: Union[ModelOptions, FaceModelOptions, ALPRModelOptions] = Form(
        ...
    ),
    global_config: GlobalConfig = Depends(get_global_config),
):
    model = get_available_model(model_uuid)
    detector = get_detector(model_uuid, model, model_options)
    return detector


# upload an image for inference
@app.post(
    "/detect/object/{model_uuid}",
    summary="Run detection on an image",
    response_model=DetectionResult,
)
async def object_detection(
    model_uuid: str,
    model_options: ModelOptions = Depends(),
    image: UploadFile = File(..., description="Image to run the ML model on"),
):
    model_uuid = model_uuid.lower().strip()
    model: MLModelConfig = get_available_model(model_uuid)
    detector: Optional[APIDetector] = get_detector(model_uuid, model, model_options)
    frame = load_image_into_numpy_array(await image.read())
    detections: Dict = detector.detect(frame)
    logger.info(f"detections -> {detections}")
    return detections


@app.post(
    "/detect/face/{model_uuid}",
    summary="Run Face detection/recognition on an image",
    response_model=DetectionResult,
)
async def face_detection(
    model_uuid: str,
    model_options: FaceModelOptions = Depends(),
    image: UploadFile = File(..., description="Image to run the ML model on"),
):
    model_uuid = model_uuid.lower().strip()
    model: MLModelConfig = get_available_model(model_uuid)
    detector: Optional[APIDetector] = get_detector(model_uuid, model, model_options)
    frame = load_image_into_numpy_array(await image.read())
    detections: Dict = detector.detect(frame)
    logger.info(f"detections -> {detections}")
    return detections


@app.post(
    "/detect/alpr/{model_uuid}",
    summary="Run ALPR detection on an image",
    response_model=DetectionResult,
)
async def alpr_detection(
    model_uuid: str,
    model_options: ALPRModelOptions = Depends(),
    image: UploadFile = File(..., description="Image to run the ML model on"),
):
    model_uuid = model_uuid.lower().strip()
    model: MLModelConfig = get_available_model(model_uuid)
    detector: Optional[APIDetector] = get_detector(model_uuid, model, model_options)
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
        """LOGGING_CONFIG: Dict[str, Any] = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(levelprefix)s %(message)s",
                    "use_colors": None,
                },
                "access": {
                    "()": "uvicorn.logging.AccessFormatter",
                    "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',  # noqa: E501
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
                "uvicorn": {"handlers": ["default"], "level": "INFO"},
                "uvicorn.error": {"level": "INFO"},
                "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            },
        }
        """
        uvicorn.config.LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s"
        uvicorn.config.LOGGING_CONFIG["formatters"]["default"]["use_colors"] = True
        uvicorn.config.LOGGING_CONFIG["handlers"]["default"]["level"] = "DEBUG"
        uvicorn.config.LOGGING_CONFIG["loggers"]["uvicorn"]["level"] = "DEBUG"
        uvicorn.config.LOGGING_CONFIG["loggers"]["uvicorn.error"]["level"] = "DEBUG"
        config = uvicorn.Config(
            "zm_mlapi.app:app",
            host=self.cached_settings.host,
            port=self.cached_settings.port,
            reload=self.cached_settings.reload,
            debug=self.cached_settings.debug,
            log_config=uvicorn.config.LOGGING_CONFIG,
            log_level="debug",
            proxy_headers=True,
        )
        lifetime = time.perf_counter()
        self.server = uvicorn.Server(config)
        self.server.run()
        lifetime = time.perf_counter() - lifetime
        logger.debug(f"server running for {lifetime:.2f} seconds")
