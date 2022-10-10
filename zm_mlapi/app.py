import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from platform import python_version
from typing import Union, Optional, Type, Dict, List

import cv2
import numpy as np
import pydantic
import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    __version__ as fastapi_version,
    Form,
    Depends,
    UploadFile,
    File, Body, Path as FastPath

)
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from zm_mlapi.imports import (
    Settings,
    DetectionResult,
    GlobalConfig,
    BaseModelOptions,
    FaceRecognitionLibModelOptions,
    OpenALPRLocalModelOptions,
    APIDetector,
    BaseModelConfig,
    ModelType,
    ModelProcessor,
    ModelFrameWork
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
LP: str = "mlapi:"

# Allow Form() to contain JSON, Nested JSON is not allowed though - Transforms into Query()
"""def as_form(cls: Type[BaseModel]):
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
    return cls"""


def get_global_config() -> GlobalConfig:
    return g


def get_settings() -> Settings:
    return get_global_config().settings


def get_available_models() -> List[BaseModelConfig]:
    return get_global_config().available_models


def locks_enabled():
    ret_ = True
    if get_settings().disable_locks is True:
        ret_ = False
    return ret_


def normalize_id(_id: str) -> str:
    return _id.strip().lower()


def get_model(model_hint: Union[str, BaseModelConfig]) -> BaseModelConfig:
    """Get a model based on the hint provided. Hint can be a model name, model id, or a model object"""
    logger.debug(f"get_model: hint TYPE: {type(model_hint)} -> {model_hint}")
    available_models = get_available_models()
    if available_models:
        if isinstance(model_hint, BaseModelConfig):
            for model in available_models:
                if model.id == model_hint.id or model == model_hint:
                    return model
        elif isinstance(model_hint, str):
            for model in available_models:
                identifiers = {normalize_id(model.name), str(model.id)}
                logger.debug(f"get_model: identifiers: {identifiers}")
                if normalize_id(model_hint) in identifiers:
                    return model
    raise HTTPException(status_code=404, detail=f"Model {model_hint} not found")


async def detect(
        model_hint: str,
        image,
):
    model_hint = normalize_id(model_hint)
    logger.info(f"{LP} detect: {model_hint}")
    model: BaseModelConfig = get_model(model_hint)
    logger.info(f"{LP} found model {model.id} -> {model}")
    detector: APIDetector = get_global_config().get_detector(model)
    image = load_image_into_numpy_array(await image.read())
    timer = time.perf_counter()
    detection: Dict = detector.detect(image)
    logger.info(f"{LP} single detection completed in {time.perf_counter() - timer:.5f}ms -> {detection}")
    return detection


async def threaded_detect(model_hints: List[str], image) -> List[Dict]:
    available_models = get_global_config().available_models
    model_hints = [normalize_id(model_hint) for model_hint in model_hints]
    logger.debug(f"threaded_detect: model_hints -> {model_hints}")
    detectors: List[APIDetector] = []
    for model in available_models:
        identifiers = {model.name, str(model.id)}
        if any([normalize_id(model_hint) in identifiers for model_hint in model_hints]):
            logger.info(f"Found model: {model.name} ({model.id})")
            detector = get_global_config().get_detector(model)
            detectors.append(detector)
    image = load_image_into_numpy_array(await image.read())
    detections: List[Dict] = []
    # logger.info(f"detectors ({len(detectors)}) -> {detectors}")
    timer = time.perf_counter()
    import concurrent.futures
    futures = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for detector in detectors:
            # logger.info(f"Starting detection for {detector}")
            futures.append(executor.submit(detector.detect, image))
    for future in futures:
        detections.append(future.result())
    # for detector in detectors:
    #     import threading
    #     thread = threading.Thread(target=detector.detect, args=(image,))
    #     threads.append(thread)
    #     thread.start()
    #     # detections.append(detector.detect(image))
    # for thread in threads:
    #     logger.info(f"Waiting for thread {thread}")
    #     thread.join()
    logger.info(f"{LP} ThreadPool detections completed in {time.perf_counter() - timer:.5f}ms -> {detections}")
    return detections


def load_image_into_numpy_array(data):
    """Load an uploaded image into a numpy array"""
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def docs():
    return RedirectResponse(url="/docs")


@app.get("/models/available/all", summary="Get a list of all available models")
async def _available_models():
    return {"models": get_global_config().available_models}


@app.get("/models/available/type/{model_type}", summary="Get a list of available models based on the type")
async def available_models_type(model_type: ModelType):
    available_models = get_global_config().available_models
    return {
        "models": [
            model.dict()
            for model in available_models
            if model.model_type == model_type
        ]
    }


@app.get("/models/available/proc/{processor}", summary="Get a list of available models based on the processor")
async def available_models_proc(processor: ModelProcessor):
    logger.info(f"available_models_proc: {processor}")
    available_models = get_global_config().available_models
    return {
        "models": [
            model.dict()
            for model in available_models
            if model.processor == processor
        ]
    }


@app.get("/models/available/framework/{framework}", summary="Get a list of available models based on the framework")
async def available_models_proc(framework: ModelFrameWork):
    logger.info(f"available_models_proc: {framework}")
    available_models = get_global_config().available_models
    return {
        "models": [
            model.dict()
            for model in available_models
            if model.framework == framework
        ]
    }


@app.post("/models/modify/{model_hint}", summary="Change a models options")
async def modify_model(model_hint: str, model_options: Union[BaseModelOptions, FaceRecognitionLibModelOptions, OpenALPRLocalModelOptions]):
    model = get_model(model_hint)
    old_options = model.detection_options
    detector = get_global_config().get_detector(model)
    logger.info(f"modify_model: '{model.name}' original: {old_options}  -> new: {model_options}")
    detector.config.detection_options = model.detection_options = model_options
    return {"original": old_options, "new": model.detection_options}


@app.post("/detect/group", summary="Detect objects in an image using a set of threaded models referenced by name")
async def group_detect(
        model_hints: List[str] = Body(..., description="model names or ids",
                                      example="yolov4,97acd7d4-270c-4667-9d56-910e1510e8e8,yolov7 tiny"),
        image: UploadFile = File(...),
):
    logger.info(f"group_detect: {model_hints}")
    model_hints = model_hints[0].split(",")
    detections = await threaded_detect(model_hints, image)
    return detections


@app.post(
    "/detect/single/{model_hint}",
    summary="Run detection using the specified model on a single image",
    # response_model=DetectionResult,
)
async def single_detection(
        model_hint: str = FastPath(..., description="model name or id", example="yolov4"),
        image: UploadFile = File(..., description="Image to run the ML model on"),
):
    logger.info(f"single_detection: ENDPOINT {model_hint}")
    detections = await detect(model_hint, image)
    return detections


class MLAPI:
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
        self.read_settings()
        if run_server:
            self.start_server()

    def read_settings(self):
        logger.info(f"reading settings from '{self.env_file.as_posix()}'")
        if self.env_file.exists():
            self.cached_settings = Settings(_env_file=self.env_file)
            get_global_config().settings = self.cached_settings
            # logger.debug(f"{g.settings = }")
            available_models = get_global_config().available_models = self.cached_settings.available_models

            if available_models:
                futures = []
                timer = time.perf_counter()
                with ThreadPoolExecutor() as executor:
                    for model in available_models:
                        futures.append(executor.submit(get_global_config().get_detector, model))
                for future in futures:
                    future.result()
                logger.info(
                    f"{LP} TOTAL ThreadPool loading {len(futures)} models took {time.perf_counter() - timer:.5f}ms")
        else:
            raise FileNotFoundError(f"'{self.env_file.as_posix()}' does not exist")

        return self.cached_settings

    def restart_server(self):
        self.read_settings()
        self.start_server()

    def start_server(self):
        logger.info("running server")
        _avail = {}
        for model in get_global_config().available_models:
            _avail[normalize_id(model.name)] = str(model.id)
        logger.info(f"AVAILABLE MODELS! --> {_avail}")
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
        uvicorn.config.LOGGING_CONFIG["formatters"]["default"][
            "fmt"
        ] = "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s"
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
