import logging
import tempfile
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union, List, Dict, Optional, IO, Any

import numpy as np
import portalocker
from fastapi import UploadFile, File, Form
from pydantic import BaseModel, Field, validator, BaseSettings
from pydantic.fields import ModelField


logger = logging.getLogger("zm_mlapi")


def str_to_path(v, values, field: ModelField) -> Optional[Path]:
    logger.debug(f"validating {field.name} - {v = } -- {type(v) = } -- {values = }")
    msg = f"{field.name} must be a path or a string of a path"

    if v is None:
        if field.name == "input_file":
            raise ValueError(f"{msg}, not 'None'")
        logger.debug(f"{field.name} is None, passing as it is Optional")
        return v
    elif not isinstance(v, (Path, str)):
        raise ValueError(msg)
    elif isinstance(v, str):
        logger.debug(f"Attempting to convert {field.name} string '{v}' to Path object")
        v = Path(v)

    logger.debug(
        f"DBG>>> {field.name} is a validated Path object -> RETURNING {type(v) = } --> {v = }"
    )
    assert isinstance(v, Path), f"{field.name} is not a Path object"
    return v


def check_model_config_file(v, values, field: ModelField) -> Optional[Path]:
    if field.name == "config_file":
        if values["input_file"].suffix == ".weights":
            msg = f"{field.name} is required when input_file is a .weights file"
            if not v:
                raise ValueError(f"{msg}, it is set as 'None' (Not Configured)!")
            elif not v.exists():
                raise ValueError(f"{msg}, it does not exist")
            elif not v.is_file():
                raise ValueError(f"{msg}, it is not a file")
    return v


def check_labels_file(v, values, field: ModelField) -> Optional[Path]:

    msg = f"{field.name} is required"
    if not v:
        raise ValueError(f"{msg}, it is set as 'None' or empty (Not Configured)!")
    elif not v.exists():
        raise ValueError(f"{msg}, it does not exist")
    elif not v.is_file():
        raise ValueError(f"{msg}, it is not a file")


class ModelType(str, Enum):
    OBJECT = "object"
    FACE = "face"
    ALPR = "alpr"
    DEFAULT = OBJECT

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name} ({str(self.name).lower()} detection)"

    def __str__(self):
        return self.__repr__()


class ModelFrameWork(str, Enum):
    OPENCV = "opencv"
    CORAL = "coral"
    PYCORAL = CORAL
    # VINO = "openvino"
    # OPENVINO = VINO
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    DEEPFACE = "deepface"
    OPENALPR = "openalpr"
    DLIB = "dlib"
    FACE_RECOGNITION = DLIB
    DEFAULT = OPENCV


class ModelProcessor(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    DEFAULT = CPU


class FaceModel(str, Enum):
    CNN = "cnn"
    HOG = "hog"
    DEFAULT = CNN


class DetectionResult(BaseModel):
    detections: bool = False
    type: ModelType = None
    processor: ModelProcessor = None
    model_name: str = None

    label: List[str] = None
    confidence: List[Union[float, int]] = None
    bounding_box: List[List[Union[float, int]]] = None


class ModelOptions(BaseModel):
    processor: Optional[ModelProcessor] = Field(
        ModelProcessor.CPU, description="Processor to use for model"
    )
    height: Optional[int] = Field(
        416, ge=1, description="Height of the input image (resized for model)"
    )
    width: Optional[int] = Field(
        416, ge=1, description="Width of the input image (resized for model)"
    )
    square: Optional[bool] = Field(
        False, description="Zero pad the image to be a square"
    )
    fp_16: Optional[bool] = Field(
        False, description="model uses Floating Point 16 Backend (EXPERIMENTAL!)"
    )
    confidence: Optional[float] = Field(
        0.5, ge=0.0, le=1.0, descritpiton="Confidence Threshold"
    )
    nms: Optional[float] = Field(
        0.4, ge=0.0, le=1.0, description="Non-Maximum Suppression Threshold"
    )


class FaceModelOptions(ModelOptions):
    # Face Detection Options
    upsample_times: int = Field(
        1,
        ge=1,
        description="How many times to upsample the image looking for faces. Higher numbers find smaller faces but take longer.",
    )
    num_jitters: int = Field(
        1,
        ge=1,
        description="How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)",
    )
    face_model: FaceModel = Field(
        FaceModel.DEFAULT, description="Face model to use for detection"
    )
    face_train_model: FaceModel = Field(
        FaceModel.DEFAULT, description="Face model to use for training"
    )
    known_faces_dir: Optional[Union[Path, str]] = Field(
        None, description="Path to parent directory for known faces"
    )
    face_max_size: int = Field(
        600,
        ge=1,
        description="Maximum size (Width) of image to load into memory for "
        "face detection",
    )
    recognition_threshold: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Recognition distance threshold for " "face recognition",
    )
    unknown_face_name: str = Field(
        "Unknown", description="Name to use for unknown faces"
    )
    save_unknown_faces: bool = Field(
        False,
        description="Save cropped unknown faces to disk, can be "
        "used to train a model",
    )
    unknown_faces_leeway_pixels: int = Field(
        0,
        description="Unknown faces leeway pixels, used when cropping the image to capture a face",
    )
    unknown_faces_dir: Optional[Union[Path, str]] = Field(
        None, description="Directory to save unknown faces to"
    )
    face_train_max_size: int = Field(
        800,
        description="Maximum size of image to load into memory for face training, "
        "Larger will consume more memory!",
    )


class ALPRModelOptions(ModelOptions):
    test: str = None


class MLModelConfig(BaseModel):
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4, description="Unique ID of the model"
    )
    name: str = Field(..., description="model name")
    input: Path = Field(..., description="model file/dir path")
    enabled: bool = Field(True, description="model enabled")
    description: str = Field(None, description="model description")
    framework: ModelFrameWork = Field(
        ModelFrameWork.DEFAULT, description="model framework"
    )
    model_type: ModelType = Field(
        ModelType.DEFAULT, description="model type (object, face, alpr)"
    )
    classes: Path = Field(default=None, description="model labels file path (Optional)")
    config: Path = Field(default=None, description="model config file path (Optional)")
    labels: List[str] = Field(
        default=None,
        description="model labels parsed into a list of strings",
        repr=False,
        exclude=True,
    )

    detection_options: ModelOptions = Field(
        default_factory=ModelOptions, description="Default Configuration for the model"
    )

    @validator("framework", pre=True)
    def framework_validator(cls, v):
        if isinstance(v, str):
            v = v.lower()
            v = ModelFrameWork(v)
        return v

    @validator("model_type", pre=True)
    def model_type_validator(cls, v):
        if isinstance(v, str):
            v = v.lower()
            v = ModelType(v)
        return v

    @validator("config", "input", "classes", pre=True, always=True)
    def str_to_path(cls, v, values, field: ModelField) -> Optional[Path]:
        # logger.debug(f"validating {field.name} - {v = } -- {type(v) = } -- {values = }")
        msg = f"{field.name} must be a path or a string of a path"
        model_name = values.get("name", "Unknown Model")
        lp = f"Model Name: {model_name} ->"

        if v is None:
            if field.name == "input":
                raise ValueError(f"{msg}, not 'None'")
            logger.debug(f"{lp} {field.name} is None, passing as it is Optional")
            return v
        elif not isinstance(v, (Path, str)):
            raise ValueError(msg)
        elif isinstance(v, str):
            # logger.debug(
            #     f"Attempting to convert {field.name} string '{v}' to Path object"
            # )
            v = Path(v)
        if field.name == "config":
            if values["input"].suffix == ".weights":
                msg = f"'{field.name}' is required when 'input' is a DarkNet .weights file"

        msg = f"{field.name} is required"
        assert v, f"{msg}, it is set as '{v}' (Not Defined)!"
        assert v.exists(), f"{msg}, it does not exist"
        assert v.is_file(), f"{msg}, it is not a file"
        assert isinstance(v, Path), f"{field.name} is not a Path object"
        # logger.debug(
        #     f"DBG>>> {field.name} is a validated Path object -> RETURNING {type(v) = } --> {v = }"
        # )
        return v

    @validator("labels", always=True)
    def _validate_labels(cls, v, values, field: ModelField) -> Optional[List[str]]:
        # logger.debug(f"validating {field.name} - {v = } -- {type(v) = } -- {values = }")
        model_name = values.get("name", "Unknown Model")
        lp = f"Model Name: {model_name} ->"
        if not v:
            if not (labels_file := values["classes"]):
                logger.debug(
                    f"{lp} 'classes' is not defined. Using *default* COCO 2017 class labels"
                )
                from zm_mlapi.ml.coco_2017 import COCO_NAMES

                v = COCO_NAMES
            else:
                logger.debug(
                    f"'classes' is defined. Parsing '{labels_file}' into a list of strings for class identification"
                )
                assert isinstance(
                    labels_file, Path
                ), f"{field.name} is not a Path object"
                assert labels_file.exists(), "labels_file does not exist"
                assert labels_file.is_file(), "labels_file is not a file"
                with labels_file.open(mode="r") as f:
                    f: IO
                    v = f.read().splitlines()
        assert isinstance(v, list), f"{field.name} is not a list"
        return v


class ALPRService(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    DEFAULT = LOCAL


class ALPRModelConfig(MLModelConfig):
    alpr_key: str = Field(None, description="ALPR Cloud API Key")
    alpr_service: ALPRService = Field(ALPRService.LOCAL, description="ALPR Service Type")
    alpr_url: str = Field(None, description="ALPR Cloud API URL")


class ModelConfigFromFile:
    file: Path
    raw: str
    parsed_raw: dict
    parsed: dict

    def __repr__(self):
        return f"{self.__class__.__name__}({self.file})->{self.parsed.get('models')}"

    def __init__(self, cfg_file: Union[Path, str]):
        self.raw = ""
        self.parsed_raw = {}
        self.parsed = {}
        if not isinstance(cfg_file, (Path, str)):
            raise TypeError(
                f"ModelConfig: The configuration file must be a string or a Path object."
            )
        if isinstance(cfg_file, str):
            cfg_file = Path(cfg_file)
        if cfg_file.exists() and cfg_file.is_file():
            self.file = cfg_file
            with open(self.file, "r") as f:
                self.raw = f.read()
            self.parsed_raw = self.model_config_parser(self.raw)
            self.parsed = self.substitution_vars(self.parsed_raw)
        else:
            logger.error(
                f"ModelConfig: The configuration file '{cfg_file}' does not exist or is not a file."
            )

    def get_model_config(self):
        return self.parsed

    def get_model_config_str(self):
        return self.raw

    def get_model_config_raw(self):
        return self.parsed_raw

    def model_config_parser(self, cfg_str: str) -> dict:
        """Parse the YAML model configuration file.

        Args:
            cfg_str (str): Configuration YAML file as a string.

        """
        cfg: dict = {}
        import yaml

        try:
            cfg = yaml.safe_load(cfg_str)
        except yaml.YAMLError as e:
            logger.error(
                f"model_config_parser: Error parsing the YAML configuration file!"
            )
            raise e
        return cfg

    def substitution_vars(self, cfg: Dict[str, str]) -> Dict[str, str]:
        """Substitute variables in the configuration file.

        Args:
            cfg (Dict[str, str]): The configuration dictionary.

        Returns:
            Dict[str, str]: The configuration dictionary with variables substituted.
        """
        # turn dict into a string to use regex search/replace globally instead of iterating over the dict
        cfg_str = str(cfg)
        # find all variables in the string
        import re

        var_list = re.findall(r"\$\{(\w+)\}", cfg_str)
        if var_list:
            var_list = list(set(var_list))
            logger.debug(
                f"substitution_vars: Found the following variables in the configuration file: {var_list}"
            )
            # substitute variables
            for var in var_list:
                num_var = len(re.findall(f"\${{{var}}}", cfg_str))
                if var in cfg:
                    logger.debug(
                        f"substitution_vars: Found {num_var} occurrence{'s' if num_var != 1 else ''} of '${{{var}}}', "
                        f"Substituting with value '{cfg[var]}'"
                    )
                    cfg_str = cfg_str.replace(f"${{{var}}}", cfg[var])
                else:
                    logger.warning(
                        f"substitution_vars: The variable '${{{var}}}' is not defined."
                    )
            from ast import literal_eval

            return literal_eval(cfg_str)
        else:
            logger.debug(
                f"substitution_vars: No variables for substituting in the configuration file."
            )
        return cfg


class LockSetting(BaseModel):
    max: int = Field(1, description="Maximum number of parallel processes")
    timeout: int = Field(30, description="Timeout in seconds for acquiring a lock")


class LockSettings(BaseModel):
    lock_dir: str = Field(f"{tempfile.gettempdir()}/zm_mlapi/locks", description="Directory for lock files (Default is Systems temporary directory)")
    gpu: LockSetting = Field(default_factory=LockSetting, description="GPU Lock Settings")
    cpu: LockSetting = Field(default_factory=LockSetting, description="CPU Lock Settings")
    tpu: LockSetting = Field(default_factory=LockSetting, description="TPU Lock Settings")


class MLLocks:
    locks: Dict[str, portalocker.BoundedSemaphore] = {}

    def __next__(self):
        return next(self.locks.__iter__())

    def __contains__(self, processor: str) -> bool:
        return processor in self.locks

    def __get__(self, instance, owner):
        return self.locks

    def __set__(self, instance, value):
        raise SyntaxError("Cannot set locks")

    def __getitem__(self, processor: str) -> portalocker.BoundedSemaphore:
        return self.get_lock(processor)

    def __iter__(self):
        return iter(self.locks)

    def __len__(self):
        return len(self.locks)

    def __repr__(self):
        return f"MLLocks({self.locks})"

    def __str__(self):
        return f"MLLocks({self.locks})"

    def __init__(self, locks: LockSettings):
        self.locks = {
            "gpu": portalocker.BoundedSemaphore(
                locks.gpu.max,
                directory=locks.lock_dir,
                name="zm_mlapi-gpu",
                timeout=locks.gpu.timeout,
            ),
            "cpu": portalocker.BoundedSemaphore(
                locks.cpu.max,
                directory=locks.lock_dir,
                name="zm_mlapi-cpu",
                timeout=locks.cpu.timeout,
            ),
            "tpu": portalocker.BoundedSemaphore(
                locks.tpu.max,
                directory=locks.lock_dir,
                name="zm_mlapi-tpu",
                timeout=locks.tpu.timeout,
            ),
        }

    def get_lock(self, processor: str) -> portalocker.BoundedSemaphore:
        if processor in self.locks:
            return self.locks[processor]
        raise SyntaxError(f"Invalid processor: {processor}")


class Settings(BaseSettings):

    model_config: Path = Field(
        ..., description="Path to the model configuration YAML file"
    )
    log_dir: Path = Field(
        default=f"{tempfile.gettempdir()}/zm_mlapi/logs", description="Logs directory"
    )
    host: str = Field(default="0.0.0.0", description="Interface IP to listen on")
    port: int = Field(default=5000, description="Port to listen on")
    jwt_secret: str = Field(default="CHANGE ME", description="JWT signing key")
    file_logger: bool = Field(False, description="Enable file logging")
    file_log_name: str = Field("zm_mlapi.log", description="File log name")

    reload: bool = Field(
        default=False, description="Uvicorn reload - For development only"
    )
    debug: bool = Field(default=False, description="Debug mode - For development only")

    models: ModelConfigFromFile = Field(
        None, description="ModelConfig object", exclude=True, repr=False
    )
    available_models: List[MLModelConfig] = Field(None, description="Available models")
    locks: LockSettings = Field(default_factory=LockSettings, description="Lock Settings", repr=False)

    ml_locks: MLLocks = Field(None, description="ML Locks class")

    class Config:
        env_nested_delimiter = '__'

    @validator("ml_locks")
    def ml_locks_validator(cls, v, values):
        locks = values.get("locks")
        if locks:
            v = MLLocks(values["locks"])
        logger.info(f"Settings._ml_locks_validator: {locks = }")
        return v

    @validator("available_models")
    def validate_available_models(cls, v, values):
        models = values.get("models")
        if models:
            v = []
            for model in models.parsed.get("models"):
                v.append(MLModelConfig(**model))
        return v

    @validator("debug")
    def check_debug(cls, v, values):
        if v:
            values["reload"] = True
            logger.info(
                f"Debug mode is enabled. The server will reload on every code change and logging level is set to DEBUG."
            )
            logger.setLevel(logging.DEBUG)
        return v

    @validator("models")
    def _validate_model_config(cls, v, field, values):
        model_config = values["model_config"]
        logger.debug(f"parsing model config: {model_config}")
        v = ModelConfigFromFile(model_config)
        return v

    @validator("model_config", "log_dir", pre=True, always=True)
    def _validate_path(cls, v, values, field: ModelField) -> Optional[Path]:
        """Take a Path or str and return a validated Path object"""
        logger.debug(f"validating {field.name} - {v = } -- {type(v) = } -- {values = }")
        if not v:
            raise ValueError(f"{field.name} is required")
        assert isinstance(v, (Path, str)), f"{field.name} must be a Path or str"
        if isinstance(v, str):
            v = Path(v)
        if field.name == "model_config":
            assert v.exists(), "model_config does not exist"
            assert v.is_file(), "model_config is not a file"
        elif field.name == "log_dir":
            if not v.exists():
                logger.warning(
                    f"{field.name} directory: {v} does not exist, creating..."
                )
                v.mkdir(parents=True, exist_ok=True)
            assert v.is_dir(), "log_dir is not a directory"
        return v


class APIDetector:
    """ML detector API class.
    Specify a processor type and then load the model into processor memory. run an inference on the processor
    """

    model_config: MLModelConfig
    model_options: ModelOptions
    model_processor: ModelProcessor

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model_config} <--> Options: {self.model_options})"

    def __get__(self, instance, owner):
        return self

    def __init__(
        self,
        model: MLModelConfig,
        options: Optional[
            Union[ModelOptions, FaceModelOptions, ALPRModelOptions]
        ] = None,
    ):
        self.model_options = options
        self.model_config = model
        if not self.model_options.processor:
            if model.framework == ModelFrameWork.CORAL:
                logger.warning(
                    f"Using default processor for {model.framework} -> {ModelProcessor.TPU}"
                )
                self.model_options.processor = ModelProcessor.TPU
            else:
                self.model_options.processor = ModelProcessor.CPU
            logger.debug(
                f"no processor specified, using {model.framework} default: {self.model_options.processor}"
            )
        self.model_processor = self.model_options.processor
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the model"""
        if not self.is_processor_available():
            raise RuntimeError(
                f"{self.model_processor} is not available on this system"
            )

        if self.model_config.framework == ModelFrameWork.OPENCV:
            from zm_mlapi.ml.opencv import OpenCVDetector as OpenCVDetector

            self.model = OpenCVDetector(self.model_config, self.model_options)

    def set_model_options(self, options: ModelOptions):
        if not options:
            logger.warning("No options specified, using existing options")
            return
        if options != self.model_options:
            logger.debug(f"updating (reloading) model with options: {options}")
            self.model_options = options
            self._load_model()

    def is_processor_available(self):
        """Check if the processor is available"""
        available = False
        if self.model_processor == ModelProcessor.CPU:
            if self.model_config.framework == ModelFrameWork.CORAL:
                logger.error(
                    f"{self.model_processor} is not supported for {self.model_config.framework}"
                )
            else:
                available = True

        elif self.model_processor == ModelProcessor.TPU:
            if self.model_config.framework == ModelFrameWork.CORAL:
                try:
                    import pycoral
                except ImportError:
                    logger.warning(
                        "pycoral not installed, cannot load any models that use the TPU processor"
                    )
                else:
                    tpus = pycoral.utils.edgetpu.list_edge_tpus()
                    if tpus:
                        available = True
                    else:
                        logger.warning(
                            "No TPU devices found, cannot load any models that use the TPU processor"
                        )
            else:
                logger.warning("TPU processor is only available for Coral models!")
        elif self.model_processor == ModelProcessor.GPU:
            if self.model_config.framework == ModelFrameWork.OPENCV:
                try:
                    import cv2.cuda
                except ImportError:
                    logger.warning(
                        "OpenCV does not have CUDA enabled/compiled, cannot load any models that use the GPU processor"
                    )
                else:
                    if not (cuda_devices := cv2.cuda.getCudaEnabledDeviceCount()):
                        logger.warning(
                            "No CUDA devices found, cannot load any models that use the GPU processor"
                        )
                    else:
                        logger.debug(f"Found {cuda_devices} CUDA device(s)")
                        available = True
            elif self.model_config.framework == ModelFrameWork.TENSORFLOW:
                try:
                    import tensorflow as tf
                except ImportError:
                    logger.warning(
                        "tensorflow not installed, cannot load any models that use tensorflow GPU processor"
                    )
                else:
                    if not tf.config.list_physical_devices("GPU"):
                        logger.warning(
                            "No CUDA devices found, cannot load any models that use the GPU processor"
                        )
                    else:
                        available = True
            elif self.model_config.framework == ModelFrameWork.PYTORCH:
                try:
                    import torch
                except ImportError:
                    logger.warning(
                        "pytorch not installed, cannot load any models that use pytorch GPU processor"
                    )
                else:
                    if not torch.cuda.is_available():
                        logger.warning(
                            "No CUDA devices found, cannot load any models that use the GPU processor"
                        )
                    else:
                        available = True
        return available

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect objects in the image"""
        assert self.model, "model not loaded"
        return self.model.detect(image)


class GlobalConfig(BaseModel):
    available_models: List[MLModelConfig] = Field(
        default_factory=dict, description="Available models, call by ID"
    )
    settings: Settings = Field(
        default=None, description="Global settings from ENVIRONMENT"
    )

    detectors: List[APIDetector] = Field(
        default_factory=list, description="Loaded detectors"
    )

    class Config:
        arbitrary_types_allowed = True


class MLSequence(BaseModel):
    """ML Sequence class, used to sequence multiple detectors"""
    id: int = Field(default=0, description="Sequence ID")

    name: str = Field(
        default="default", description="Name of the sequence, used for logging"
    )
    detectors: List[APIDetector] = Field(
        default_factory=list, description="Detectors to run in sequence"
    )
    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name} <--> Detectors: {self.detectors})"

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect objects in the image using threads"""
        from concurrent.futures import ThreadPoolExecutor
        threads = []
        results = {}
        for detector in self.detectors:
            threads.append(
                ThreadPoolExecutor(max_workers=10).submit(detector.detect, image=image)
            )

        for thread in threads:

            results.update(thread.result(timeout=10))
        return results
