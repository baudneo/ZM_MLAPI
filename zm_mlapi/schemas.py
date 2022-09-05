import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union, List, Dict, Optional, IO

import numpy as np
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


class DetectionResult(BaseModel):
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


class AvailableModel(BaseModel):
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


class ModelSequence(BaseModel):
    based_on: uuid.UUID = Field(
        ..., description="UUID of the available model to base the sequence on"
    )
    __source_model__: AvailableModel = Field(
        None, description="The source model", repr=False, exclude=True
    )

    id: int = Field(
        ...,
        description="Unique ID of the model sequence (duplicates will be overwritten)",
    )
    name: str = Field(
        None, description="Name that will be shown in the annotated frame"
    )
    enabled: bool = Field(True, description="Enable/Disable the model")
    framework: ModelFrameWork = Field(
        ModelFrameWork.OPENCV, description="model framework"
    )
    processor: ModelProcessor = Field(
        ModelProcessor.CPU, description="Processor used for this model"
    )
    # Model input size (image will be resized if needed)
    height: int = Field(
        416, description="model input height (resized before inference)"
    )
    width: int = Field(416, description="model input width (resized before inference)")
    fp_16: bool = Field(
        False, description="Floating Point 16 target (OpenCV - CUDA/cuDNN[GPU] only)"
    )
    square_image: bool = Field(
        False, description="square image by zero padding (if needed)"
    )

    @validator("based_on", always=True, pre=True)
    def _validate_based_on(
        cls, v, values, field: ModelField
    ) -> Optional[AvailableModel]:
        from zm_mlapi.app import get_global_config

        g = get_global_config()
        logger.debug(f"validating {field.name} - {v = } -- {type(v) = } -- {values = }")
        if not v:
            raise ValueError(f"{field.name} is required")
        if not isinstance(v, uuid.UUID):
            raise ValueError(f"{field.name} must be a UUID")
        if not (model := g.available_models.get(v)):
            raise ValueError(f"{field.name} must be a valid UUID of an existing model")
        values["_source_model_data"] = model
        return v

    @validator("fp_16")
    def check_fp_16(cls, v, values):
        if (
            values["framework"] != ModelFrameWork.OPENCV
            and values["processor"] != ModelProcessor.GPU
        ):
            logger.warning(
                f"fp_16 is not supported for {values['framework']} and {values['processor']}. "
                f"Only OpenCV->GPU is supported."
            )
            v = False
        return v


class ModelConfig:
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


class Settings(BaseSettings):
    from tempfile import gettempdir

    model_config: Path = Field(
        ..., description="Path to the model configuration YAML file"
    )
    log_dir: Path = Field(
        default=f"{gettempdir()}/zm_mlapi/logs", description="Logs directory"
    )
    host: str = Field(default="0.0.0.0", description="Interface IP to listen on")
    port: int = Field(default=5000, description="Port to listen on")
    jwt_secret: str = Field(default="CHANGE ME", description="JWT signing key")

    reload: bool = Field(
        default=False, description="Uvicorn reload - For development only"
    )
    debug: bool = Field(default=False, description="Debug mode - For development only")

    models: ModelConfig = Field(None, description="ModelConfig object", exclude=True)
    available_models: List[AvailableModel] = Field(None, description="Available models")

    # def __init__(self, args, **kwargs):
    #     logger.info(f"Settings: {args = } -- {kwargs = }")
    #     print(f"Settings: {args = } -- {kwargs = }")
    #     super().__init__(args, **kwargs)

    @validator("available_models")
    def validate_available_models(cls, v, values):
        models = values.get("models")
        if models:
            v = []
            for model in models.parsed.get("models"):
                v.append(AvailableModel(**model))
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
        v = ModelConfig(model_config)
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
    """Base class for detectors
    Specify a processor type and then load the model into processor memory. run an inference on the processor
    """

    model_config: AvailableModel
    model_options: ModelOptions
    model_processor: ModelProcessor

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model_config}) Options: {self.model_options}"

    def __init__(
        self,
        model: AvailableModel,
        options: Optional[ModelOptions] = None,
    ):
        self.model_options = options
        self.model_config = model
        if not self.model_options.processor:
            if model.framework == ModelFrameWork.CORAL:
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
            from zm_mlapi.ml.opencv import Detector as OpenCVDetector

            self.model = OpenCVDetector(self.model_config, self.model_options)

    def set_model_options(self, options: ModelOptions):
        if options != self.model_options:
            self.model_options = options
            self._load_model()
        self.model_options = options

    def is_processor_available(self):
        """Check if the processor is available"""
        available = False
        if self.model_processor == ModelProcessor.CPU:
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
                    if not cv2.cuda.getCudaEnabledDeviceCount():
                        logger.warning(
                            "No CUDA devices found, cannot load any models that use the GPU processor"
                        )
                    else:
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

    def detect(self, image: np.ndarray):
        """Detect objects in the image"""
        assert self.model, "model not loaded"
        return self.model.detect(image)


class GlobalConfig(BaseModel):
    available_models: List[AvailableModel] = Field(
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
