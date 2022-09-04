import uuid
from enum import Enum
import logging
from pathlib import Path
from typing import Union, List, Dict, Optional, IO

from fastapi import UploadFile, File, Query
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
    object = OBJECT
    face = FACE
    alpr = ALPR


class ModelFrameWork(str, Enum):
    OPENCV = "opencv"
    CORAL = "coral"
    PYCORAL = CORAL
    VINO = "openvino"
    OPENVINO = VINO
    TENSORFLOW = "tensorflow"
    TF = TENSORFLOW
    PYTORCH = "pytorch"
    DEEPFACE = "deepface"
    OPENALPR_CLOUD = "openalpr_cloud"
    OPENALPR = "openalpr"
    PLATERECOGNIZER = "platerecognizer"


class ModelProcessor(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    CLOUD = "cloud"
    cpu = CPU
    gpu = GPU
    tpu = TPU
    cloud = CLOUD


class DetectionResult(BaseModel):
    type: ModelType = None
    label: str = None
    confidence: float = None
    bounding_box: tuple[Union[float, int]] = None
    model_name: str = None


class AvailableModel(BaseModel):
    class DefaultConfig(BaseModel):
        class Thresholds(BaseModel):
            confidence: float = Field(
                0.5, ge=0.0, le=1.0, descritpiton="Confidence Threshold"
            )
            nms: float = Field(
                0.4, ge=0.0, le=1.0, description="Non-Maximum Suppression Threshold"
            )

        height: int = Field(
            416, ge=1, description="Height of the input image (resized for model)"
        )
        width: int = Field(
            416, ge=1, description="Width of the input image (resized for model)"
        )
        square: bool = Field(False, description="Zero pad the image to be a square")
        thresholds: Thresholds = Field(
            default_factory=Thresholds, description="Thresholds for the model"
        )

    id: int = Field(default_factory=uuid.uuid4, description="Unique ID of the model")
    name: str = Field(..., description="model name")
    input: Path = Field(..., description="model file/dir path")
    enabled: bool = Field(True, description="model enabled")
    description: str = Field(None, description="model description")
    classes: Path = Field(default=None, description="model labels file path (Optional)")
    config: Path = Field(default=None, description="model config file path (Optional)")
    labels: List[str] = Field(
        default=None,
        description="model labels parsed into a list of strings",
        repr=False,
        exclude=True,
    )

    default_config: DefaultConfig = Field(
        default_factory=DefaultConfig, description="Default Configuration for the model"
    )

    @validator("config", "input", "classes", pre=True, always=True)
    def str_to_path(cls, v, values, field: ModelField) -> Optional[Path]:
        logger.debug(f"validating {field.name} - {v = } -- {type(v) = } -- {values = }")
        msg = f"{field.name} must be a path or a string of a path"

        if v is None:
            if field.name == "input":
                raise ValueError(f"{msg}, not 'None'")
            logger.debug(f"{field.name} is None, passing as it is Optional")
            return v
        elif not isinstance(v, (Path, str)):
            raise ValueError(msg)
        elif isinstance(v, str):
            logger.debug(
                f"Attempting to convert {field.name} string '{v}' to Path object"
            )
            v = Path(v)
        if field.name == "config":
            if values["input"].suffix == ".weights":
                msg = f"'{field.name}' is required when 'input' is a DarkNet .weights file"

        msg = f"{field.name} is required"
        assert v, f"{msg}, it is set as '{v}' (Not Defined)!"
        assert v.exists(), f"{msg}, it does not exist"
        assert v.is_file(), f"{msg}, it is not a file"
        assert isinstance(v, Path), f"{field.name} is not a Path object"
        logger.debug(
            f"DBG>>> {field.name} is a validated Path object -> RETURNING {type(v) = } --> {v = }"
        )
        return v

    @validator("labels", always=True)
    def _validate_labels(cls, v, values, field: ModelField) -> Optional[List[str]]:
        logger.debug(f"validating {field.name} - {v = } -- {type(v) = } -- {values = }")
        if not v:
            if not (labels_file := values["classes"]):
                logger.debug(
                    f"'classes' is not defined. Using *default* COCO 2017 class labels"
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

    db_driver: str = Field(
        "mysql+pymysql", description="Database driver (Default: mysql+pymysql)"
    )

    db_user: str = Field("zmuser", description="Database user (Default: zmuser)")
    db_pass: str = Field(
        "zmpass", description="Database user password (Default: zmpass)"
    )
    db_host: str = Field("localhost", description="Database host (Default: localhost)")
    db_port: int = Field(3306, description="Database port (Default: 3306)")
    db_name: str = Field("zmai", description="Database name (Default: zmai)")
    db_options: str = Field(None, description="Database options (Default: None)")
    # mysql+pymysql://<username>:<password>@<host>/<dbname>[?<options>]
    db_connection_string: str = Field(
        None, description="SQLAlchemy database connection string (Default: None)"
    )

    models: ModelConfig = Field(None, description="ModelConfig object", exclude=True)
    available_models: List[AvailableModel] = Field(None, description="Available models")

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
    @validator("db_connection_string")
    def _validate_db_connection_string(cls, v, values):
        if not v:
            v = f"{values['db_driver']}://{values['db_user']}:{values['db_pass']}@{values['db_host']}:{values['db_port']}/{values['db_name']}"
            if values["db_options"]:
                v += f"?{values['db_options']}"
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

    def parse_model_config(self):
        """Parse the model configuration YAML file"""
        logger.debug(f"parsing model config: {self.model_config}")
        self.models = ModelConfig(self.model_config)


class GlobalConfig(BaseModel):
    available_models: Dict[str, AvailableModel] = Field(
        default_factory=dict, description="Available models, call by ID"
    )
    settings: Settings = Field(
        default=None, description="Global settings from ENVIRONMENT"
    )

    class Config:
        arbitrary_types_allowed = True


class Detector:
    """Base class for detectors"""

    def __init__(self, model_config: AvailableModel):
        self.model_config = model_config
        self.model = None
        self.input_size = None
        self.labels = None
        self.confidence = None
        self.device = None
        self._load_model()

    def _load_model(self):
        """Load the model"""
        raise NotImplementedError

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image"""
        raise NotImplementedError

    def _postprocess(self, predictions: np.ndarray) -> List[Detection]:
        """Postprocess the predictions"""
        raise NotImplementedError

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect objects in the image"""
        raise NotImplementedError

class DetectionRequest(BaseModel):
    image: UploadFile = File(..., description="Image to run the ML model on")
    min_conf: float = Field(
        0.2,
        le=1.0,
        ge=0.0,
        description="Minimum confidence to return a detection expressed as a float between 0.0 and 1.0",
    )
    nms_threshold: float = Field(
        0.4,
        le=1.0,
        ge=0.0,
        description="Non Max Suppression (NMS) threshold to apply to the model results expressed as a float between 0.0 and 1.0",
    )
