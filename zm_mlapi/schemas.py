import uuid
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Union, List, Dict, Optional, IO

from pydantic import BaseModel, Field, validator, BaseSettings
from pydantic.fields import ModelField

logger = getLogger("zm_mlapi")


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
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4, description="Unique ID of the model"
    )
    name: str = Field(..., description="model name")
    input_file: Path = Field(..., description="model file path")
    labels_file: Path = Field(
        default=None, description="model labels file path (Optional)"
    )
    config_file: Path = Field(
        default=None, description="model config file path (Optional)"
    )
    labels: List[str] = Field(
        default=None, description="model labels parsed into a list of strings", repr=False, exclude=True
    )

    # validators
    # _validate_input_file = validator("input_file", allow_reuse=True)(str_to_path)
    # _validate_labels_file = validator("labels_file", allow_reuse=True)(str_to_path)
    # _validate_config_file = validator("config_file", allow_reuse=True)(str_to_path)

    @validator("config_file", "input_file", "labels_file", pre=True, always=True)
    def str_to_path(cls, v, values, field: ModelField) -> Optional[Path]:
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
            logger.debug(
                f"Attempting to convert {field.name} string '{v}' to Path object"
            )
            v = Path(v)
        if field.name == "config_file":
            if values["input_file"].suffix == ".weights":
                msg = f"{field.name} is required when input_file is a .weights file"
                if not v:
                    raise ValueError(f"{msg}, it is set as 'None' (Not Configured)!")
                elif not v.exists():
                    raise ValueError(f"{msg}, it does not exist")
                elif not v.is_file():
                    raise ValueError(f"{msg}, it is not a file")

        msg = f"{field.name} is required"
        if not v:
            raise ValueError(f"{msg}, it is set as 'None' or empty (Not Configured)!")
        elif not v.exists():
            raise ValueError(f"{msg}, it does not exist")
        elif not v.is_file():
            raise ValueError(f"{msg}, it is not a file")
        logger.debug(
            f"DBG>>> {field.name} is a validated Path object -> RETURNING {type(v) = } --> {v = }"
        )
        assert isinstance(v, Path), f"{field.name} is not a Path object"
        return v

    @validator("labels", always=True)
    def _validate_labels(cls, v, values, field: ModelField) -> Optional[List[str]]:
        logger.debug(f"validating {field.name} - {v = } -- {type(v) = } -- {values = }")
        if not v:
            if not (labels_file := values["labels_file"]):
                logger.debug(
                    f"'labels' and 'labels_file' are unset. Using *default* COCO labels"
                )
                from zm_mlapi.ml.coco_2017 import COCO_NAMES

                v = COCO_NAMES
            else:
                logger.debug(
                    f"'labels' is unset and 'labels_file' is set. Parsing 'labels_file' ({labels_file}) into a list of strings"
                )
                assert isinstance(
                    values["labels_file"], Path
                ), f"{field.name} is not a Path object"
                assert values["labels_file"].exists(), "labels_file does not exist"
                assert values["labels_file"].is_file(), "labels_file is not a file"
                with values["labels_file"].open(mode="r") as f:
                    f: IO
                    v = f.read().splitlines()
        assert isinstance(v, list), f"{field.name} is not a list"
        return v


class ModelSequence(BaseModel):
    based_on: uuid.UUID = Field(..., description="UUID of the available model to base the sequence on")
    _source_model_data: AvailableModel = Field(None, description="The source model", repr=False, exclude=True)

    id: int = Field(..., description="Unique ID of the model sequence (duplicates will be overwritten)")
    name: str = Field(None, description="Name that will be shown in the annotated frame")
    enabled: bool = Field(True, description="Enable/Disable the model")
    framework: ModelFrameWork = Field(ModelFrameWork.OPENCV, description="model framework")
    processor: ModelProcessor = Field(ModelProcessor.CPU, description="Processor used for this model")
    # Model input size (image will be resized if needed)
    height: int = Field(416, description="model input height (resized before inference)")
    width: int = Field(416, description="model input width (resized before inference)")
    fp_16: bool = Field(False, description="Floating Point 16 target (OpenCV - CUDA/cuDNN[GPU] only)")
    square_image: bool = Field(False, description="square image by zero padding (if needed)")


    @validator("based_on", always=True, pre=True)
    def _validate_based_on(cls, v, values, field: ModelField) -> Optional[AvailableModel]:
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


class Settings(BaseSettings):
    from tempfile import gettempdir

    data_dir: Union[str, Path] = Field(
        default=f"{gettempdir()}/zm_mlapi",
        description="Base data path for the running server",
    )
    model_dir: Union[str, Path] = Field(
        default=f"{data_dir}/models", description="Machine Learning models directory"
    )
    log_dir: Union[str, Path] = Field(
        default=f"{data_dir}/logs", description="Logs directory"
    )
    host: str = Field(default="0.0.0.0", description="Interface IP to listen on")
    port: int = Field(default=5000, description="Port to listen on")
    jwt_secret: str = Field(default="CHANGE ME", description="JWT signing key")

    reload: bool = Field(
        default=False, description="Uvicorn reload - For development only"
    )
    debug: bool = Field(default=False, description="Debug mode - For development only")

    db_driver: str = Field("mysql+pymysql", description="Database driver (Default: mysql+pymysql)")

    db_user: str = Field("zmuser", description="Database user (Default: zmuser)")
    db_pass: str = Field("zmpass", description="Database user password (Default: zmpass)")
    db_host: str = Field("localhost", description="Database host (Default: localhost)")
    db_name: str = Field("zmai", description="Database name (Default: zmai)")

    @validator("data_dir", "model_dir", "log_dir", "db_dir", pre=True, always=True)
    def create_dirs(cls, v, values, field, **kwargs):
        v = Path(v)
        if not v.exists():
            logger.warning(f"{field.name} directory: {v} does not exist, creating...")
            v.mkdir(parents=True, exist_ok=True)
        else:
            logger.debug(f"{field.name} directory: {v} exists")
        return v

    def parse_model_dir(self) -> Dict[str, Optional[AvailableModel]]:
        """Parse the model directory to get the available models"""
        import glob

        available_models = {}
        if self.model_dir.is_dir():
            all_files = glob.glob(self.model_dir.as_posix() + "/*")
            for _dir in all_files:
                if Path(_dir).is_dir():
                    model = self._parse_model(_dir)
                    if model:
                        available_models.update(model)
        else:
            logger.warning(f"{self.model_dir} is not a directory")
        logger.debug(
            f"Found {len(available_models)} models in {self.model_dir} -> {available_models = }"
        )
        return available_models

    def _parse_model(self, model_dir: str) -> Optional[Dict[str, AvailableModel]]:
        """Parse a model directory to get the available model"""
        lp: str = "model config:"
        available_models = {}
        model_dir = Path(model_dir)
        if model_dir.exists():
            if model_dir.is_dir():
                model_config_file = model_dir / "config.json"
                if model_config_file.exists():
                    if model_config_file.is_file():
                        with model_config_file.open() as f:
                            import json

                            model_config = json.load(f)
                        logger.debug(
                            f"{lp} {model_config_file} -> {model_config} - {len(model_config) = }"
                        )
                        for model_name, model_cfg in model_config.items():
                            logger.debug(f"{lp} {type(model_cfg) = } -- {model_cfg = }")
                            model_cfg["name"] = model_name
                            model_cfg["input_file"] = (
                                model_dir / model_cfg["input_file"]
                            )
                            model_cfg["config_file"] = (
                                model_dir / model_cfg["config_file"]
                            )
                            model_cfg["labels_file"] = (
                                model_dir / model_cfg["labels_file"]
                                if "labels_file" in model_cfg
                                else None
                            )
                            model = AvailableModel(**model_cfg)
                            available_models[str(model.id)] = model
                        return available_models
                    else:
                        logger.warning(
                            f"{lp} {model_config_file} is not a file, skipping..."
                        )
                else:
                    logger.warning(f"{lp} {model_config_file} does not exist, skipping...")
            else:
                logger.warning(f"{lp} {model_dir} is not a directory, skipping...")
        else:
            logger.warning(f"{lp} {model_dir} does not exist, skipping...")
        return None


class GlobalConfig(BaseModel):
    available_models: Dict[str, AvailableModel] = Field(
        default_factory=dict, description="Available models, call by ID"
    )
    settings: Settings = Field(
        default=None, description="Global settings from ENVIRONMENT"
    )


    class Config:
        arbitrary_types_allowed = True
