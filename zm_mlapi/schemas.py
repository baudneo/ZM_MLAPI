from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Union, List, Dict

from pydantic import BaseModel, Field, validator, BaseSettings
from pydantic.fields import ModelField

logger = getLogger()


class ModelType(Enum):
    OBJECT = "object"
    FACE = "face"
    ALPR = "alpr"
    object = OBJECT
    face = FACE
    alpr = ALPR


class ModelFrameWork(Enum):
    OPENCV = "opencv"
    CORAL = "coral"
    PYCORAL = CORAL
    VINO = "openvino"
    OPENVINO = VINO
    TENSORFLOW = "tensorflow"
    TF = TENSORFLOW
    PYTORCH = "pytorch"


class ModelProcessor(Enum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    cpu = CPU
    gpu = GPU
    tpu = TPU


class DetectionResult(BaseModel):
    type: ModelType = None
    label: str = None
    confidence: float = None
    bounding_box: tuple[Union[float, int]] = None
    model_name: str = None


class Strategy(Enum):
    FIRST = "first"
    MOST_LABELS = "most_labels" or "most labels" or "most"
    MOST_CONFIDENT = "most confident" or "most_confident"
    MOST_UNIQUE_LABELS = "most_unique_labels" or "most unique labels" or "most_unique" or "most unique" or "unique"
    JOIN = "join"


class ModelSequence(BaseModel):
    name: str
    model_file: Path
    labels_file: Path = None

    show_name: str = None
    enabled: bool = True
    framework: ModelFrameWork = ModelFrameWork.OPENCV
    processor: ModelProcessor = ModelProcessor.CPU
    config_file: Path = None
    # Parsed labels file
    labels: List[str] = None
    height: int = 416
    width: int = 416
    # Floating Point 16 target (OpenCV - GPU only)
    fp_16: bool = False
    square_image: bool = Field(False, description="square image by zero padding")

    filters: dict = None

    @validator("config_file", "model_file", "labels_file", pre=True, always=True)
    def str_to_path(cls, path, field: ModelField) -> Path:
        msg = f"{field.name} must be a path or a string of a path"
        if path is None:
            if field.name == "config_file":
                pass
            else:
                raise ValueError(f"{msg}, not 'None'")
        elif not isinstance(path, (Path, str)):
            raise ValueError(msg)
        elif isinstance(path, str):
            logger.debug(f"Attempting to convert {field.name} string '{path}' to Path object")
            try:
                path = Path(path)
            except Exception as e:
                raise ValueError(f"{msg}, {path} is not a valid path") from e
            else:
                if not path.exists():
                    raise ValueError(f"{msg}, {path} does not exist")
                elif not path.is_file():
                    raise ValueError(f"{msg}, {path} is not a file")
        # path is a Path() object
        return path

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

    @validator("config_file")
    def check_config_file(cls, v, values):
        # If it is a .weights file there needs to be a config file
        if values["model_file"].suffix == ".weights":
            msg = "config_file is required when model_file is a .weights file"
            if not v:
                raise ValueError(f"{msg}, it is set as 'None' (Not Configured)!")
            elif not v.exists():
                raise ValueError(f"{msg}, it does not exist")
            elif not v.is_file():
                raise ValueError(f"{msg}, it is not a file")
        return v


class DetectionModel(BaseModel):
    type: ModelType = Field(
        ..., description="model type (object|face|alpr)"
    )
    detection_pattern: str = Field(..., description="detection pattern in regex format")
    strategy: Strategy = Field(
        default=Strategy.FIRST,
        description="sequence strategy (first|most_labels|most_confident|most_unique_labels|join)",
    )
    sequence: ModelSequence = Field(..., description="model sequence")


class FullDetectionSequence(BaseModel):
    model_sequence: List[ModelType] = Field(
        default=[ModelType.OBJECT, ModelType.FACE, ModelType.ALPR],
        description="ordered model sequence",
    )
    aliases: dict = Field(..., description="label aliases for grouping")
    filters: dict = Field(..., description="Global filters for detection")
    models: Dict[ModelType, DetectionModel] = Field(..., description="Configured models")


class Singleton(type):
    """Implementation of the Singleton pattern, to be used as a metaclass."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def str_to_path(path, field: ModelField) -> Path:
    msg = f"{field.name} must be a path or a string of a path"
    if path is None:
        raise ValueError(f"{msg}, not 'None'")
    elif not isinstance(path, (Path, str)):
        raise ValueError(f"{msg}, not {type(path)}")
    elif isinstance(path, str):
        logger.debug(f"Attempting to convert {field.name} string '{path}' to Path object")
        try:
            path = Path(path)
        except Exception as e:
            raise ValueError(f"{msg}, {path} is not a valid path") from e
        else:
            if not path.exists():
                raise ValueError(f"{msg}, {path} does not exist")
            elif not path.is_dir():
                raise ValueError(f"{msg}, {path} is not a directory")
    # path is a Path() object
    return path

class AvailableModel(BaseModel):
    name: str = Field(..., description="model name")
    file: Path = Field(..., description="model file path")
    labels: Path = Field(..., description="model labels file path")
    config: Path = Field(default=None, description="model config file path")



class Settings(BaseSettings):
    data_dir: str = Field(default=None, description="Base data path for the running server")
    model_dir: str = Field(default=None, description="Machine Learning models directory")
    log_dir: str = Field(default=None, description="Logs directory")
    db_dir: str = Field(default=None, description="User database directory")
    host: str = Field(default=None, description="Interface IP to listen on")
    port: int = Field(default=None, description="Port to listen on")
    jwt_secret: str = Field(default=None, description="JWT signing key")
    reload: bool = Field(default=False, description="Uvicorn reload")

