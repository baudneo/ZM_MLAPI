import json
import logging
import tempfile
import uuid
from enum import Enum
from pathlib import Path
from typing import Union, List, Dict, Optional, IO, Any

import numpy as np
from pydantic import BaseModel, Field, validator, BaseSettings
from pydantic.fields import ModelField
from zm_mlapi.ml.coco17_cv2 import COCO17

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


def check_labels_file(v, values, field: ModelField):
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
    YOLO = "yolo"
    CV_YOLO = YOLO
    CORAL = "coral"
    PYCORAL = CORAL
    EDGETPU = CORAL
    EDGE_TPU = CORAL
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    DEEPFACE = "deepface"
    ALPR = "alpr"
    FACE_RECOGNITION = "face_recognition"
    DEFAULT = CV_YOLO


class ModelProcessor(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    DEFAULT = CPU


class FaceRecognitionLibModelTypes(str, Enum):
    CNN = "cnn"
    HOG = "hog"
    DEFAULT = CNN


class ALPRAPIType(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    DEFAULT = LOCAL


class ALPRService(str, Enum):
    OPENALPR = "openalpr"
    PLATE_RECOGNIZER = "plate_recognizer"
    SCOUT = OPENALPR
    DEFAULT = OPENALPR


class DetectionResult(BaseModel):
    detections: bool = False
    type: ModelType = None
    processor: ModelProcessor = None
    model_name: str = None

    label: List[str] = None
    confidence: List[Union[float, int]] = None
    bounding_box: List[List[Union[float, int]]] = None


class BaseModelOptions(BaseModel):
    confidence: Optional[float] = Field(
        0.2, ge=0.0, le=1.0, descritpiton="Confidence Threshold"
    )


class CV2YOLOModelOptions(BaseModelOptions):
    nms: Optional[float] = Field(
        0.4, ge=0.0, le=1.0, description="Non-Maximum Suppression Threshold"
    )


class FaceRecognitionLibModelOptions(BaseModelOptions):
    # face_recognition lib config Options
    upsample_times: int = Field(
        1,
        ge=0,
        description="How many times to upsample the image looking for faces. "
        "Higher numbers find smaller faces but take longer.",
    )
    num_jitters: int = Field(
        1,
        ge=0,
        description="How many times to re-sample the face when calculating encoding. "
        "Higher is more accurate, but slower (i.e. 100 is 100x slower)",
    )

    max_size: int = Field(
        600,
        ge=100,
        description="Maximum size (Width) of image to load into memory for "
        "face detection (image will be scaled)",
    )
    recognition_threshold: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Recognition distance threshold for face recognition",
    )


class ALPRModelOptions(BaseModelOptions):
    max_size: int = Field(
        600,
        ge=1,
        description="Maximum size (Width) of image to load into memory",
    )


class OpenALPRLocalModelOptions(ALPRModelOptions):
    openalpr_binary: str = Field("alpr", description="OpenALPR binary name")
    openalpr_binary_params: str = Field(
        "-d",
        description="OpenALPR binary parameters (-j is ALWAYS passed)",
        example="-p ca -c US",
    )


class OpenALPRCloudModelOptions(ALPRModelOptions):
    # For an explanation of params, see http://doc.openalpr.com/api/?api=cloudapi
    recognize_vehicle: bool = Field(
        True,
        description="If True, will attempt to recognize the vehicle type (ie: Ford Mustang)",
    )
    country: str = Field(
        "us",
        description="Country of license plate to recognize.",
    )
    state: str = Field(
        None,
        description="State of license plate to recognize.",
    )


class PlateRecognizerModelOptions(BaseModelOptions):
    stats: bool = Field(
        False,
        description="Return stats about the plate recognition request",
    )
    payload: Dict = Field(
        None,
        description="Override the payload sent to the Plate Recognizer API, must be a JSON serializable string",
    )
    config: Dict = Field(
        None,
        description="Override the config sent to the Plate Recognizer API, must be a JSON serializable string",
    )
    # If you want to specify regions. See http://docs.platerecognizer.com/#regions-supported
    regions: List[str] = Field(
        None,
        example=["us", "cn", "kr", "ca"],
        description="List of regions to search for plates in. If not specified, all regions are searched. See http://docs.platerecognizer.com/#regions-supported",
    )
    # minimal confidence for actually detecting a plate (Just a license plate, not the actual license plate text)
    min_dscore: float = Field(
        0.1,
        ge=0,
        le=1.0,
        description="Minimal confidence for actually detecting a plate",
    )
    # minimal confidence for the translated text (plate number)
    min_score: float = Field(
        0.5,
        ge=0,
        le=1.0,
        description="Minimal confidence for the translated text from the plate",
    )

    @validator("config", "payload")
    def check_json_serializable(cls, v):
        try:
            json.dumps(v)
        except TypeError:
            raise ValueError("Must be JSON serializable")
        return v


class DeepFaceModelOptions(BaseModelOptions):
    pass


class CV2TFModelOptions(BaseModelOptions):
    pass


class PyTorchModelOptions(BaseModelOptions):
    pass


class BaseModelConfig(BaseModel):
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4, description="Unique ID of the model"
    )
    name: str = Field(..., description="model name")
    enabled: bool = Field(True, description="model enabled")
    description: str = Field(None, description="model description")
    framework: ModelFrameWork = Field(
        ModelFrameWork.DEFAULT, description="model framework"
    )
    model_type: ModelType = Field(
        ModelType.DEFAULT, description="model type (object, face, alpr)"
    )
    processor: Optional[ModelProcessor] = Field(
        ModelProcessor.CPU, description="Processor to use for model"
    )

    detection_options: Union[
        BaseModelOptions,
        FaceRecognitionLibModelOptions,
        OpenALPRLocalModelOptions,
        OpenALPRCloudModelOptions,
        PlateRecognizerModelOptions,
        ALPRModelOptions,
    ] = Field(BaseModelOptions, description="Default Configuration for the model")

    @validator("name")
    def check_name(cls, v):
        v = str(v).strip().casefold()
        return v


class CV2YOLOModelConfig(BaseModelConfig):
    input: Path = Field(None, description="model file/dir path (Optional)")
    classes: Path = Field(default=None, description="model labels file path (Optional)")
    config: Path = Field(default=None, description="model config file path (Optional)")
    height: Optional[int] = Field(
        416, ge=1, description="Model input height (resized for model)"
    )
    width: Optional[int] = Field(
        416, ge=1, description="Model input width (resized for model)"
    )
    square: Optional[bool] = Field(
        False, description="Zero pad the image to be a square"
    )
    cv2_cuda_fp_16: Optional[bool] = Field(
        False, description="model uses Floating Point 16 Backend (EXPERIMENTAL!)"
    )

    labels: List[str] = Field(
        default=COCO17,
        description="model labels parsed into a list of strings",
        repr=False,
        exclude=True,
    )

    @validator("config", "input", "classes", pre=True, always=True)
    def str_to_path(cls, v, values, field: ModelField) -> Optional[Path]:
        # logger.debug(f"validating {field.name} - {v = } -- {type(v) = } -- {values = }")
        msg = f"{field.name} must be a path or a string of a path"
        model_name = values.get("name", "Unknown Model")
        lp = f"Model Name: {model_name} ->"

        if v is None:
            # logger.debug(f"{lp} {field.name} is None, passing as it is Optional")
            return v
        elif not isinstance(v, (Path, str)):
            raise ValueError(msg)
        elif isinstance(v, str):
            v = Path(v)
        if field.name == "config":
            if values["input"].suffix == ".weights":
                msg = f"'{field.name}' is required when 'input' is a DarkNet .weights file"
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
                from zm_mlapi.ml.coco17_cv2 import COCO17

                v = COCO17
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


class FaceRecognitionLibModelConfig(BaseModelConfig):
    """Config cant be changed after loading - Options can be changed"""

    detection_options: FaceRecognitionLibModelOptions = Field(
        default_factory=FaceRecognitionLibModelOptions,
        description="Default Configuration for the model",
    )
    model: FaceRecognitionLibModelTypes = Field(
        FaceRecognitionLibModelTypes.DEFAULT,
        description="Face Detection Model to use. 'cnn' is more accurate but slower on CPUs. "
        "'hog' is faster but less accurate",
    )
    train_max_size: int = Field(
        800,
        description="Maximum size of image to load into memory for face training, "
        "Larger will consume more memory!",
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
    detection_model: FaceRecognitionLibModelTypes = Field(
        FaceRecognitionLibModelTypes.DEFAULT,
        description="Face model to use for detection",
    )
    training_model: FaceRecognitionLibModelTypes = Field(
        FaceRecognitionLibModelTypes.DEFAULT,
        description="Face model to use for training",
    )
    known_faces_dir: Optional[Union[Path, str]] = Field(
        None, description="Path to parent directory of known faces for training"
    )


class ALPRModelConfig(BaseModelConfig):
    api_type: ALPRAPIType = Field(ALPRAPIType.LOCAL, description="ALPR Service Type")
    service: ALPRService = Field(ALPRService.DEFAULT, description="ALPR Service to use")
    api_key: str = Field(None, description="ALPR API Key (Cloud/Local)")
    api_url: str = Field(None, description="ALPR API URL (Cloud/Local)")


class HOGModelConfig(BaseModelConfig):
    stride: str = None
    padding: str = None
    scale: float = None
    mean_shift: bool = False


class DeepFaceModelConfig(BaseModelConfig):
    pass


class CV2TFModelConfig(BaseModelConfig):
    pass


class PyTorchModelConfig(BaseModelConfig):
    pass


class ModelConfigFromFile:
    file: Path
    raw: str
    parsed_raw: dict
    parsed: dict

    def __iter__(self):
        return iter(self.parsed.get("models", []))

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
    name: str = Field(default_factory=str, description="Name of the lock file")


class LockSettings(BaseModel):
    gpu: LockSetting = Field(
        default_factory=LockSetting, description="GPU Lock Settings"
    )
    cpu: LockSetting = Field(
        default_factory=LockSetting, description="CPU Lock Settings"
    )
    tpu: LockSetting = Field(
        default_factory=LockSetting, description="TPU Lock Settings"
    )
    lock_dir: Union[str, Path] = Field(
        None,
        description="Directory for lock files (Default is Systems temporary directory)",
    )

    @validator("gpu", "cpu", "tpu", pre=True, always=True)
    def set_lock_name(cls, v, field, values):
        # logger.debug(f"locks validator {v = } --- {field.name = } -- {values = }")
        if v:
            v = LockSetting(**v)
            v.name = f"zm-mlapi_{field.name}"
        return v

    def get(self, device: str) -> LockSetting:
        device = device.casefold()
        if device == "gpu":
            return self.gpu
        elif device == "cpu":
            return self.cpu
        elif device == "tpu":
            return self.tpu
        else:
            raise ValueError(f"Invalid device type: {device}")

    @validator("lock_dir", pre=True, always=True)
    def validate_lock_dir(cls, v):
        if not v:
            v = f"{tempfile.gettempdir()}/zm_mlapi/locks"
        # assert isinstance(v, str), "lock_dir must be a string"
        temp_dir = Path(v)
        if not temp_dir.exists():
            logger.debug(f"Creating lock directory: {temp_dir}")
            temp_dir.mkdir(parents=True)
        return v


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
    available_models: List[BaseModelConfig] = Field(None, description="Available models")
    disable_locks: bool = Field(False, description="Disable file locking")
    lock_settings: LockSettings = Field(
        default_factory=LockSettings, description="Lock Settings", repr=False
    )

    class Config:
        env_nested_delimiter = "__"

    def get_lock_settings(self):
        return self.lock_settings

    @validator("available_models")
    def validate_available_models(cls, v, values):
        models = values.get("models")
        # logger.info(f"Settings._validate_available_models: {models = }")
        if models:
            v = []
            for model in models:
                if model.get("enabled", True) is True:
                    # logger.debug(f"Adding model: {type(model) = } ------ {model = }")
                    _framework = model.get("framework", "yolo")
                    _options = model.get("detection_options", {})
                    _type = model.get("model_type", "object")
                    logger.debug(f"DBG<<< {_framework} FRAMEWORK RAW options are {_options}")
                    if _framework == ModelFrameWork.FACE_RECOGNITION:
                        model["model_type"] = ModelType.FACE
                        model["detection_options"] = FaceRecognitionLibModelOptions(**_options)
                        v.append(FaceRecognitionLibModelConfig(**model))
                    elif _framework == ModelFrameWork.ALPR:
                        model["model_type"] = ModelType.ALPR
                        api_type = model.get("api_type", "local")
                        api_service = model.get("service", "openalpr")
                        logger.debug(f"DEBUG>>> FrameWork: {_framework}  Service: {api_service} [{api_type}]")
                        config = ALPRModelConfig(**model)
                        if api_service == ALPRService.OPENALPR:
                            if api_type == ALPRAPIType.LOCAL:
                                config.detection_options = OpenALPRLocalModelOptions(
                                    **_options
                                )
                            elif api_type == ALPRAPIType.CLOUD:
                                config.processor = None
                                config.detection_options = OpenALPRCloudModelOptions(
                                    **_options
                                )
                        elif api_service == ALPRService.PLATE_RECOGNIZER:
                            if api_service == ALPRAPIType.CLOUD:
                                config.processor = None
                            config.detection_options = PlateRecognizerModelOptions(
                                **_options
                            )
                        logger.debug(
                            f"DEBUG>>> FINAL ALPR OPTIONS {config.detection_options = }"
                        )
                        v.append(config)
                    elif _framework == ModelFrameWork.CV_YOLO:
                        config = CV2YOLOModelConfig(**model)
                        config.detection_options = CV2YOLOModelOptions(**_options)
                        logger.debug(
                            f"DEBUG>>> FINAL YOLO OPTIONS {config.detection_options = }"
                        )
                        v.append(config)
                    else:
                        logger.debug(
                            f"DEBUG>>> this FRAMEWORK is NOT IMPLEMENTED -> {_framework}"
                        )
                        v.append(BaseModelConfig(**model))
                else:
                    logger.debug(f"Skipping disabled model: {model.get('name')}")

                # logger.info(f"Settings._validate_available_models: {model = }")
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

    id: uuid.UUID = Field(None, description="Model ID")
    _config: Union[
        BaseModelConfig,
        CV2YOLOModelConfig,
        ALPRModelConfig,
        FaceRecognitionLibModelConfig,
        DeepFaceModelConfig,
    ]
    _options: Union[
        BaseModelOptions,
        OpenALPRLocalModelOptions,
        OpenALPRCloudModelOptions,
        PlateRecognizerModelOptions,
        ALPRModelOptions,
        FaceRecognitionLibModelOptions,
        CV2YOLOModelOptions
    ]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.config})"

    def __get__(self, instance, owner):
        return self

    def __init__(
        self,
        model_config: Union[
            BaseModelConfig,
            CV2YOLOModelConfig,
            ALPRModelConfig,
            FaceRecognitionLibModelConfig,
            DeepFaceModelConfig,
        ],
    ):
        from zm_mlapi.ml.detectors.opencv.cv_yolo import CV2YOLODetector

        self.config = model_config
        self.id = self.config.id
        self.options = model_config.detection_options
        self.model: Optional[CV2YOLODetector] = None
        self._load_model()

    @property
    def options(
        self,
    ) -> Union[
        BaseModelOptions,
        OpenALPRLocalModelOptions,
        OpenALPRCloudModelOptions,
        PlateRecognizerModelOptions,
        ALPRModelOptions,
        FaceRecognitionLibModelOptions,
    ]:
        return self._options

    @options.setter
    def options(
        self,
        options: Union[
            BaseModelOptions,
            OpenALPRLocalModelOptions,
            OpenALPRCloudModelOptions,
            PlateRecognizerModelOptions,
            ALPRModelOptions,
            FaceRecognitionLibModelOptions,
        ],
    ):
        self._options = options

    def _load_model(
        self,
        config: Optional[
            Union[
                BaseModelConfig,
                CV2YOLOModelConfig,
                ALPRModelConfig,
                FaceRecognitionLibModelConfig,
                DeepFaceModelConfig,
            ]
        ] = None,
    ):
        """Load the model"""

        self.model = None
        if config:
            self.config = config
        if self.config.processor and not self.is_processor_available():
            raise RuntimeError(
                f"{self.config.processor} is not available on this system"
            )
        if self.config.framework == ModelFrameWork.YOLO:
            from zm_mlapi.ml.detectors.opencv.cv_yolo import CV2YOLODetector

            self.model = CV2YOLODetector(self.config)
        elif self.config.framework == ModelFrameWork.FACE_RECOGNITION:
            from zm_mlapi.ml.detectors.face_recognition import (
                FaceRecognitionLibDetector,
            )
            self.model = FaceRecognitionLibDetector(self.config)
        elif self.config.framework == ModelFrameWork.ALPR:
            from zm_mlapi.ml.detectors.alpr import OpenAlprCmdLine, OpenAlprCloud, PlateRecognizer
            if self.config.service == ALPRService.PLATE_RECOGNIZER:
                self.model = PlateRecognizer(self.config)
            elif self.config.service == ALPRService.OPENALPR:
                if self.config.api_type == ALPRAPIType.LOCAL:
                    self.model = OpenAlprCmdLine(self.config)
                elif self.config.api_type == ALPRAPIType.CLOUD:
                    self.model = OpenAlprCloud(self.config)
        else:
            logger.warning(
                f"CANT CREATE DETECTOR -> Framework NOT IMPLEMENTED!!! {self.config.framework}"
            )

    def is_processor_available(self) -> bool:
        """Check if the processor is available"""
        available = False
        processor = self.config.processor
        framework = self.config.framework
        if framework == ModelFrameWork.ALPR:
            available = True
        elif not processor:
            available = True
        elif processor == ModelProcessor.CPU:
            if framework == ModelFrameWork.CORAL:
                logger.error(f"{processor} is not supported for {framework}")
            else:
                available = True

        elif processor == ModelProcessor.TPU:
            if framework == ModelFrameWork.CORAL:
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
        elif processor == ModelProcessor.GPU:
            if framework == ModelFrameWork.OPENCV or framework == ModelFrameWork.YOLO:
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
            elif framework == ModelFrameWork.TENSORFLOW:
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
            elif framework == ModelFrameWork.PYTORCH:
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
            elif framework == ModelFrameWork.FACE_RECOGNITION:
                try:
                    import dlib
                except ImportError:
                    logger.warning(
                        "dlib not installed, cannot load any models that use dlib GPU processor"
                    )
                else:
                    if dlib.DLIB_USE_CUDA and dlib.cuda.get_num_devices() >= 1:
                        available = True
            elif framework == ModelFrameWork.DEEPFACE:
                logger.warning("WORKING ON DeepFace models!")
                pass
        return available

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect objects in the image"""
        assert self.model, "model not loaded"
        return self.model.detect(image)


class GlobalConfig(BaseModel):
    available_models: List[BaseModelConfig] = Field(
        default_factory=list, description="Available models, call by ID"
    )
    settings: Settings = Field(
        default=None, description="Global settings from ENVIRONMENT"
    )
    detectors: List[APIDetector] = Field(
        default_factory=list, description="Loaded detectors"
    )

    class Config:
        arbitrary_types_allowed = True

    def get_detector(self, model: BaseModelConfig) -> Optional[APIDetector]:
        """Get a detector by ID"""
        ret_: Optional[APIDetector] = None
        for detector in self.detectors:
            if detector.config.id == model.id:
                ret_ = detector
        if not ret_:
            logger.debug(
                f"Creating new detector for '{model.name}'"
            )
            ret_ = APIDetector(model)
            self.detectors.append(ret_)
        if not ret_:
            logger.error(f"Unable to create detector for {model.name}")
        return ret_
