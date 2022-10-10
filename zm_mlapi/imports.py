from logging import getLogger

import portalocker

from zm_mlapi.schemas import (
    ModelType,
    ModelFrameWork,
    ModelProcessor,
    Settings,
    GlobalConfig,
    APIDetector,
    DetectionResult,
    ALPRService,
    ALPRAPIType,
    LockSettings,
    FaceRecognitionLibModelTypes,

    BaseModelConfig,
    CV2YOLOModelConfig,
    FaceRecognitionLibModelConfig,
    ALPRModelConfig,
    DeepFaceModelConfig,
    CV2TFModelConfig,
    PyTorchModelConfig,
    HOGModelConfig,


    BaseModelOptions,
    CV2YOLOModelOptions,
    FaceRecognitionLibModelOptions,
    ALPRModelOptions,
    OpenALPRLocalModelOptions,
    OpenALPRCloudModelOptions,
    PlateRecognizerModelOptions,
    DeepFaceModelOptions,
    CV2TFModelOptions,
    PyTorchModelOptions,
)


logger = getLogger("zm_mlapi")

try:
    import cv2
except ImportError as e:
    logger.error("OpenCV is not installed, please install it")
    raise e

try:
    import numpy as np
except ImportError as e:
    logger.error("Numpy is not installed, please install it")
    raise e
