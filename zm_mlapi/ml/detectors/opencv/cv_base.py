from logging import getLogger
from typing import Optional, Union

from zm_mlapi.imports import ModelProcessor, BaseModelOptions, BaseModelConfig, CV2YOLOModelConfig, CV2YOLOModelOptions, CV2TFModelConfig, CV2TFModelOptions, CV2HOGModelConfig

import cv2
import numpy as np

from zm_mlapi.ml.file_locks import FileLock

logger = getLogger("zm_mlapi")
LP: str = "OpenCV DNN:"


class CV2Base(FileLock):
    def __init__(self, model_config: Union[CV2YOLOModelConfig, CV2TFModelConfig, CV2HOGModelConfig]):
        super().__init__()
        self.lock_name = ""
        self.lock_maximum = 0
        self.lock_dir = ""
        self.lock_timeout = 0.0
        if not model_config:
            raise ValueError(f"{LP} no config passed!")
        # Model init params
        self.config = model_config
        self.options: Union[CV2YOLOModelOptions, CV2TFModelOptions, BaseModelOptions] = self.config.detection_options
        self.processor: ModelProcessor = self.config.processor
        self.name = self.config.name
        self.net: Optional[cv2.dnn] = None
        self.model = None
        self.id = self.config.id


    @staticmethod
    def square_image(frame: np.ndarray):
        """Zero pad the matrix to make the image squared"""
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        logger.debug(
            f"{LP}squaring image-> before padding: {frame.shape} - after padding: {result.shape}"
        )
        return result

    @staticmethod
    def cv2_version() -> int:
        _maj, _min, _patch = "", "", ""
        x = cv2.__version__.split(".")
        x_len = len(x)
        if x_len <= 2:
            _maj, _min = x
            _patch = "0"
        elif x_len == 3:
            _maj, _min, _patch = x
            _patch = _patch.replace("-dev", "") or "0"
        else:
            logger.error(f'come and fix me again, cv2.__version__.split(".")={x}')
        return int(_maj + _min + _patch)

    def cv2_processor_check(self):
        if self.config.processor == ModelProcessor.GPU:
            logger.debug(
                f"{LP} '{self.name}' GPU configured as the processor, running checks..."
            )
            cv_ver = self.cv2_version()
            if cv_ver < 420:
                logger.error(
                    f"{LP} '{self.name}' You are using OpenCV version {cv2.__version__} which does not support CUDA for DNNs. A minimum"
                    f" of 4.2 is required. See https://medium.com/@baudneo/install-zoneminder-1-36-x-6dfab7d7afe7"
                    f" on how to compile and install OpenCV with CUDA"
                )
                self.processor = self.config.processor = ModelProcessor.CPU
            else:  # Passed opencv version check, using GPU
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                if self.config.cv2_cuda_fp_16:
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                    logger.debug(
                        f"{LP} '{self.name}' half precision floating point (FP16) cuDNN target enabled (turn this off if it"
                        f" makes detections slower or you see 'NaN' errors!)"
                    )
                else:
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        elif self.config.processor == ModelProcessor.CPU:
            logger.debug(f"{LP} '{self.name}' CPU configured as the processor")
        self.create_lock()
