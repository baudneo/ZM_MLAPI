import time
from logging import getLogger
from typing import Optional

# Constants
from zm_mlapi.imports import ModelProcessor, ModelOptions, AvailableModel

LP: str = "DNN:"
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


# cv2 version check for unconnected layers fix
def cv2_version() -> int:
    # Sick and tired of OpenCV playing games....
    maj, min, patch = "", "", ""
    x = cv2.__version__.split(".")
    x_len = len(x)
    if x_len <= 2:
        maj, min = x
        patch = "0"
    elif x_len == 3:
        maj, min, patch = x
        patch = patch.replace("-dev", "") or "0"
    else:
        logger.error(f"come and fix me again, cv2.__version__.split(\".\")={x}")
    return int(maj + min + patch)


class Detector:
    def __init__(self, model_config: AvailableModel, model_options: ModelOptions):
        if not model_config:
            raise ValueError(f"{LP} no config passed!")
        # Model init params
        self.config: AvailableModel = model_config
        self.options: ModelOptions = model_options
        self.processor: ModelProcessor = self.options.processor
        self.model_name = self.config.name
        logger.info(f"{LP} initializing...")

        logger.debug(f"{LP} configuration: {self.config}")
        logger.debug(f"{LP} options: {self.options}")

        self.original_image: Optional[np.ndarray] = None
        self.net: Optional[cv2.dnn] = None
        self.model: Optional[cv2.dnn.DetectionModel] = None

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, model_options: ModelOptions):
        self._options = model_options

    @options.deleter
    def options(self):
        self._options = None

    def get_classes(self):
        return self.config.labels

    def load_model(self):
        logger.debug(f"{LP} loading data from named model: {self.model_name}")
        load_timer = time.perf_counter()
        try:
            self.net = cv2.dnn.readNet(self.config.input, self.config.config)
        except Exception as model_load_exc:
            logger.error(
                f"{LP} Error while loading model file and/or config! (May need to re-download the model/cfg file) => {model_load_exc}"
            )
            raise ValueError(repr(model_load_exc))
        self.model = cv2.dnn.DetectionModel(self.net)
        self.model.setInputParams(
            scale=1 / 255, size=(self.options.width, self.options.height), swapRB=True
        )
        if self.options.processor == ModelProcessor.GPU:
            cv_ver = cv2_version()
            if cv_ver < 420:
                logger.error(
                    f"You are using OpenCV version {cv2.__version__} which does not support CUDA for DNNs. A minimum"
                    f" of 4.2 is required. See https://medium.com/@baudneo/install-zoneminder-1-36-x-6dfab7d7afe7"
                    f" on how to compile and install OpenCV with CUDA"
                )
                self.options.processor = ModelProcessor.CPU
            else:  # Passed opencv version check, using GPU
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                if self.options.fp_16:
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                    logger.debug(
                        f"{LP} half precision floating point (FP16) cuDNN target enabled (turn this off if it"
                        f" makes yolo slower or you see NaN errors!)"
                    )
                else:
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        logger.debug(
            f"{LP} using {self.options.processor} for detection"
            f"{', set CUDA/cuDNN backend and target' if self.options.processor == ModelProcessor.GPU else ''}"
        )
        logger.debug(
            f"perf:{LP} loading of DetectionModel completed in {time.perf_counter() - load_timer:.5f}ms"
        )

    @staticmethod
    def square_image(frame):
        """Zero pad the matrix to make the image squared"""
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        logger.debug(f"{LP}squaring image-> before padding: {frame.shape} - after padding: {result.shape}")
        return result

    def detect(self, input_image: Optional[np.ndarray] = None, retry: bool = False):
        if input_image is None:
            logger.error(f"{LP} no image passed!?!")
            raise ValueError("NO_IMAGE")
        blob, outs = None, None
        class_ids, confidences, boxes = [], [], []
        bbox, label, conf = [], [], []
        h, w = input_image.shape[:2]
        nms_threshold, conf_threshold = self.options.thresholds.nms, self.options.thresholds.confidence
        logger.debug(f"{LP} confidence threshold: {conf_threshold} -- NMS threshold: {nms_threshold}")
        if self.options.square:
            input_image = self.square_image(input_image)
            h, w = input_image.shape[:2]
        try:
            if not self.net or (self.net and retry):
                # model has not been loaded or this is a retry detection, so we want to rebuild
                # the model with changed options.
                logger.debug(f"DEBUGGING - self.net? {'yes' if self.net else 'no'} -- {retry = } ---<> LOADING MODEL")
                self.load_model()
            logger.debug(
                f"{LP} '{self.model_name}' ({self.options.processor}) - input image {w}*{h} - model input set as: "
                f"{self.options.width}*{self.options.height}"
            )
            t = time.perf_counter()

            class_ids, confidences, boxes = self.model.detect(
                input_image, conf_threshold, nms_threshold
            )
            for (class_id, confidence, box) in zip(class_ids, confidences, boxes):
                confidence = float(confidence)
                if confidence >= conf_threshold:
                    x, y, _w, _h = (
                        int(round(box[0])),
                        int(round(box[1])),
                        int(round(box[2])),
                        int(round(box[3])),
                    )
                    bbox.append(
                        [
                            x,
                            y,
                            x + _w,
                            y + _h,
                        ]
                    )
                    label.append(self.config.labels[class_id])
                    conf.append(confidence)
        except Exception as all_ex:
            err_msg = repr(all_ex)
            # cv2.error: OpenCV(4.2.0) /home/<Someone>/opencv/modules/dnn/src/cuda/execution.hpp:52: error: (-217:Gpu
            # API call) invalid device function in function 'make_policy'
            logger.error(f"{LP} exception during detection -> {all_ex}")
            if (
                err_msg.find("-217:Gpu") > 0
                and err_msg.find("'make_policy'") > 0
                and self.options.processor == ModelProcessor.GPU
            ):
                logger.error(
                    f"{LP} (-217:Gpu # API call) invalid device function in function 'make_policy' - "
                    f"This happens when OpenCV is compiled with the incorrect Compute Capability "
                    f"(CUDA_ARCH_BIN). There is a high probability that you need to recompile OpenCV with "
                    f"the correct CUDA_ARCH_BIN before GPU detections will work properly!"
                )
                # set arch to cpu and retry?
                self.options.processor = ModelProcessor.CPU
                logger.info(
                    f"{LP} GPU detection failed due to probable incorrect CUDA_ARCH_BIN (Compute Capability)."
                    f" Switching to CPU and retrying detection!"
                )
                self.detect(input_image, retry=True)
            raise Exception(f"during detection -> {all_ex}")
        diff_time = time.perf_counter() - t
        logger.debug(
            2,
            f"perf:{LP}{self.options.processor}: '{self.options.name}' detection took: {diff_time}",
        )
        if label:
            logger.debug(f"{LP} {label} -- {bbox} -- {conf}")
        else:
            logger.debug(f"{LP} no detections to return!")
        _model_name = [
            f"{self.model_name}[{self.options.processor}]"
        ] * len(label)
        logger.debug(f"DBG => {_model_name = }")

        return bbox, label, conf, _model_name
        # from collections import namedtuple
        # detection = namedtuple("detection", "label bbox conf model_name")
        # detections = []
        # for i in range(len(label)):
        #     detections.append(
        #         detection(
        #             label=label[i],
        #             bbox=bbox[i],
        #             conf=conf[i],
        #             model_name=_model_name[i],
        #         )
        #     )
        # return detections
