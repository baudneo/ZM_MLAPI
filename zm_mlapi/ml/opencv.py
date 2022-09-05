import time
from logging import getLogger
from typing import Optional


from zm_mlapi.imports import ModelProcessor, ModelOptions, AvailableModel

LP: str = "OpenCV DNN:"
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


class Detector:
    def __init__(self, model_config: AvailableModel, model_options: ModelOptions):
        if not model_config:
            raise ValueError(f"{LP} no config passed!")
        # Model init params
        self.config: AvailableModel = model_config
        self.options: ModelOptions = model_options
        self.processor: ModelProcessor = self.options.processor
        self.model_name = self.config.name
        self.net: Optional[cv2.dnn] = None
        self.model: Optional[cv2.dnn.DetectionModel] = None
        logger.info(f"{LP} initializing...")

        logger.debug(f"{LP} configuration: {self.config}")
        logger.debug(f"{LP} options: {self.options}")



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
        logger.debug(f"{LP} loading model into processor memory: {self.model_name} ({self.config.id})")
        load_timer = time.perf_counter()
        try:
            # Allow for .weights/.cfg and .onnx YOLO architectures
            model_file: str = self.config.input.as_posix()
            config_file: Optional[str] = None
            if self.config.config and self.config.config.exists():
                config_file = self.config.config.as_posix()
            self.net = cv2.dnn.readNet(model_file, config_file)
        except Exception as model_load_exc:
            logger.error(
                f"{LP} Error while loading model file and/or config! "
                f"(May need to re-download the model/cfg file) => {model_load_exc}"
            )
            raise model_load_exc
        # DetectionModel allows to set params for preprocessing input image. DetectionModel creates net
        # from file with trained weights and config, sets preprocessing input, runs forward pass and return
        # result detections. For DetectionModel SSD, Faster R-CNN, YOLO topologies are supported.
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
                        f" makes detections slower or you see 'NaN' errors!)"
                    )
                else:
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        logger.debug(
            f"{LP} using {self.options.processor} for detection"
            f"{', set CUDA/cuDNN backend and target' if self.options.processor == ModelProcessor.GPU else ''}"
        )
        logger.debug(
            f"perf:{LP} loading completed in {time.perf_counter() - load_timer:.5f}ms"
        )

    @staticmethod
    def square_image(frame):
        """Zero pad the matrix to make the image squared"""
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        logger.debug(
            f"{LP}squaring image-> before padding: {frame.shape} - after padding: {result.shape}"
        )
        return result

    def detect(self, input_image: Optional[np.ndarray] = None, retry: bool = False):
        if input_image is None:
            raise ValueError("NO_IMAGE")
        if self.options.square:
            input_image = self.square_image(input_image)
        class_ids, confidences, boxes = [], [], []
        bboxs, labels, confs = [], [], []
        h, w = input_image.shape[:2]
        nms_threshold, conf_threshold = self.options.nms, self.options.confidence
        try:
            if not self.net or (self.net and retry):
                # model has not been loaded or this is a retry detection, so we want to rebuild
                # the model with changed options.
                self.load_model()
            logger.debug(
                f"{LP} '{self.model_name}' ({self.options.processor}) - input image {w}*{h} - "
                f"model input set as: {self.options.width}*{self.options.height}"
            )
            detection_timer = time.perf_counter()

            class_ids, confidences, boxes = self.model.detect(
                input_image, conf_threshold, nms_threshold
            )
            logger.debug(
                f"perf:{LP}{self.options.processor}: '{self.model_name}' detection "
                f"took: {time.perf_counter() - detection_timer:.5f}ms"
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
                    bboxs.append(
                        [
                            x,
                            y,
                            x + _w,
                            y + _h,
                        ]
                    )
                    labels.append(self.config.labels[class_id])
                    confs.append(confidence)
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

        if not labels:
            logger.debug(f"{LP} no detections to return!")
        return {
            "detections": True if labels else False,
            "type": self.config.model_type,
            "processor": self.options.processor,
            "model_name": self.model_name,
            "label": labels,
            "confidence": confs,
            "bounding_box": bboxs,
        }
