import time
from pathlib import Path
from typing import Optional, List
from logging import getLogger

# Constants
from zm_mlapi.utils import str2bool
from zm_mlapi.schemas import ModelType, ModelFrameWork, ModelProcessor, ModelSequence, FullDetectionSequence, DetectionModel

LP: str = "yolo:"
logger = getLogger()

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


class Yolo:
    def __init__(self, options: ModelSequence):
        # Log Prefix and the global config
        self.lp = LP
        if not options:
            raise ValueError(f"no options passed!")
        # Model init params
        self.options: ModelSequence = options

        logger.debug(4, f"{LP} initialization params: {self.options}")

        self.original_image: Optional[np.ndarray] = None
        self.version: int = 3
        self.model_name = "Yolo"
        self.net: Optional[cv2.dnn] = None
        self.model: Optional[cv2.dnn_DetectionModel] = None

    def get_sequence_name(self) -> str:
        return self.options.name

    def get_options(self):
        return self.options

    def get_classes(self):
        return self.options.labels

    def populate_class_labels(self):
        if self.options.labels_file:
            with self.options.labels_file.open("r") as f:
                self.options.labels = [line.strip() for line in f.readlines()]

    def load_model(self):
        logger.debug(f"{LP} loading model data from sequence '{self.options.name}'")
        t = time.perf_counter()
        try:
            self.net = cv2.dnn.readNet(self.options.model_file, self.options.config_file)
        except Exception as model_load_exc:
            logger.error(
                f"{LP} Error while loading model file and/or config! (May need to re-download the model/cfg file) => {model_load_exc}"
            )
            raise ValueError(repr(model_load_exc))
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(
            scale=1 / 255, size=(self.options.width, self.options.height), swapRB=True
        )
        logger.debug(
            f"perf:{LP} '{self.options.name}' initialization -> loading "
            f"'{self.options.model_file}' took: {time.perf_counter() - t:.5f}ms"
        )
        if self.options.show_name:
            self.options.name = self.options.show_name
        else:
            self.options.name = self.options.model_file.stem
        logger.debug(f"{LP} model name: {self.options.name}")

        if self.options.processor == ModelProcessor.GPU:
            cv_ver = cv2_version()
            # 4.5.4 and above (@pliablepixels tracked down the exact version change
            # see https://github.com/ZoneMinder/mlapi/issues/44)
            if cv_ver < 420:
                logger.error(
                    f"You are using OpenCV version {cv2.__version__} which does not support CUDA for DNNs. A minimum"
                    f" of 4.2 is required. See https://medium.com/@baudneo/install-zoneminder-1-36-x-6dfab7d7afe7"
                    f" on how to compile and install openCV 4.5.4 with CUDA"
                )
                self.options.processor = ModelProcessor.CPU
            else:  # Passed opencv version check, using GPU
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                if str2bool(self.options.fp_16):
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
        self.populate_class_labels()

    def parse_detection_outputs(self, input_image, output_data, conf_threshold, nms_threshold):
        class_ids, confidences, boxes = [], [], []
        rows = output_data.shape[0]
        image_width, image_height = input_image.shape[:2]
        # compute factors for scaling bounding boxes to the original image
        x_factor = image_width / self.options.width
        y_factor = image_height / self.options.height
        from inspect import stack as ins_stack
        func_name = ins_stack()[0][3]
        logger.debug(f"DBG -=> INSIDE {func_name}! {range(rows) = } - {x_factor = } - {y_factor = }")
        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= conf_threshold:
                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if classes_scores[class_id] >= conf_threshold:
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    x, y, w, h = (
                        row[0].item(),
                        row[1].item(),
                        row[2].item(),
                        row[3].item(),
                    )
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = [left, top, width, height]
                    boxes.append(box)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        result_class_ids, result_confidences, result_boxes = [], [], []
        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        return result_class_ids, result_confidences, result_boxes

    def square_image(self, frame):
        """Zero pad the matrix to make the image squared"""
        logger.debug(f"DBG -=> INSIDE square_image(), frame shape before padding: {frame.shape}")
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        logger.debug(f"DBG -=> INSIDE square_image(), frame shape after padding: {result.shape}")
        return result

    def detect(self, input_image: Optional[np.ndarray] = None, retry: bool = False):
        if input_image is None:
            logger.error(f"{LP} no image passed!?!")
            raise ValueError("NO_IMAGE")
        blob, outs = None, None
        class_ids, confidences, boxes = [], [], []
        bbox, label, conf = [], [], []
        h, w = input_image.shape[:2]
        nms_threshold, conf_threshold = 0.4, 0.2
        filters = self.options.filters
        logger.debug(f"DBG -=> INSIDE detect(), MIN CONF: {filters.get('object_min_confidence')}")
        min_conf = filters.get('object_min_confidence')
        configured_conf_thresh = float(min_conf)
        if configured_conf_thresh < conf_threshold:
            logger.warning(f"{LP} confidence threshold too low (less than 20%), setting to {conf_threshold}")
            conf_threshold = configured_conf_thresh
        else:
            conf_threshold = configured_conf_thresh
        logger.debug(f"{LP} confidence threshold: {conf_threshold}")
        try:
            if not self.net or (self.net and retry):
                # model has not been loaded or this is a retry detection, so we want to rebuild
                # the model with changed options.
                logger.debug(f"DEBUGGING - self.net? {'yes' if self.net else 'no'} -- {retry = }")
                self.load_model()
            logger.debug(
                f"{LP} '{self.options.name}' ({self.options.processor}) - input image {w}*{h} - model input set as: "
                f"{self.options.width}*{self.options.height}"
            )
            t = time.perf_counter()
            if self.options.square_image:
                logger.debug(
                    f"DBG => creating a squared zero-padded image for YOLO input"
                )
                input_image = self.square_image(input_image)
                blob = cv2.dnn.blobFromImage(
                    input_image,
                    scalefactor=1 / 255,
                    size=(self.options.width, self.options.height),
                    swapRB=True,
                )
                self.net.setInput(blob)
                outs = self.net.forward()
                class_ids, confidences, boxes = self.parse_detection_outputs(
                    input_image, outs[0], conf_threshold, nms_threshold
                )
                for (classid, confidence, box) in zip(class_ids, confidences, boxes):
                    # label, bbox, conf
                    x = box[0]
                    y = box[1]
                    w_ = box[2]
                    h_ = box[3]
                    bbox.append(
                        [
                            x,
                            y,
                            x + w_,
                            y + h_,
                        ]
                    )
                    label.append(self.options.labels[classid])
                    conf.append(confidence)

            else:

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
                        label.append(self.options.labels[class_id])
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
