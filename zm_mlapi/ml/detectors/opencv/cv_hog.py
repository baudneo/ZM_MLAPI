from logging import getLogger

from zm_mlapi.imports import CV2HOGModelConfig
from zm_mlapi.imports import cv2
from zm_mlapi.ml.detectors.opencv.cv_base import CV2Base

LP: str = "OpenCV:HOG:"
logger = getLogger("zm_mlapi")


class CV2HOG(CV2Base):
    def __init__(self, model_config: CV2HOGModelConfig):
        self.config = model_config
        self.options = self.config.detection_options
        self.model = cv2.HOGDescriptor()
        self.model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.winStride = self.config.stride
        self.padding = self.config.padding
        self.scale = self.config.scale
        self.meanShift = self.config.mean_shift
        logger.debug(f"{LP} initialized HOG")

    def detect(self, input_image):
        r, w = self.model.detectMultiScale(
            input_image,
            winStride=self.winStride,
            padding=self.padding,
            scale=self.scale,
            useMeanshiftGrouping=self.meanShift,
        )
        labels = []
        confs = []
        b_boxes = []

        for i in r:
            labels.append("person")
            confs.append(1.0)
            i = i.tolist()
            (x1, y1, x2, y2) = (round(i[0]), round(i[1]), round(i[0] + i[2]), round(i[1] + i[3]))
            b_boxes.append((x1, y1, x2, y2))

        return {
            "success": True if labels else False,
            "type": self.config.model_type,
            "processor": self.processor,
            "model_name": self.name,
            "label": labels,
            "confidence": confs,
            "bounding_box": b_boxes,
        }
