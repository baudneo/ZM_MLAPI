import time
from logging import getLogger

import cv2
import numpy as np
from PIL import Image

from zm_mlapi.imports import (
    ModelProcessor,
    FaceModelOptions,
    MLModelConfig,
)

logger = getLogger("zm_mlapi")
LP: str = "Face:Coral:"
# global placeholders for TPU lib imports
common = None
detect = None
make_interpreter = None


# Class to handle face recognition
class TpuFaceDetector:
    def __init__(self, model_config: MLModelConfig, model_options: FaceModelOptions):
        if not model_config:
            raise ValueError(f"{LP} no config passed!")
        # Model init params
        self.config: MLModelConfig = model_config
        self.options: FaceModelOptions = model_options
        self.processor: ModelProcessor = self.options.processor
        self.model_name: str = self.config.name
        try:
            global common, detect, make_interpreter
            from pycoral.adapters import common as common, detect as detect
            from pycoral.utils.edgetpu import make_interpreter as make_interpreter
        except ImportError:
            logger.warning(
                f"{LP} pycoral libs not installed, this is ok if you do not plan to use "
                f"TPU as detection processor. If you intend to use a TPU please install the TPU libs "
                f"and pycoral!"
            )
        else:
            logger.debug(f"{LP} TPU libraries have been installed and imported!")
            logger.debug(f"{LP} init params: {self.options}")
            self.knn = None
            self.model = None

    def get_options(self):
        return self.options

    def load_model(self):
        logger.debug(f"{LP} loading model into processor memory: {self.model_name} ({self.config.id})")
        load_timer = time.perf_counter()
        try:
            self.model = make_interpreter(self.config.input)
            self.model.allocate_tensors()
        except Exception as e:
            logger.error(f"{LP} failed to load model: {self.model_name}")
            raise e
        logger.debug(
            f"perf:{LP} loading completed in {time.perf_counter() - load_timer:.5f}ms"
        )

    def detect(self, input_image: np.ndarray):
        Height, Width = input_image.shape[:2]
        img = input_image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        t = time.perf_counter()
        try:
            if not self.model:
                self.load_model()
            logger.debug(f"{LP} (input image: {Width}*{Height})")
            _, scale = common.set_resized_input(self.model, img.size, lambda size: img.resize(size, Image.ANTIALIAS))
            self.model.invoke()
            objs = detect.get_objects(self.model, self.options.confidence, scale)
        except Exception as all_ex:
            raise all_ex
        logger.debug(f"perf:{LP} '{self.model_name}' detection took: {time.perf_counter() - t:.5f}ms")

        bboxs = []
        labels = []
        confs = []

        for obj in objs:
            # box = obj.bbox.flatten().astype("int")
            bboxs.append(
                [
                    int(round(obj.bbox.xmin)),
                    int(round(obj.bbox.ymin)),
                    int(round(obj.bbox.xmax)),
                    int(round(obj.bbox.ymax)),
                ]
            )

            labels.append(self.options.unknown_face_name)
            confs.append(float(obj.score))
        logger.debug(f"{LP} returning -> {labels} {bboxs} {confs}")
        return {
            "detections": True if labels else False,
            "type": self.config.model_type,
            "processor": self.options.processor,
            "model_name": self.model_name,
            "label": labels,
            "confidence": confs,
            "bounding_box": bboxs,
        }
