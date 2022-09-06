from os import getuid
from pathlib import Path
import time
from logging import getLogger

import cv2
import numpy as np
from PIL import Image




from zm_mlapi.imports import (
    ModelProcessor,
    ModelOptions,
    MLModelConfig,
)

logger = getLogger("zm_mlapi")
LP: str = "Coral:"

LP: str

# global placeholders for TPU lib imports
common = None
detect = None
make_interpreter = None


class TPUException(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        super().__init__(self.message)

    def __str__(self):
        if self.message:
            return self.message
        else:
            return "TPUException has been raised"


class TpuDetector:
    def __init__(self, model_config: MLModelConfig, model_options: ModelOptions):
        if not model_config:
            raise ValueError(f"{LP} no config passed!")
        # Model init params
        self.config: MLModelConfig = model_config
        self.options: ModelOptions = model_options
        self.processor: ModelProcessor = self.options.processor
        self.model_name = self.config.name
        try:
            global common, detect, make_interpreter
            from pycoral.adapters import common, detect
            from pycoral.utils.edgetpu import make_interpreter
        except ImportError:
            logger.warning(
                f"{LP} pycoral libs not installed, this is ok if you do not plan to use "
                f"TPU as detection processor. If you intend to use a TPU please install the TPU libs "
                f"and pycoral!"
            )
        else:
            logger.debug(f"{LP} the pycoral library has been successfully imported!")

            logger.debug(f"{LP} initializing edge TPU with params: {self.options}")

            self.model = None

    def load_model(self):
        logger.debug(f"{LP} loading model into processor memory: {self.model_name} ({self.config.id})")
        t = time.perf_counter()
        try:
            self.model = make_interpreter(self.config.input.as_posix())
        except Exception as ex:
            ex = repr(ex)
            words = ex.split(" ")
            for word in words:
                if word.startswith("libedgetpu"):
                    logger.info(
                        f"{LP} TPU error detected (replace cable with a short high quality one, dont allow "
                        f"TPU/cable to move around). Reset the USB port or reboot!"
                    )
                    raise TPUException("TPU NO COMM")
        else:
            self.model.allocate_tensors()
            logger.debug(f"perf:{LP} loading took: {time.perf_counter() - t:.5f}s")


    def detect(self, input_image: np.ndarray):
        orig_h, orig_w = h, w = input_image.shape[:2]
        downscaled = False
        upsize_xfactor = None
        upsize_yfactor = None
        model_resize = False
        if self.options.height and self.options.width:
            model_resize = True

        if model_resize:
            downscaled = True
            logger.debug(2, f"{LP} model dimensions requested -> " f"{self.model_width}*{self.model_height}")
            input_image = cv2.resize(
                input_image, (int(self.model_width), int(self.model_height)), interpolation=cv2.INTER_AREA
            )
            newHeight, newWidth = input_image.shape[:2]
            # get scaling so we can make correct bounding boxes
            upsize_xfactor = w / newWidth
            upsize_yfactor = h / newHeight

        h, w = input_image.shape[:2]
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = Image.fromarray(input_image)
        try:
            if not self.model:
                self.load_model()

            logger.debug(
                f"{LP} '{self.sequence_name}' input image (w*h): {orig_w}*{orig_h} resized by model_width/height "
                f"to {self.model_width}*{self.model_height}"
            )
            t = time.perf_counter()
            _, scale = common.set_resized_input(
                self.model, input_image.size, lambda size: input_image.resize(size, Image.ANTIALIAS)
            )
            self.model.invoke()
        except Exception as ex:
            raise ex
        else:
            objs = detect.get_objects(self.model, self.options.confidence, scale)
            logger.debug(f"perf:{LP} '{self.model_name}' detection took: {time.perf_counter() - t:.5f}s")

        bbox, labels, conf = [], [], []

        for obj in objs:
            # box = obj.bbox.flatten().astype("int")
            bbox.append(
                [
                    int(round(obj.bbox.xmin)),
                    int(round(obj.bbox.ymin)),
                    int(round(obj.bbox.xmax)),
                    int(round(obj.bbox.ymax)),
                ]
            )

            labels.append(self.config.labels[obj.id])
            conf.append(float(obj.score))
        if downscaled and labels:
            logger.debug(
                f"{LP} The image was resized before processing by the 'model width/height', scaling "
                f"bounding boxes in image back up by factors of -> x={upsize_xfactor:.4} "
                f"y={upsize_yfactor:.4}",
            )
            for box in bbox:
                box[0] = round(box[0] * upsize_xfactor)
                box[1] = round(box[1] * upsize_yfactor)
                box[2] = round(box[2] * upsize_xfactor)
                box[3] = round(box[3] * upsize_yfactor)

        return {
            "detections": True if labels else False,
            "type": self.config.model_type,
            "processor": self.options.processor,
            "model_name": self.model_name,
            "label": labels,
            "confidence": conf,
            "bounding_box": bbox,
        }
