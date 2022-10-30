import time
from logging import getLogger
from typing import Optional


from zm_mlapi.imports import ModelProcessor, BaseModelOptions, BaseModelConfig, cv2, np, CV2YOLOModelConfig
from zm_mlapi.ml.detectors.opencv.cv_base import CV2Base

LP: str = "OpenCV:YOLO:"
logger = getLogger("zm_mlapi")


class CV2YOLODetector(CV2Base):
    def __init__(self, model_config: CV2YOLOModelConfig):
        super().__init__(model_config)
        # self.is_locked: bool = False
        # Model init params not initialized in super()
        self.model: Optional[cv2.dnn.DetectionModel] = None
        # logger.debug(f"{LP} configuration: {self.config}")
        self.load_model()

    def get_classes(self):
        return self.config.labels

    def load_model(self):
        logger.debug(
            f"{LP} loading model into processor memory: {self.name} ({self.id})"
        )
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
            scale=1 / 255, size=(self.config.width, self.config.height), swapRB=True
        )
        self.cv2_processor_check()
        logger.debug(
            f"{LP} set CUDA/cuDNN backend and target"
        ) if self.processor == ModelProcessor.GPU else None

        logger.debug(
            f"perf:{LP} '{self.name}' loading completed in {time.perf_counter() - load_timer:.5f}ms"
        )

    def detect(self, input_image: np.ndarray):
        if input_image is None:
            raise ValueError(f"{LP} no image passed!")
        if not self.net:
            self.load_model()
        _h, _w = self.config.height, self.config.width
        if self.config.square:
            input_image = self.square_image(input_image)
        h, w = input_image.shape[:2]
        # dnn.DetectionModel resizes the image and calculates scale of bounding boxes for us
        labels, confs, b_boxes = [], [], []
        nms_threshold, conf_threshold = self.options.nms, self.options.confidence

        logger.debug(
            f"{LP}detect: '{self.name}' ({self.processor}) - "
            f"input image {w}*{h} - model input {_w}*{_h}"
            f"{' [squared]' if self.config.square else ''}"
        )
        self.acquire_lock()
        try:
            detection_timer = time.perf_counter()

            l, c, b = self.model.detect(
                input_image, conf_threshold, nms_threshold
            )

            logger.debug(
                f"perf:{LP}{self.processor}: '{self.name}' detection "
                f"took: {time.perf_counter() - detection_timer:.5f}ms"
            )
            for (class_id, confidence, box) in zip(l, c, b):
                confidence = float(confidence)
                if confidence >= conf_threshold:
                    x, y, _w, _h = (
                        int(round(box[0])),
                        int(round(box[1])),
                        int(round(box[2])),
                        int(round(box[3])),
                    )
                    b_boxes.append(
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
                and self.processor == ModelProcessor.GPU
            ):
                logger.error(
                    f"{LP} (-217:Gpu # API call) invalid device function in function 'make_policy' - "
                    f"This happens when OpenCV is compiled with the incorrect Compute Capability "
                    f"(CUDA_ARCH_BIN). There is a high probability that you need to recompile OpenCV with "
                    f"the correct CUDA_ARCH_BIN before GPU detections will work properly!"
                )
            import traceback
            logger.error(f"{LP} traceback: {traceback.format_exc()}")
        finally:
            self.release_lock()
        return {
            "success": True if labels else False,
            "type": self.config.model_type,
            "processor": self.processor,
            "model_name": self.name,
            "label": labels,
            "confidence": confs,
            "bounding_box": b_boxes,
        }
