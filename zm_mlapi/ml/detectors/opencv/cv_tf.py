from logging import getLogger

from zm_mlapi.ml.detectors.opencv.cvbase import CV2Base
from zm_mlapi.imports import ModelProcessor, BaseModelOptions, BaseModelConfig, cv2, np, CV2YOLOModelConfig


LP: str = "OpenCV:TF:"
logger = getLogger("zm_mlapi")


class CV2TFDetector(CV2Base):
    def __init__(self, model_config: BaseModelConfig):
        # pb = 'frozen_inference_graph.pb'
        # pbt = 'ssd_inception_v2_coco_2017_11_17.pbtxt'
        super().__init__(model_config)
        self.load_model()

    def load_model(self):
        logger.debug(f"{LP} loading model into processor memory: {self.name} ({self.config.id})")
        load_timer = time.perf_counter()
        try:
            # Allow for .weights/.cfg and .onnx YOLO architectures
            model_file: str = self.config.input.as_posix()
            config_file: Optional[str] = None
            if self.config.config and self.config.config.exists():
                config_file = self.config.config.as_posix()
            self.net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
        except Exception as model_load_exc:
            logger.error(
                f"{LP} Error while loading model file and/or config! "
                f"(May need to re-download the model/cfg file) => {model_load_exc}"
            )
            raise model_load_exc

        logger.debug(
            f"{LP} using {self.options.processor} for detection"
            f"{', set CUDA/cuDNN backend and target' if self.options.processor == ModelProcessor.GPU else ''}"
        )
        logger.debug(
            f"perf:{LP} loading completed in {time.perf_counter() - load_timer:.5f}ms"
        )

    def detect(self, input_image: Optional[np.ndarray] = None, retry: bool = False):
        if input_image is None:
            raise ValueError("NO_IMAGE")
        if self.options.square:
            input_image = self.square_image(input_image)
        rows = input_image.shape[0]
        cols = input_image.shape[1]
        classes, confs, boxes = [], [], []
        bboxs, labels, confs = [], [], []
        h, w = input_image.shape[:2]
        nms_threshold, conf_threshold = self.options.nms, self.options.confidence
        try:
            if not self.net or (self.net and retry):
                # model has not been loaded or this is a retry detection, so we want to rebuild
                # the model with changed options.
                self.cv2_processor_check()
            logger.debug(
                f"{LP} '{self.name}' ({self.options.processor}) - input image {w}*{h} - "
                f"model input set as: {self.options.width}*{self.options.height}"
            )
            detection_timer = time.perf_counter()
            blob = cv2.dnn.blobFromImage(input_image, size=(300, 300), swapRB=True, crop=False)
            self.net.setInput(blob)

            # Run object detection
            outs = self.net.forward()
        except Exception as detect_exc:
            logger.error(f"{LP} Error while detecting objects: {detect_exc}")
            raise detect_exc
        else:
            logger.debug(
                f"perf:{LP}{self.options.processor}: '{self.name}' detection "
                f"took: {time.perf_counter() - detection_timer:.5f}ms"
            )
            for detection in outs[0, 0, :, :]:
                confidence = float(detection[2])
                if confidence >= conf_threshold:
                    class_id = int(detection[1])  # prediction class index.
                    x = int(round(detection[3] * cols))
                    y = int(round([4] * rows))
                    right = int(round([5] * cols))
                    bottom = int(round([6] * rows))
                    bboxs.append(
                        [
                            x,
                            y,
                            right,
                            bottom,
                        ]
                    )
                    labels.append(self.config.labels[class_id])
                    confs.append(confidence)
        return {
            "detections": True if labels else False,
            "type": self.config.model_type,
            "processor": self.options.processor,
            "model_name": self.name,
            "label": labels,
            "confidence": confs,
            "bounding_box": bboxs,
        }
