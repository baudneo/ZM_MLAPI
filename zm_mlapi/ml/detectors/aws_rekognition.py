# AWS Rekognition support for ZM object detection
# Author: Michael Ludvig
import time
from logging import getLogger

import numpy as np

from zm_mlapi.imports import RekognitionModelConfig, cv2


LP: str = "AWS:Rekognition:"
logger = getLogger("zm_mlapi")

try:
    import boto3
except ImportError:
    logger.warning(f"{LP} the 'boto3' package is needed for aws rekognition support! Please install boto3")


class AwsRekognition:
    def __init__(self, model_config: RekognitionModelConfig):
        self.config = model_config
        self.options = model_config.detection_options
        self.name: str = self.config.name
        self.processor = self.config.processor

        boto3_kwargs = {
            "region_name": self.config.region_name,
            "aws_access_key_id": self.config.aws_access_key_id,
            "aws_secret_access_key": self.config.aws_secret_access_key,
        }
        self.model = boto3.client("rekognition", **boto3_kwargs)
        logger.debug(f"{LP} initialized")

    def detect(self, input_image: np.ndarray):
        b_boxes, labels, confs = [], [], []
        h, w = input_image.shape[:2]
        t = time.perf_counter()
        logger.debug(
            f"{LP}detect: '{self.name}' - input image {w}*{h}"
        )
        _conf = self.options.confidence
        _conf = _conf * 100
        response = self.model.detect_labels(Image={"Bytes": input_image.tobytes()}, MinConfidence=_conf)
        logger.debug(f"perf:{LP} took {time.perf_counter() - t} -> detection response -> {response}")

        # Parse the returned labels
        for item in response["Labels"]:
            if "Instances" not in item:
                continue
            for instance in item["Instances"]:
                if "BoundingBox" not in instance or "Confidence" not in instance:
                    continue
                conf = instance["Confidence"] / 100
                if conf < self.options.confidence:
                    continue
                label = item["Name"].lower()
                box = instance["BoundingBox"]
                bbox = (
                    round(w * box["Left"]),
                    round(h * box["Top"]),
                    round(w * (box["Left"] + box["Width"])),
                    round(h * (box["Top"] + box["Height"])),
                )
                b_boxes.append(bbox)
                labels.append(label)
                confs.append(conf)

        return {
            "detections": True if labels else False,
            "type": self.config.model_type,
            "processor": self.processor,
            "model_name": self.name,
            "label": labels,
            "confidence": confs,
            "bounding_box": b_boxes,
        }
