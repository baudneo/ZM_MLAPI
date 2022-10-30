import json
import os
import subprocess
import time
import uuid
from glob import glob
from pathlib import Path
from typing import Optional, Union, Dict
from logging import getLogger


import cv2
import numpy as np
import requests
from requests import Response

from zm_mlapi.imports import ModelProcessor, OpenALPRLocalModelOptions, ALPRModelConfig, ALPRAPIType, ALPRService, OpenALPRCloudModelOptions, PlateRecognizerModelOptions
from zm_mlapi.utils import resize_cv2_image

logger = getLogger("zm_mlapi")


class AlprBase:
    def __init__(self, model_config: ALPRModelConfig):
        if not model_config:
            raise ValueError(f"{self.lp} no config passed!")
        # Model init params
        self.config: ALPRModelConfig = model_config
        self.options: Union[PlateRecognizerModelOptions, OpenALPRLocalModelOptions, OpenALPRCloudModelOptions] = self.config.detection_options
        self.processor: ModelProcessor = self.config.processor
        self.name = self.config.name
        self.api_key = self.config.api_key
        import tempfile
        self.tempdir = tempfile.gettempdir()
        self.url = self.config.api_url
        self.filename = None
        self.remove_temp = None
        if not self.config.api_key and self.config.api_type == ALPRAPIType.CLOUD:
            logger.debug(f"{self.lp} CLOUD API key not specified and you are not using the "
                         f"command line ALPR, did you forget?")
        # get rid of left over alpr temp files
        files = glob(f"{self.tempdir}/*-alpr.png")
        if files:
            logger.debug(f"{self.lp} removing {len(files)} old alpr temp files from '{self.tempdir}'")
            for file in files:
                os.remove(file)
        logger.debug(f"{self.lp} '{self.name}' configuration: {self.config}")

    def set_api_key(self, key):
        self.api_key = key
        logger.debug(f"{self.lp} API key changed")

    def _prepare_image(self, alpr_object: Union[np.ndarray, str, Path]):
        if isinstance(alpr_object, np.ndarray):
            logger.debug(f"{self.lp} the supplied object is a blob, creating temp file on disk")
            if self.options.max_size:
                logger.debug(f"{self.lp} resizing image blob using max_size={self.options.max_size}")
                alpr_object = resize_cv2_image(alpr_object, self.options.max_size)
            # use png so there is no loss
            self.filename = f"{self.tempdir}/ALPR-{uuid.uuid4()}.png"
            cv2.imwrite(self.filename, alpr_object)
            self.remove_temp = True
        elif isinstance(alpr_object, str):
            logger.debug(f"{self.lp} the supplied object is a STRING assuming absolute file path -> '{alpr_object}'")
            file_ = Path(alpr_object)
            if not file_.is_file():
                raise FileNotFoundError(f"{self.lp} the supplied file path does not exist -> '{file_}'")
            self.filename = alpr_object
            self.remove_temp = False
        elif isinstance(alpr_object, Path):
            logger.debug(f"{self.lp} the supplied object is a Path object -> '{alpr_object}'")
            if not alpr_object.is_file():
                raise FileNotFoundError(f"{self.lp} the supplied file path does not exist -> '{alpr_object}'")
            self.filename = alpr_object.as_posix()
            self.remove_temp = False


class PlateRecognizer(AlprBase):
    # plate rec: API response JSON={'processing_time': 84.586, 'results': [{'box': {'xmin': 370, 'ymin': 171, 'xmax': 726, 'ymax': 310}, 'plate': 'cft4539'
    #   , 'region': {'code': 'ca-ab', 'score': 0.607}, 'score': 0.901, 'candidates': [{'score': 0.901, 'plate': 'cft4539'}], 'dscore': 0.76, 'vehicle': {'score': 0.244, 'type': 'Sedan', 'box': {'xmin': 49, 'ymin': 75,
    #    'xmax': 770, 'ymax': 418}}}], 'filename': '0517_HpKIJ_94bReShZAkFqUXRs-alpr.jpg', 'version': 1, 'camera_id': None, 'timestamp': '2021-09-12T05:17:16.788039Z'}]
    def __init__(self, model_config: ALPRModelConfig):
        """Wrapper class for platerecognizer.com API"""
        self.lp = f"Plate Recognizer:"
        super().__init__(model_config)
        if not self.config.api_url:
            self.url = "https://api.platerecognizer.com/v1"
        self.options: PlateRecognizerModelOptions = self.config.detection_options
        logger.debug(f"{self.lp} initialized with url: {self.url}")

    def stats(self):
        """Returns API statistics

        Returns:
            HTTP Response: HTTP response of statistics API
        """
        if self.config.api_type == ALPRAPIType.LOCAL:
            logger.debug(f"{self.lp} local SDK does not provide stats")
            return {}
        try:
            if self.api_key:
                headers = {"Authorization": f"Token {self.api_key}"}
            else:
                headers = {}
            response = requests.get(f"{self.url}/statistics/", headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            response = {"error": str(e)}
        else:
            response = response.json()
        return response

    def detect(self, input_image=None):
        """Detects license plate using platerecognizer

        Args:
            input_image (image): image buffer

        Returns:
            boxes, labels, confidences: 3 objects, containing bounding boxes, labels and confidences
        """
        inf_object = input_image
        bbox = []
        labels = []
        confs = []
        self._prepare_image(inf_object)
        if self.options.stats:
            logger.debug(f"{self.lp} API usage stats: {json.dumps(self.stats())}")
        with open(self.filename, "rb") as fp:
            try:
                platerec_url = self.url
                if self.config.api_type == ALPRAPIType.CLOUD:
                    platerec_url += "/plate-reader"

                platerec_payload = {}
                if self.options.regions:
                    platerec_payload["regions"] = self.options.regions
                if self.options.payload:
                    logger.debug(
                        f"{self.lp} found API payload, overriding existing payload"
                    )
                    platerec_payload = self.options.payload

                if self.options.config:
                    logger.debug(f"{self.lp} found API config, using it")
                    platerec_payload["config"] = self.options.config
                response = requests.post(
                    platerec_url,
                    timeout=15,
                    files=dict(upload=fp),
                    data=platerec_payload,
                    headers={"Authorization": f"Token {self.api_key}"},
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                c = response.content
                response = {"error": f"{self.lp} rejected the upload with: {e}.", "results": []}
                logger.error(f"{self.lp} API rejected the upload with {e} and body:{c}")
            else:
                response = response.json()
                logger.debug(f"{self.lp} API response JSON={response}")

        if self.remove_temp:
            os.remove(self.filename)
        response: Union[Dict, Response]
        plates: Dict
        if response.get("results"):
            for plates in response.get("results"):
                label = plates["plate"]
                dscore = plates["dscore"]
                score = plates["score"]

                if dscore >= self.options.min_dscore and score >= self.options.min_score:
                    x1 = round(int(plates["box"]["xmin"]))
                    y1 = round(int(plates["box"]["ymin"]))
                    x2 = round(int(plates["box"]["xmax"]))
                    y2 = round(int(plates["box"]["ymax"]))
                    labels.append(label)
                    bbox.append([x1, y1, x2, y2])
                    confs.append(score)
                else:
                    logger.debug(
                        f"{self.lp} discarding plate:{label} because its dscore:{dscore}/score:{score} are not in "
                        f"range of configured dscore:{self.options.min_dscore} score:"
                        f"{self.options.min_score}"
                    )

        return {
            "success": True if labels else False,
            "type": self.config.model_type,
            "processor": self.processor,
            "model_name": self.name,
            "label": labels,
            "confidence": confs,
            "bounding_box": bbox,
        }


class OpenAlprCloud(AlprBase):
    def __init__(self, model_config: ALPRModelConfig):
        """Wrapper class for Open ALPR Cloud service
        """
        self.lp = f"OpenAlpr:Cloud:"
        super().__init__(model_config)
        if not self.config.api_url:
            self.url = "https://api.openalpr.com/v2/recognize"

        logger.debug(f"{self.lp} initialized with url: {self.url}")

    def detect(self, input_image: np.ndarray):
        """Detection using OpenALPR

        Args:
            input_image (image): image buffer
        """
        alpr_object = input_image
        bbox = []
        labels = []
        confs = []

        self._prepare_image(alpr_object)
        with Path(self.filename).open("rb") as fp:
            try:
                params = ""
                if self.options.country:
                    params = f"{params}&country={self.options.country}"
                if self.options.state:
                    params = f"{params}&state={self.options.state}"
                if self.options.recognize_vehicle:
                    params = f"{params}&recognize_vehicle={self.options.recognize_vehicle}"

                cloud_url = f"{self.url}?secret_key={self.api_key}{params}"
                logger.debug(f"{self.lp} trying with url: {cloud_url}")
                response = requests.post(cloud_url, files={"image": fp})
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                response = {"error": f"Open ALPR rejected the upload with {e}", "results": []}
                logger.debug(f"{self.lp} rejected the upload with {e}")
            else:
                response = response.json()
                logger.debug(f"{self.lp} JSON: {response}")

        rescale = False
        if self.remove_temp:
            os.remove(self.filename)

        response: Union[Dict, Response]
        plates: Dict
        if response.get("results"):
            for plates in response.get("results"):
                label = plates["plate"]
                conf = float(plates["confidence"]) / 100
                if conf < self.options.confidence:
                    logger.debug(
                        f"{self.lp} discarding plate: {label} because detected confidence {conf} is less than "
                        f"configured min confidence: {self.options.confidence}"
                    )
                    continue

                if plates.get("vehicle"):  # won't exist if recognize_vehicle is off
                    veh = plates.get("vehicle")
                    for attribute in ["color", "make", "make_model", "year"]:
                        if veh[attribute]:
                            label = label + "," + veh[attribute][0]["name"]

                x1 = round(int(plates["coordinates"][0]["x"]))
                y1 = round(int(plates["coordinates"][0]["y"]))
                x2 = round(int(plates["coordinates"][2]["x"]))
                y2 = round(int(plates["coordinates"][2]["y"]))
                labels.append(label)
                bbox.append([x1, y1, x2, y2])
                confs.append(conf)

        return {
            "success": True if labels else False,
            "type": self.config.model_type,
            "processor": self.processor,
            "model_name": self.name,
            "label": labels,
            "confidence": confs,
            "bounding_box": bbox,
        }


class OpenAlprCmdLine(AlprBase):
    def __init__(self, model_config: ALPRModelConfig):
        """Wrapper class for OpenALPR command line utility
        """
        self.lp = f"OpenAlpr:CmdLine:"
        super().__init__(model_config)
        cmd = self.options.openalpr_binary
        self.cmd = f"{cmd} {self.options.openalpr_binary_params}"
        if self.cmd.lower().find("-j") == -1:
            logger.debug(f"{self.lp} Adding -j to force JSON output")
            self.cmd = f"{self.cmd} -j"
        logger.debug(f"{self.lp} initialized with cmd: {self.cmd}")

    def detect(self, input_image: np.ndarray):
        """Detection using OpenALPR command line

        Args:
            input_image (image): image buffer
        """
        bbox = []
        labels = []
        confs = []

        alpr_cmdline_exc_start = time.perf_counter()
        self._prepare_image(input_image)
        do_cmd = f"{self.cmd} {self.filename}"

        logger.debug(f"{self.lp} executing: '{do_cmd}'")
        response = subprocess.check_output(do_cmd, shell=True)
        # response = subprocess.check_output(do_cmd)
        # this will cause the json.loads to fail if using gpu (haven't tested openCL)
        from re import sub
        response = response.decode("utf-8")
        p = "--\(\!\)Loaded CUDA classifier\n"
        response = sub(p, "", response)
        logger.debug(f"perf:{self.lp} took {time.perf_counter() - alpr_cmdline_exc_start}")
        logger.debug(f"{self.lp} JSON response -> {response}")
        try:
            response = json.loads(response)
        except ValueError as e:
            logger.error(f"{self.lp} Error parsing JSON response -> {e}")
            response = {}

        rescale = False
        if self.remove_temp:  # move to BytesIO buffer?
            os.remove(self.filename)
        results = response.get("results")
        # all_matches = response.get("candidates")

        """{"version":2,"data_type":"alpr_results","epoch_time":1631393388251,"img_width":800,"img_height":450,"processing_time_ms":501.42929
          1,"regions_of_interest":[{"x":0,"y":0,"width":800,"height":450}],

          "results":[{"plate":"CFT4539","confidence":90.140419,"matches_template":0,"plate_index":0,"region":"","region_confidence":0,"processing_time_ms"
          :93.152191,"requested_topn":10,"coordinates":[{"x":412,"y":175},{"x":694,"y":180},{"x":694,"y":299},{"x":412,"y":295}],

          "candidates":[{"plate":"CFT4539","confidence":90.140419,"matches_template":0},{"plate":"CF
          T4S39","confidence":82.398186,"matches_template":0},{"plate":"CFT439","confidence":79.333336,"matches_template":0},{"plate":"GFT4539","confidence":80.629532,"matches_template":0},{"plate":"CT4539","confidence"
          :80.943665,"matches_template":0},{"plate":"CPT4539","confidence":80.256454,"matches_template":0},{"plate":"CFT459","confidence":77.853737,"matches_template":0},{"plate":"CFT4B39","confidence":77.567482,"matche
          s_template":0},{"plate":"CF4539","confidence":75.923660,"matches_template":0}]}]}"""
        if results:
            for plates in results:
                label = plates["plate"]
                conf = float(plates["confidence"]) / 100
                if conf < self.options.confidence:
                    logger.debug(
                        f"{self.lp} discarding plate: {label} ({conf}) is less than the configured min confidence "
                        f"-> '{self.options.confidence}'"
                    )
                    continue
                # todo all_matches = 'candidates' and other data points from a successful detection via alpr_cmdline

                x1 = round(int(plates["coordinates"][0]["x"]))
                y1 = round(int(plates["coordinates"][0]["y"]))
                x2 = round(int(plates["coordinates"][2]["x"]))
                y2 = round(int(plates["coordinates"][2]["y"]))
                labels.append(label)
                bbox.append([x1, y1, x2, y2])
                confs.append(conf)

        return {
            "success": True if labels else False,
            "type": self.config.model_type,
            "processor": self.processor,
            "model_name": self.name,
            "label": labels,
            "confidence": confs,
            "bounding_box": bbox,
        }
