import os
import pickle
import time
from pathlib import Path
from typing import Optional
from logging import getLogger

import numpy as np

from zm_mlapi.imports import (
    ModelProcessor,
    FaceModelOptions,
    FaceModel,
    MLModelConfig,
    ModelFrameWork,
)

import cv2
from sklearn import neighbors

logger = getLogger("zm_mlapi")

face_recognition = None
dlib = None
LP = "Face:DLib:"


# Class to handle face recognition
class DLibFaceDetector:
    def __init__(self, model_config: MLModelConfig, model_options: FaceModelOptions):
        if not model_config:
            raise ValueError(f"{LP} no config passed!")
        # Model init params
        self.config: MLModelConfig = model_config
        self.options: FaceModelOptions = model_options
        self.processor: ModelProcessor = self.options.processor
        self.model_name: str = self.config.name
        self.knn: Optional[neighbors.KNeighborsClassifier] = None
        self.downscaled: bool = False
        self.x_factor: float = 1.0
        self.y_factor: float = 1.0
        self.original_image: Optional[np.ndarray] = None

        self.face_locations: list = []
        self.face_encodings: list = []
        self.trained_faces_file: Optional[Path] = None

        import_timer: time.perf_counter
        import_end_time: time.perf_counter
        try:
            global dlib

            import dlib
        except ImportError as e:
            logger.error(f"{LP} UNABLE to import DLIB library, is it installed?")
            raise e
        else:
            logger.debug(f"{LP} successfully imported DLIB library")

        try:
            import_timer = time.perf_counter()
            global face_recognition

            import face_recognition
        except ImportError as e:
            logger.error(
                f"{LP} Could not import face_recognition library, is it installed?"
            )
            raise e
        else:
            import_end_time = time.perf_counter() - import_timer
            logger.debug(f"{LP} successfully imported face_recognition library!")
        logger.debug(f"{LP} init params: {model_options}")
        dlib: dlib
        face_recognition: face_recognition
        if self.processor == ModelProcessor.GPU:
            if dlib.DLIB_USE_CUDA and dlib.cuda.get_num_devices() >= 1:
                logger.debug(
                    f"{LP} dlib was compiled with CUDA support and there is an available GPU "
                    f"to use for processing! (Total GPUs dlib could use: {dlib.cuda.get_num_devices()})"
                )
            elif dlib.DLIB_USE_CUDA and not dlib.cuda.get_num_devices() >= 1:
                logger.error(
                    f"{LP} It appears dlib was compiled with CUDA support but there is not an available GPU "
                    f"for dlib to use! Using CPU for dlib detections..."
                )
                self.processor = ModelProcessor.CPU
            elif not dlib.DLIB_USE_CUDA:
                logger.error(
                    f"{LP} It appears dlib was not compiled with CUDA support! "
                    f"Using CPU for dlib detections..."
                )
                self.processor = ModelProcessor.CPU

        if import_end_time:
            logger.debug(
                f"perf:{LP}{self.processor}: importing Face Recognition library "
                f"took: {import_end_time:.5f}ms"
            )

        logger.debug(
            f"{LP} initializing face_recognition library with DNN model: '{self.options.face_model}'"
        )

    def load_trained_faces(self, faces_file: Optional[Path] = None):
        if faces_file and faces_file.is_file():
            self.trained_faces_file = faces_file
        else:
            self.trained_faces_file = Path(
                f"{self.options.known_faces_dir}/trained_faces.dat"
            )
        # to increase performance, read encodings from file
        if self.trained_faces_file.is_file():
            logger.debug(
                f"{LP} Trained faces file found. If you want to add new images/people, "
                f"remove: '{self.trained_faces_file}' and retrain"
            )
            try:
                with self.trained_faces_file.open("rb") as f:
                    knn = pickle.load(f)
                    self.knn = knn
            except Exception as e:
                logger.error(
                    f"{LP} error loading KNN model from '{self.trained_faces_file}' -> {e}"
                )
                raise e
        else:
            logger.error(
                f"{LP} trained faces file not found! Please train the model first!"
            )
            raise FileNotFoundError(
                f"{LP} trained faces file not found! Please train the model first!"
            )

    def get_options(self):
        return self.options

    def get_classes(self):
        if self.knn:
            return self.knn.classes_
        return []

    def detect(self, input_image: np.ndarray):
        detect_start_timer = time.perf_counter()
        h, w = input_image.shape[:2]
        max_size: int = self.options.face_max_size or w
        resized_w, resized_h = None, None

        if w > max_size:
            self.downscaled = True
            logger.debug(f"{LP} scaling image down using {max_size} as width")
            self.original_image = input_image.copy()
            from zm_mlapi.utils import resize_cv2_image

            input_image = resize_cv2_image(input_image, max_size)
            resized_h, resized_w = input_image.shape[:2]
            self.x_factor = w / resized_w
            self.y_factor = h / resized_h

        logger.debug(
            f"|---------- D-lib Face Detection"
            f"{f'(original dimensions {w}*{h}) ' if resized_h else ''}"
            f"{f'(resized input image: {resized_w}*{resized_h}) ----------|' if resized_h else f'(input image: {w}*{h}) ----------|'}"
        )
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the target image
        self.face_locations = face_recognition.face_locations(
            rgb_image,
            model=self.options.face_model,
            number_of_times_to_upsample=self.options.upsample_times,
        )
        self.face_encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=self.face_locations,
            num_jitters=self.options.num_jitters,
            model="small",  # small is default but faster, large is more accurate
        )

        logger.debug(
            f"perf:{LP}{self.processor}: computing locations and encodings took "
            f"{time.perf_counter() - detect_start_timer:.5f}ms"
        )

        # return self.face_locations, self.face_encodings



    # def recognize(self, input_image: np.ndarray):
        if not self.knn:

            raise RuntimeError("KNN model not loaded!")
        logger.debug(f"{LP} comparing detected faces to trained faces...")
        comparing_timer = time.perf_counter()
        closest_distances = self.knn.kneighbors(self.face_encodings, n_neighbors=1)
        logger.debug(
            f"{LP} closest KNN match indexes (smaller is better): {closest_distances}",
        )
        are_matches = [
            closest_distances[0][i][0] <= self.options.recognition_threshold
            for i in range(len(self.face_locations))
        ]
        prediction_labels = self.knn.predict(self.face_encodings)
        logger.debug(
            f"{LP} KNN predictions: {prediction_labels} - are_matches: {are_matches}",
        )
        logger.debug(
            f"perf:{LP}{self.processor}: matching detected faces to known faces took "
            f"{time.perf_counter() - comparing_timer:.5f}ms"
        )
        labels = []
        bboxs = []
        if self.downscaled:
            logger.debug(f"{LP} scaling image back up to original size")
            input_image = self.original_image
            scaled_face_locations = []
            for loc in self.face_locations:
                a, b, c, d = loc
                a = round(a * self.y_factor)
                b = round(b * self.x_factor)
                c = round(c * self.y_factor)
                d = round(d * self.x_factor)
                scaled_face_locations.append((a, b, c, d))
            self.face_locations = scaled_face_locations
            del scaled_face_locations

        for pred, loc, rec in zip(prediction_labels, self.face_locations, are_matches):
            label = pred if rec else self.options.unknown_face_name
            if not rec and self.options.save_unknown_faces:
                if Path(self.options.unknown_faces_dir).is_dir() and os.access(
                    self.options.unknown_faces_dir, os.W_OK
                ):
                    time_str = time.strftime("%b%d-%Hh%Mm%Ss-")
                    import uuid

                    unf = (
                        f"{self.options.unknown_faces_dir}/{time_str}{uuid.uuid4()}.jpg"
                    )
                    h, w = input_image.shape[:2]
                    leeway = int(self.options.unknown_faces_leeway_pixels)
                    x1 = max(
                        loc[3] - leeway,
                        0,
                    )
                    y1 = max(
                        loc[0] - leeway,
                        0,
                    )
                    x2 = min(
                        loc[1] + leeway,
                        w,
                    )
                    y2 = min(
                        loc[2] + leeway,
                        h,
                    )
                    crop_img = input_image[y1:y2, x1:x2]
                    # crop_img = image

                    logger.info(
                        f"{LP} saving cropped '{self.options.unknown_face_name}' face "
                        f"at [{x1},{y1},{x2},{y2} - includes leeway of "
                        f"{self.options.unknown_faces_leeway_pixels}px] to {unf}"
                    )
                    # cv2.imwrite won't throw an exception it outputs a WARN to console
                    cv2.imwrite(unf, crop_img)

            bboxs.append([loc[3], loc[0], loc[1], loc[2]])
            labels.append(f"face: {label}")

        logger.debug(
            f"perf:{LP} recognition sequence took {time.perf_counter() - comparing_timer:.5f}ms"
        )

        return {
            "detections": True if labels else False,
            "type": self.config.model_type,
            "processor": self.options.processor,
            "model_name": self.model_name,
            "label": labels,
            "confidence": [1] * len(labels),
            "bounding_box": bboxs,
        }

    def train(self, face_resize_width: Optional[int] = None):
        t = time.perf_counter()
        train_model = self.options.face_train_model
        knn_algo = "ball_tree"
        upsample_times = self.options.upsample_times
        num_jitters = self.options.num_jitters

        ext = (".jpg", ".jpeg", ".png")
        known_face_encodings = []
        known_face_names = []
        try:
            known_parent_dir = Path(self.options.known_faces_dir)
            for train_person_dir in known_parent_dir.glob("*"):
                if train_person_dir.is_dir():
                    for file in train_person_dir.glob("*"):
                        if file.suffix.lower() in ext:
                            logger.info(f"{LP} training on {file}")
                            known_face_image = cv2.imread(file.as_posix())
                            # known_face_image = face_recognition.load_image_file(file)
                            if known_face_image is None or known_face_image.size == 0:
                                logger.error(f"{LP} Error reading file, skipping")
                                continue
                            if not face_resize_width:
                                face_resize_width = self.options.face_train_max_size
                            logger.debug(f"{LP} resizing to {face_resize_width}")
                            from zm_mlapi.utils import resize_cv2_image

                            known_face_image = resize_cv2_image(
                                known_face_image, face_resize_width
                            )
                            known_face_image = cv2.cvtColor(
                                known_face_image, cv2.COLOR_BGR2RGB
                            )

                            face_locations = face_recognition.face_locations(
                                known_face_image,
                                model=train_model,
                                number_of_times_to_upsample=upsample_times,
                            )
                            if len(face_locations) != 1:
                                extra_err_msg: str = ""
                                if self.options.face_train_model == FaceModel.HOG:
                                    extra_err_msg = (
                                        "If you think you have only 1 face try using 'cnn' "
                                        "for training mode. "
                                    )

                                logger.error(
                                    f"{LP} Image has {len(face_locations)} faces, cannot use for training. "
                                    f"We need exactly 1 face. {extra_err_msg}Ignoring..."
                                )
                            else:
                                face_encoding = face_recognition.face_encodings(
                                    known_face_image,
                                    known_face_locations=face_locations,
                                    num_jitters=num_jitters,
                                )
                                if face_encoding:
                                    known_face_encodings.append(face_encoding[0])
                                    known_face_names.append(train_person_dir.name)
                                else:
                                    logger.warning(
                                        f"{LP} no face found in {file} - skipping"
                                    )
                        else:
                            logger.warning(
                                f"{LP} image is not in allowed format! skipping {file}"
                            )
                else:
                    logger.warning(
                        f"{LP} {train_person_dir} is not a directory! skipping"
                    )
        except Exception as e:
            logger.error(f"{LP} Error during recognition training: {e}")
            raise e

        if not len(known_face_names):
            logger.warning(
                f"{LP} No known faces found to train, skipping saving of face encodings to file..."
            )
        else:
            import math

            n_neighbors = int(round(math.sqrt(len(known_face_names))))
            logger.debug(f"{LP} using algo: {knn_algo} n_neighbors: {n_neighbors}")
            knn = neighbors.KNeighborsClassifier(
                n_neighbors=n_neighbors, algorithm=knn_algo, weights="distance"
            )

            logger.debug(f"{LP} training model ...")
            knn.fit(known_face_encodings, known_face_names)

            try:
                with open(self.trained_faces_file, "wb") as f:
                    pickle.dump(knn, f)
            except Exception as exc:
                logger.error(
                    f"{LP} error pickling face encodings to {self.trained_faces_file} -> {exc}"
                )
            else:
                logger.debug(f"{LP} wrote KNN encodings to '{self.trained_faces_file}'")

        logger.debug(
            f"perf:{LP} Recognition training took: {time.perf_counter() - t:.5f}ms"
        )
