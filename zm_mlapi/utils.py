import os
from datetime import datetime, timedelta
from logging import getLogger
from typing import Union, Any, Optional

import cv2
import numpy as np
from jose import jwt
from passlib.context import CryptContext

logger = getLogger("zm_mlapi")
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # 30 minutes
REFRESH_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days
ALGORITHM = "HS256"
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")  # should be kept secret
JWT_REFRESH_SECRET_KEY = os.environ.get("JWT_REFRESH_SECRET_KEY")  # should be kept secret

password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def str2bool(v: Optional[Union[Any, bool]]) -> Union[Any, bool]:
    """Convert a string to a boolean value, if possible.

    .. note::
        - The string is converted to all lower case before evaluation.
        - Strings that will return True -> ("yes", "true", "t", "y", "1", "on", "ok", "okay", "da").
        - Strings that will return False -> ("no", "false", "f", "n", "0", "off", "nyet").
        - None is converted to False.
        - A boolean is returned as-is.
    """
    if v is not None:
        true_ret = ("yes", "true", "t", "y", "1", "on", "ok", "okay", "da", "enabled")
        false_ret = ("no", "false", "f", "n", "0", "off", "nyet", "disabled")
        if isinstance(v, bool):
            return v
        elif isinstance(v, str):
            if (normalized_v := str(v).lower().strip()) in true_ret:
                return True
            elif normalized_v in false_ret:
                pass
            else:
                return logger.warning(
                    f"str2bool: The value '{v}' (Type: {type(v)}) is not able to be parsed into a boolean operator"
                )
        else:
            return logger.warning(
                f"str2bool: The value '{v}' (Type: {type(v)}) is not able to be parsed into a boolean operator"
            )
    else:
        return None
    return False


def get_hashed_password(password: str) -> str:
    return password_context.hash(password)


def verify_password(password: str, hashed_pass: str) -> bool:
    return password_context.verify(password, hashed_pass)


def create_token(
    token_type: str, subject: Union[str, Any], expires_delta: timedelta = None
) -> str:
    if expires_delta is not None and isinstance(expires_delta, timedelta):
        expires_delta = datetime.utcnow() + expires_delta
    else:
        if token_type == "access":
            expires_delta = datetime.utcnow() + timedelta(
                minutes=ACCESS_TOKEN_EXPIRE_MINUTES
            )
        elif token_type == "refresh":
            expires_delta = datetime.utcnow() + timedelta(
                minutes=REFRESH_TOKEN_EXPIRE_MINUTES
            )

    to_encode = {"exp": expires_delta, "sub": str(subject)}
    return jwt.encode(to_encode, JWT_SECRET_KEY, ALGORITHM)


def resize_cv2_image(img: np.ndarray, resize_w: int):
    """Resize a CV2 (numpy.ndarray) image using ``resize_w``. Width is used and height will be scaled accordingly."""
    lp = "resize:img:"

    if img is not None:
        h, w = img.shape[:2]
        aspect_ratio: float = resize_w / w
        dim: tuple = (resize_w, int(h * aspect_ratio))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        logger.debug(
            f"{lp} success using resize_width={resize_w} - original dimensions: {w}*{h}"
            f" - resized dimensions: {dim[1]}*{dim[0]}",
        )
    else:
        logger.debug(f"{lp} 'resize' called but no image supplied!")
    return img
