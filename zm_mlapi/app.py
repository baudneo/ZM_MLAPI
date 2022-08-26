from functools import lru_cache
from pathlib import Path
from typing import Coroutine, Union

import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from zm_mlapi.schemas import Settings, ModelType

app = FastAPI()


@lru_cache()
async def get_settings(env_file):
    return Settings(_env_file=env_file)


@app.get('/', response_class=RedirectResponse, include_in_schema=False)
async def docs():
    return RedirectResponse(url='/docs')


@app.get('/available_models')
async def available_models():
    return {"models": g.available_models}


# upload an image for inference
@app.post("/detect/{infer_type}", summary="Run detection on an image")
async def object_detection(infer_type: ModelType, image: UploadFile = File(...)):
    if infer_type == ModelType.OBJECT:
        return {"message": "object detection", "image_name": image.filename}
    elif infer_type == ModelType.FACE:
        return {"message": "face detection", "image_name": image.filename}
    elif infer_type == ModelType.ALPR:
        return {"message": "alpr detection", "image_name": image.filename}
    else:
        raise HTTPException(status_code=404, detail="Inference type not found")


class MLAPI:
    def __init__(self, env_file: str):
        # test that the env file exists
        if not Path(env_file).exists():
            raise FileNotFoundError(f"{env_file} does not exist")
        self.settings: Union[Settings, Coroutine] = get_settings(env_file)
        self._run()

    def _run(self):
        uvicorn.run(app, host=self.settings.host, port=self.settings.port, reload=self.settings.reload)
