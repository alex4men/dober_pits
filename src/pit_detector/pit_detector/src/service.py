from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import shutil
import tempfile
import os
from typing_extensions import Annotated
from pipeline import Pipeline


app = FastAPI()
pipe = Pipeline()


@app.post("/infer_video/")
async def infer_video(data: UploadFile, prompt: Annotated[str, Form()]):
    temp_directory = tempfile.mkdtemp()
    os.mkdir(os.path.join(temp_directory, 'out'))
    vin_path = os.path.join(temp_directory, data.filename)
    vout_path = os.path.join(temp_directory, 'out', data.filename)
    json_path = os.path.join(temp_directory, 'out', data.filename + '.json')
    print(data.file)
    with open(vin_path, 'wb') as f:
        f.write(data.file.read())
    obj_dict = pipe.infer_video(prompt, vin_path, vout_path, json_path)

    shutil.make_archive(os.path.join(temp_directory, "arch"), 'zip', os.path.join(temp_directory, 'out'))
    
    return FileResponse(os.path.join(temp_directory, "arch.zip"))
