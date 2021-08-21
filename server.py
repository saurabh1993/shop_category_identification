from fastapi import FastAPI

from fastapi import FastAPI, File, UploadFile
import json
import shutil
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse
import cv2
import io


from inference_google import autoMLModel
from inference_local import localModel
from image_correction import image_rectifier

autoModel =autoMLModel()
localModel= localModel()
rec_model = image_rectifier()

app = FastAPI()




@app.post("/shop_category/")
async def categorize_shop(file: UploadFile = File(...)):
    global autoModel
    global localModel
    path="temp.jpeg"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    out={}
    autoResult=autoModel.run(path)
    out["autoML"]=autoResult
    localResult=localModel.run(path)
    out["Mobilenet"]=localResult
    
    json_compatible_item_data = jsonable_encoder(out)
    print("final",json_compatible_item_data)
    return JSONResponse(content=json_compatible_item_data)
   
    
@app.post("/rectifyImage/")
async def rectify_image(file: UploadFile = File(...)):
    global rec_model
    path="temp.jpeg"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    frame= rec_model.run(path)
    _,frame = cv2.imencode('.JPEG', frame)
    return StreamingResponse(io.BytesIO(frame.tobytes()), media_type="image/png")
        
    
