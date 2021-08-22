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



'''Route for identifying shop category,
Runs inferencing for autoML model
Runs inferencing for locally trained Mobilenet model
Sends a json response of both outputs
'''
@app.post("/shop_category/")
async def categorize_shop(file: UploadFile = File(...)):
    global autoModel
    global localModel
    
    # Save the uploaded image to local directory
    path="temp.jpeg"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    out={}
    
    # Pass the path to the models to run inferencing
    autoResult=autoModel.run(path)
    out["autoML"]=autoResult
    
    localResult=localModel.run(path)
    out["Mobilenet"]=localResult
    
    # Convert to json compatible response
    json_compatible_item_data = jsonable_encoder(out)
    print("final",json_compatible_item_data)
    return JSONResponse(content=json_compatible_item_data)

'''Route for correcting inverted images to
straight images, returns straight output image
'''    
@app.post("/rectifyImage/")
async def rectify_image(file: UploadFile = File(...)):
    global rec_model
    path="temp.jpeg"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run the rectification process    
    frame= rec_model.run(path)
    _,frame = cv2.imencode('.JPEG', frame)
    
    # Return rectified image
    return StreamingResponse(io.BytesIO(frame.tobytes()), media_type="image/png")
        
    
