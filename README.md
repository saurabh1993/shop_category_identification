# Shop_category_identification
This project aims to identify shop category-(fresh,retail) based on images provided by merchants

## Steps
### 1. Identify categories based on feature similarity<br>
Images in the fresh category were very dissimilar like fruits/veggies shop were dissimilar to chicken/poultry shops<br>
Therefore created seperate sub categories for the below categories<br>
&nbsp;&nbsp;&nbsp;&nbsp;    a. Poultry-meat<br>
&nbsp;&nbsp;&nbsp;&nbsp;   b. Fruits-veggies<br>
&nbsp;&nbsp;&nbsp;&nbsp;   c. Misc(eggs, bread)<br>
<br>
### 2. Balance the dataset - The dataset seemed to disbalanced in few categories. Hence manually added few training labels<br>

### 3. Local Training 
Mobilenet classifier was selected as the base model. Few extra layers for better complexity handling were added.<br>
Needed to tune the trainable layers based on training evaluation.Only extra added layers were set to trainable<br>
To train- Run
```
python3 training.py
```
#### Training params<br>
&nbsp;&nbsp;&nbsp;&nbsp; non-Trainable layers:82<br>
&nbsp;&nbsp;&nbsp;&nbsp; Trainable layers:5<br>
&nbsp;&nbsp;&nbsp;&nbsp; Training epochs :20<br>
&nbsp;&nbsp;&nbsp;&nbsp; Training batch-size :8<br>
&nbsp;&nbsp;&nbsp;&nbsp; Training-validation data split : 0.8-0.2
#### Training Metrics:
![alt text](https://raw.githubusercontent.com/saurabh1993/shop_category_identification/master/local_evaluation.png)
&nbsp;&nbsp;&nbsp;&nbsp; Training loss: 0.0291 <br>
&nbsp;&nbsp;&nbsp;&nbsp; Training accuracy: 0.9945<br>
&nbsp;&nbsp;&nbsp;&nbsp; val_loss: 1.4986 <br>
&nbsp;&nbsp;&nbsp;&nbsp; val_accuracy: 0.7500 <br>
#### Model path - localModels folder
#### Inference usage
```
from inference_local import localModel
localModel =autoMLModel()
Result=localModel.run('/home/dog.jpg')
```
#### Output
```
{
    "veg-fresh": "0.015644",
    "egg-fresh": "0.018934",
    "meat-fresh": "0.947371",
    "grocery": "0.018051"
  }
```
### 4. AutoML Training
Google's auo ml training was also used parallely for the best output results<br>
Needed basic configurations and labelling config scripts to configure training<br>
Result received in less than a hour<br>
#### Inference usage
```
from inference_google import autoMLModel
autoModel =autoMLModel()
Result=autoModel.run('/home/dog.jpg')
```
#### Output
```
{
    "veg-fresh": "0.015644",
    "egg-fresh": "0.018934",
    "meat-fresh": "0.947371",
    "grocery": "0.018051"
  }
```
#### Model path - autoMLModel folder
#### Evaluation Metrics
![alt text](https://raw.githubusercontent.com/saurabh1993/shop_category_identification/master/evaluation.png)

### 5. Image view rectification
Sometimes due to wrong camera orientation , images can be sideways. To rectify, I tried to rotate the image in clock and counterwise direction<br>
Objection detection inferencing was done and image with highest numbers of detections is assumed to be the the straight image

#### Usage
```
from image_correction import image_rectifier
rec_model = image_rectifier()
Result=rec_model.run('/home/dog.jpg')# Returs jpeg buffer

```
### 6. Serving
Created a rest based fast-api server with the below routes:<br>
To run the server<br>
```
uvicorn server:app --reload
```
##### 1. Shop_category
```
POST localhost:8000/shop_category
# post input- image file in a multipart request
```
##### output
```
{
  "autoML": {
    "veg-fresh": "0.015644",
    "egg-fresh": "0.018934",
    "meat-fresh": "0.947371",
    "grocery": "0.018051"
  },
  "Mobilenet": {
    "egg-fresh": "0.000000",
    "grocery": "0.999906",
    "meat-freah": "0.000094",
    "veg-fresh": "0.000000"
  }
}
```
##### 2. rectify_category
```
POST localhost:8000/rectify
# post input- image file in a multipart request
```

### 7. Next steps
    1. Use text based recognition to identify she category-[chicken,bread,milk,fruit] etc<br>
    2. Use object detection based approach to identify shop categories-[apple,bird,specific object training like chips packets etc]<br>

