# Shop_category_identification
This project aims to identify shop category-(fresh/retail) based on images.

## Steps
### 1. Identify categories based on feature similarity<br>
Images in the fresh category had a very high variance example fruits/veggies shop images were dissimilar to chicken/poultry shops<br>
Therefore I created seperate sub categories for the below categories(dairy and fresh images were categorized into further groups)<br>
a. Poultry-meat(meat-fresh)<br>
b. Fruits-veggies(veg-fresh)<br>
c. Egg-dairy(egg-fresh)<br>
d. grocery(grocery)<br>

### 2. Balance the dataset 
The dataset seemed to disbalanced in few categories. Hence I also manually added few images to some training categories<br>

### 3. Local Training 
Mobilenet classifier was selected as the base model. Few extra layers for better complexity handling were added.<br>
Needed to tune the trainable layers based on training evaluation.Only extra added layers were set to trainable<br>
To train- Run the training script
```
python3 training.py
```
#### Training params<br>
Non-Trainable layers:82<br>
Trainable layers:5<br>
Training epochs :20<br>
Training batch-size :8<br>
Training-validation data split : 0.8-0.2
#### Training classes
```
CLASSES={0:"egg-fresh",1:"grocery",2:"meat-freah",3:"veg-fresh"}
```
#### Training Metrics:
![alt text](https://raw.githubusercontent.com/saurabh1993/shop_category_identification/master/local_evaluation.png)
Training loss: 0.0291 <br>
Training accuracy: 0.9945<br>
val_loss: 1.4986 <br>
val_accuracy: 0.7500 <br>
#### Model path - localModels folder
#### Inference usage
```
from inference_local import localModel
localModel =localModel()
Result=localModel.run('/home/dog.jpg')
```
#### Example Output
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
#### Evaluation Metrics
![alt text](https://raw.githubusercontent.com/saurabh1993/shop_category_identification/master/evaluation.png)

#### Inference usage
```
from inference_google import autoMLModel
autoModel =autoMLModel()
Result=autoModel.run('/home/dog.jpg')
```
#### Examample Output
```
{
    "veg-fresh": "0.015644",
    "egg-fresh": "0.018934",
    "meat-fresh": "0.947371",
    "grocery": "0.018051"
  }
```
#### Model path - autoMLModel folder
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
Created a rest based fast-api server to serve mobilenet+autoML model with the below routes:<br>
To run the server<br>
```
uvicorn server:app --reload
```
##### 1. Shop_category
```
POST localhost:8000/shop_category
# post input- image file in a multipart request
```
##### Input image
![alt text](https://raw.githubusercontent.com/saurabh1993/shop_category_identification/master/temp.jpeg)
##### output
```
{
  "autoML": {
    "veg-fresh": "0.828243",
    "egg-fresh": "0.054569",
    "meat-fresh": "0.037253",
    "grocery": "0.079934"
  },
  "Mobilenet": {
    "egg-fresh": "0.000000",
    "grocery": "0.000000",
    "meat-freah": "0.000000",
    "veg-fresh": "1.000000"
  }
}
```
##### 2. rectify_category
```
POST localhost:8000/rectify
# post input- image file in a multipart request
```
##### Input
![alt text](https://raw.githubusercontent.com/saurabh1993/shop_category_identification/master/side.jpeg)
##### Output
![alt text](https://raw.githubusercontent.com/saurabh1993/shop_category_identification/master/right.jpeg)


### 7. Next steps
    1. Use text based recognition to identify she category-[chicken,bread,milk,fruit] etc
    2. Use object detection based approach to identify shop categories-[apple,bird,specific object training like chips packets etc]


