# shop_category_identification
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
#### Training params<br>
&nbsp;&nbsp;&nbsp;&nbsp; non-Trainable layers:82<br>
&nbsp;&nbsp;&nbsp;&nbsp; Trainable layers:5<br>
&nbsp;&nbsp;&nbsp;&nbsp; Training epochs :20<br>
&nbsp;&nbsp;&nbsp;&nbsp; Training batch-size :8<br>
&nbsp;&nbsp;&nbsp;&nbsp; Training-validation data split : 0.8-0.2
#### Training Metrics:
&nbsp;&nbsp;&nbsp;&nbsp; Training loss: 0.0291 <br>
&nbsp;&nbsp;&nbsp;&nbsp; Training accuracy: 0.9945<br>
&nbsp;&nbsp;&nbsp;&nbsp; val_loss: 1.4986 <br>
&nbsp;&nbsp;&nbsp;&nbsp; val_accuracy: 0.7500 <br>

### 4. AutoML Training
Google's auo ml training was also used parallely for the best output results<br>
Needed basic configurations and labelling config scripts to configure training<br>
Result received in less than a hour<br>
#### Evaluation Metrics
&nbsp;&nbsp;&nbsp;&nbsp; Training-validation data split : 0.9-0.1



