# Car_License_Plate_Recognition(Realtime)
Car License Plate numbers and characters Recognition by YOLOv3 and Pytesseract
<li>Recognize the numbers and characters of license plates based on Hong Kong.</li>
<li>Support Hong Kong license plate (one&two-line).</li>

## Enviroment and package
- Python 3.6.8
- [Labelimg](https://github.com/tzutalin/labelImg "Labelimg")
- [Python virtual environments (venv)](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/ "Python virtual environments (venv)")
- [OpenCV](https://github.com/skvark/opencv-python "OpenCV")
- [pytesseract](https://github.com/madmaze/pytesseract "pytesseract")
- [matplotlib](https://matplotlib.org/ "matplotlib") 

## Project idea
<ol>
  <li>Use the YOLOv3 custom model to recognize the position of the car plate.</li>
  <li>Use OpenCV to process the position of the car plate. Based on Hong Kong car plates there have two types so need concern. </li>
  <li>Dependence the position to do the OCR process by using the pytesseract to recognize those car codes and characters.</li>
</ol>

## Annotating Images using LabelImg
Collect the car license plate data set of more than 150 images of each class(one & two line car plates). I found the Hong Kong license plate from Google, Flickr which is free.
![Picture1](https://user-images.githubusercontent.com/52642596/174001566-65481608-9a55-4c9c-b64f-627856a64d4a.jpg)

The labeling IMG software will save a file within the classes containing the boxes for each class. to one of its image classes to save the list class.txt

In yolov3, the values are fed into the system which has a specified format and the data is in txt format which contains the classes and some values. These values look like below. 
The order of yolo format txt files follows class, x, y, w, h 
<li>x = Absolute x / width of total image</li>
<li>y = Absolute y / height of total image</li>
<li>w = Absolute width / width of total image </li>
<li>h = Absolute height / height of total image </li>

## Model training
Step 1, in a new colab notebook go to Runtime Change runtime type, and select GPU
![image](https://user-images.githubusercontent.com/52642596/174005713-8c7bf584-2153-466f-b728-64dbd4708dbe.png)

Step 2 Mount Google Drive, In Google Drive, create a backup folder. I’ve named mine yolo-license-plates. That’s where model weights and configuration will get stored.

In the first cell, execute the following code to mount Google Drive:
```bash
from google.colab import drive 
drive.mount('/content/gdrive')

!ln -s /content/gdrive/My\ Drive/ /mydrive
```
Step 3 Download and configure Darknet,Darknet is an open-source neural network framework that features a YOLO object detection system. To download it, execute this line from a code cell:
```bash
!git clone https://github.com/AlexeyAB/darknet
```
Step 4 Configure settings files,To know how to set up the YOLO configuration file, you need to know how many classes there are. Next, you need to calculate the number of batches and the number of filters. Here are the formulas:

<li>Batches = number of classes * 2000</li>
<li>Filters = (number of classes + 5) * 3</li>
