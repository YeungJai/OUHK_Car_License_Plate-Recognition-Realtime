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
