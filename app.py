
from flask import Flask,render_template,Response
import cv2
from pydoc import stripid
import string
from traceback import print_tb
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pytesseract

app=Flask(__name__)
#camera=cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(Realtime().run(),mimetype='multipart/x-mixed-replace; boundary=frame')

class Realtime:

    def __init__(self):
        self.loadDnn()
    
    def loadDnn(self):
        self.net = cv2.dnn.readNet("yolov3_train_last333.weights", "yolov3_testing1.cfg")
        self.classes = None
        self.roi_image = None
        self.fig_image = None
        self.img = None
        self.color = (255,0,0)
        with open("classes1.txt", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, 3)
        # Loading camera
        self.cap = cv2.VideoCapture(0)
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.starting_time = time.time()
        self.frame_id = 0

    def run(self):
        cap = cv2.VideoCapture(0)
        while(cap.isOpened()):
            ret, frame = cap.read()
            self.frame_id += 1
            self.fig_image = frame
            self.img = frame.copy()
            height, width, channels = frame.shape
    
    
        # Detecting objects
            blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)

            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            coordinates = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.2:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width - 2)
                        h = int(detection[3] * height - 2)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

            for i in range(len(boxes)):
                if i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    
                    label = str(self.classes[class_ids[i]])
                    confidence = str(round(confidences[i],1))
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), self.color, 15)
                    coordinates = (x,y,w,h)
                    print(coordinates)

                    if(label == 'license-plate'):
                        self.fig_image = self.img
                        x, y, w, h = coordinates
                        roi = self.img[y:y + h, x:x + w]
                        self.roi_image = roi
                        #plt.figure(figsize=(20, 14))
                        plt.axis('off')
                        #plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        gray = cv2.threshold(gray, 230, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
                        thresh = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                        cv2.imwrite("oneline.png", gray)
                        plt.savefig('94.png')
                        #plt.savefig('94.png', bbox_inches='tight',pad_inches = 0)
                        custom_config = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNPQRSTUVWXYZ1234567890 --psm 7'
                        code = pytesseract.image_to_string(gray, config=custom_config)
                        code = code.replace("\n", "")
                        code2 = code.replace("\t", "")
                        code3 = code2.rstrip("\n")
                        
                        print (code3.rstrip("\n"))
                        cv2.putText(frame, code3.strip() , (x, y + 5), self.font, 2, (0,255,0), 2)

                    if(label == 'license-plate-twoline'):
                        self.fig_image = self.img
                        x, y, w, h = coordinates
                        roi = self.img[y:y + h, x:x + w]
                        self.roi_image = roi
                        #plt.figure(figsize=(10, 4))
                        plt.axis('off')
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite("twoline.png", gray)

                        image = cv2.imread('twoline.png')
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                        thresh = cv2.adaptiveThreshold(blurred, 255,
                        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)

                        _, labels = cv2.connectedComponents(thresh)
                        mask = np.zeros(thresh.shape, dtype="uint8")

                        # Set lower/upper bound criteria for characters
                        total_pixels = image.shape[0] * image.shape[1]
                        lower = total_pixels // 70 
                        upper = total_pixels // 20 
                        # Loop over the unique components
                        for (i, label) in enumerate(np.unique(labels)):
                        # If this is the background label, ignore it
                            if label == 0:
                                continue
                    
                        labelMask = np.zeros(thresh.shape, dtype="uint8")
                        labelMask[labels == label] = 230
                        numPixels = cv2.countNonZero(labelMask)
                
                        # add it to our mask
                        if numPixels > lower and numPixels < upper:
                            mask = cv2.add(mask, labelMask)
                
                        cv2.imwrite('result.png', thresh)
                        # cv2.imread ##
                        img = cv2.imread('result.png')

                        h, w, channels = img.shape
                
                        half = w//2
                
                
                        # this will be the first column
                        left_part = img[:, :half] 
                
                        # [:,:half] means all the rows and
                        # all the columns upto index half
                
                        # this will be the second column
                        right_part = img[:, half:]  
                
                        # this is horizontal division
                        half2 = h//2
                
                        top = img[:half2, :]
                        bottom = img[half2:, :]
                
                        custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNPQRSTUVWXYZ1234567890 --psm 7'
                        text = pytesseract.image_to_string(top, config=custom_config)
                        custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNPQRSTUVWXYZ1234567890 --psm 7'
                        text2 = pytesseract.image_to_string(bottom, config=custom_config)
                        cv2.putText(frame, text.strip() + text2.strip() , (x, y + 5), self.font, 2, (0,255,0), 2)
                
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__=="__main__":
    run = Realtime.__new__(Realtime)
    run.__init__()
    app.run(debug=True)