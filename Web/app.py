from fileinput import filename
import re
from flask import Flask, request, render_template, redirect
import tensorflow as tf
import numpy as np
from numpy import expand_dims, asarray
import cv2
import keras
import PIL
from numpy import expand_dims
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.ndimage import zoom
from skimage.feature import hog
import time
import scipy
from werkzeug.utils import secure_filename
from flask import Flask, render_template, Response
from FER_Camera import VideoCamera, BoundBox


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


model = load_model('models/model.h5')
clf = load_model('models/xception_model_combined_best1.h5')

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2] # 0 and 1 is row and column 13*13
    nb_box = 3 # 3 anchor boxes
    netout = netout.reshape((grid_h, grid_w, nb_box, -1)) #13*13*3 ,-1
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh
    
    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if(objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

#intersection over union        
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    
    
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin  
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    #Union(A,B) = A + B - Inter(A,B)
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union

def do_nms(boxes, nms_thresh):    #boxes from correct_yolo_boxes and  decode_netout
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename) #load_img() Keras function to load the image .
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape) # target_size argument to resize the image after loading
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0  #rescale the pixel values from 0-255 to 0-1 32-bit floating point values.
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height

# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
    
    return v_boxes, v_labels, v_scores

def get_label(argument):
    labels = {0:'Neutral', 1:'Happy', 2:'Sad', 3:'Surprise', 4:'Fear' , 5:'Disgust', 6:'Anger', 7:'Contempt'}
    return(labels.get(argument, "Invalid emotion"))

def face_crop(filename, v_boxes, v_labels, v_scores):
    img = cv2.imread(filename)
    rows, cols = img.shape[0], img.shape[1]
    shape_x=48.0
    shape_y=48.0
    faces=[]
    for i in range(len(v_boxes)):
        if (not v_boxes):
            x1,y1 = 0,0
            x2,y2 = 0,0
        else:
            box = v_boxes[i]
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            cropped_image = img[y1:y2, x1:x2]
            if cropped_image.shape[0]==0 or cropped_image.shape[1]==0:
              print(cropped_image.shape,v_boxes[i],filename)
              continue
            new_extracted_face = zoom(cropped_image, (shape_x / cropped_image.shape[0],shape_y / cropped_image.shape[1],1),order=3, mode='wrap')
            #cast type float
            new_extracted_face = new_extracted_face.astype(np.float32)
            #scale
            new_extracted_face /= 255.0
            faces.append(new_extracted_face)
    return faces

# define the anchors
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]  

# define the probability threshold for detected objects
class_threshold = 0.6

labels = ["face"]

input_h = 480
input_w = 640
def find_face(photo_filename):
  boxes = list()
  image, image_w, image_h = load_image_pixels(photo_filename, (input_h, input_w))
  yhat = model.predict(image)
  for i in range(len(yhat)):
      # decode the output of the network
      boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
      
  # correct the sizes of the bounding boxes for the shape of the image
  correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

  # suppress non-maximal boxes
  do_nms(boxes, 0.5)  #Discard all boxes with pc less or equal to 0.5

  # get the details of the detected objects
  v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

  
  return v_boxes, v_labels, v_scores

def draw_boxes(img , box, label):
    y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
    start_point = (x1, y1) 
    # Ending coordinate
    # represents the bottom right corner of rectangle 
    end_point = (x2, y2) 
    # Red color in BGR 
    color = (0, 0, 255) 
    # Line thickness of 2 px 
    thickness = 2
    # font 
    font = cv2.FONT_HERSHEY_PLAIN 
    # fontScale 
    fontScale = 1.5
    #create the shape
    img = cv2.rectangle(img, start_point, end_point, color, thickness) 
    # draw text and score in top left corner
    text = "%s" % (get_label(label))
    img = cv2.putText(img, text, (x1,y1), font,  
                fontScale, color, thickness, cv2.LINE_AA)
    return img


app = Flask(__name__)

flag = 1

@app.route('/v2',methods=['POST','GET'])
def index1():
    if request.method=='POST':
        result=request.files['file']
        result.save(secure_filename(result.filename))
        filename = secure_filename(result.filename)
        v_boxes, v_labels, v_scores = find_face(filename)
        img=cv2.imread(filename)
        if len(v_boxes):
            for i in range(len(v_boxes)):
                if (not v_boxes):
                    x1,y1 = 0,0
                    x2,y2 = 0,0
                else:
                    box = v_boxes[i]
                    y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
                    cropped_image = img[y1:y2, x1:x2]
                    if cropped_image.shape[0]==0 or cropped_image.shape[1]==0:
                        print(cropped_image.shape,v_boxes[i],filename)
                        continue
                    face_1 = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    new_extracted_face = zoom(face_1, (48 / cropped_image.shape[0],48 / cropped_image.shape[1]),order=3, mode='wrap')
                    #cast type float
                    new_extracted_face = new_extracted_face.astype(np.float32)
                    new_extracted_face /= 255.0
                    f, hog_image = hog(new_extracted_face, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True)
                    new_combined = np.concatenate((new_extracted_face,hog_image),axis=0)
                    image1 = new_combined.reshape((1,96, 48,1))
                    ypred=clf.predict(image1)
                    img = draw_boxes(img,box,ypred.argmax(1)[0])
        cv2.imwrite("static/output.jpg",img)
        return render_template('show.html')
    return redirect("/")
        

@app.route("/", methods = ['GET','POST'])
@app.route("/index", methods = ['GET','POST'])
def index():
    return render_template("index.html")

def generate(camera):
    while True:
        frame = camera.get_frame(flag)
        try:
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n') 
        except:
            pass
        
@app.route("/capture")   
def capture():
    flag = 0
    print(flag,"flag")
    return redirect("/result")


@app.route("/video_feed")
def video_feed():
    if flag ==0:
        return redirect("/result")
    return Response(generate(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video")
def video():
    return render_template("video.html")

@app.route("/result")
def result():
    ret,frame = cv2.VideoCapture(0).read()
    cv2.VideoCapture(0).release()
    resized_img = cv2.resize(frame, (640, 480))
    # _, jpeg = cv2.imencode('.jpg', resized_img)
    # jpeg = cv2.resize(jpeg, (480, 640))
    print(resized_img.shape)
    filename = "static/input.jpg"
    try:
        cv2.imwrite("static/input.jpg",resized_img)
    except:
        print("pass")
    v_boxes, v_labels, v_scores = find_face(filename)
    img=cv2.imread(filename)
    if len(v_boxes):
        for i in range(len(v_boxes)):
            if (not v_boxes):
                x1,y1 = 0,0
                x2,y2 = 0,0
            else:
                box = v_boxes[i]
                y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
                cropped_image = img[y1:y2, x1:x2]
                if cropped_image.shape[0]==0 or cropped_image.shape[1]==0:
                    print(cropped_image.shape,v_boxes[i],filename)
                    continue
                face_1 = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                new_extracted_face = zoom(face_1, (48 / cropped_image.shape[0],48 / cropped_image.shape[1]),order=3, mode='wrap')
                #cast type float
                new_extracted_face = new_extracted_face.astype(np.float32)
                new_extracted_face /= 255.0
                f, hog_image = hog(new_extracted_face, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True)
                new_combined = np.concatenate((new_extracted_face,hog_image),axis=0)
                image1 = new_combined.reshape((1,96, 48,1))
                ypred=clf.predict(image1)
                img = draw_boxes(img,box,ypred.argmax(1)[0])
    cv2.imwrite("static/output.jpg",img)
    cv2.VideoCapture(0).release()
    cv2.destroyAllWindows()
    return render_template('show.html')

if __name__=='__main__':
    app.run(debug=True)
