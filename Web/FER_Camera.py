from tensorflow.keras.models import model_from_json

import cv2
import numpy as np
from keras.models import load_model
from skimage.feature import hog
import time
from scipy.ndimage import zoom

# json_file = open("model.json", "r")
# loaded_json_model = json_file.read()
# json_file.close()

# model = model_from_json(loaded_json_model)
# model.load_weights("model_weights.h5")
clf = load_model('models/xception_model_combined_best1.h5')
model = load_model('models/model.h5')
# face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
labels = {0:'Neutral', 1:'Happy', 2:'Sad', 3:'Surprise', 4:'Fear' , 5:'Disgust', 6:'Anger', 7:'Contempt'}

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1
        
    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        
        return self.score



class VideoCamera:

    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()

    def get_frame(self,flag):
        if flag:
            ret,test_img=self.video.read()
            resized_img = cv2.resize(test_img, (1000, 600))

            _, jpeg = cv2.imencode('.jpg', resized_img)

            return jpeg.tobytes()

        
    # def draw_boxes(self,img , box, label):
    #     y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
    #     start_point = (x1, y1) 
    #     # Ending coordinate
    #     # represents the bottom right corner of rectangle 
    #     end_point = (x2, y2) 
    #     # Red color in BGR 
    #     color = (0, 0, 255) 
    #     # Line thickness of 2 px 
    #     thickness = 2
    #     # font 
    #     font = cv2.FONT_HERSHEY_PLAIN 
    #     # fontScale 
    #     fontScale = 1.5
    #     #create the shape
    #     img = cv2.rectangle(img, start_point, end_point, color, thickness) 
    #     # draw text and score in top left corner
    #     text = ""
    #     img = cv2.putText(img, text, (x1,y1), font,  
    #                 fontScale, color, thickness, cv2.LINE_AA)
    #     return img

    # def correct_yolo_boxes(self,boxes, image_h, image_w, net_h, net_w):
    #     new_w, new_h = net_w, net_h
    #     for i in range(len(boxes)):
    #         x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
    #         y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
    #         boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
    #         boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
    #         boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
    #         boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

    # def _interval_overlap(self,interval_a, interval_b):
    #     x1, x2 = interval_a
    #     x3, x4 = interval_b
    #     if x3 < x1:
    #         if x4 < x1:
    #             return 0
    #         else:
    #             return min(x2,x4) - x1
    #     else:
    #         if x2 < x3:
    #             return 0
    #         else:
    #             return min(x2,x4) - x3

    # def _sigmoid(self,x):
    #     return 1. / (1. + np.exp(-x))

    # #intersection over union        
    # def bbox_iou(self,box1, box2):
    #     intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    #     intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    #     intersect = intersect_w * intersect_h
        
        
    #     w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin  
    #     w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
        
    #     #Union(A,B) = A + B - Inter(A,B)
    #     union = w1*h1 + w2*h2 - intersect
    #     return float(intersect) / union

    # def do_nms(self,boxes, nms_thresh):    #boxes from correct_yolo_boxes and  decode_netout
    #     if len(boxes) > 0:
    #         nb_class = len(boxes[0].classes)
    #     else:
    #         return
    #     for c in range(nb_class):
    #         sorted_indices = np.argsort([-box.classes[c] for box in boxes])
    #         for i in range(len(sorted_indices)):
    #             index_i = sorted_indices[i]
    #             if boxes[index_i].classes[c] == 0: continue
    #             for j in range(i+1, len(sorted_indices)):
    #                 index_j = sorted_indices[j]
    #                 if self.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
    #                     boxes[index_j].classes[c] = 0

    # def decode_netout(self,netout, anchors, obj_thresh, net_h, net_w):
    #     grid_h, grid_w = netout.shape[:2] # 0 and 1 is row and column 13*13
    #     nb_box = 3 # 3 anchor boxes
    #     netout = netout.reshape((grid_h, grid_w, nb_box, -1)) #13*13*3 ,-1
    #     nb_class = netout.shape[-1] - 5
    #     boxes = []
    #     netout[..., :2]  = self._sigmoid(netout[..., :2])
    #     netout[..., 4:]  = self._sigmoid(netout[..., 4:])
    #     netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    #     netout[..., 5:] *= netout[..., 5:] > obj_thresh
        
    #     for i in range(grid_h*grid_w):
    #         row = i / grid_w
    #         col = i % grid_w
    #         for b in range(nb_box):
    #             # 4th element is objectness score
    #             objectness = netout[int(row)][int(col)][b][4]
    #             if(objectness.all() <= obj_thresh): continue
    #             # first 4 elements are x, y, w, and h
    #             x, y, w, h = netout[int(row)][int(col)][b][:4]
    #             x = (col + x) / grid_w # center position, unit: image width
    #             y = (row + y) / grid_h # center position, unit: image height
    #             w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
    #             h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
    #             # last elements are class probabilities
    #             classes = netout[int(row)][col][b][5:]
    #             box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
    #             boxes.append(box)
    #     return boxes
    
    # def get_boxes(self,boxes, labels, thresh):
    #     v_boxes, v_labels, v_scores = list(), list(), list()
    #     # enumerate all boxes
    #     for box in boxes:
    #         # enumerate all possible labels
    #         for i in range(len(labels)):
    #             # check if the threshold for this label is high enough
    #             if box.classes[i] > thresh:
    #                 v_boxes.append(box)
    #                 v_labels.append(labels[i])
    #                 v_scores.append(box.classes[i]*100)
        
    #     return v_boxes, v_labels, v_scores

    # def find_face_img(self,image):
    #     anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]] 
    #     image = image.reshape(1,480,640,3)
    #     class_threshold = 0.6 
    #     input_h = 480
    #     input_w = 640
    #     boxes = list()
    #     yhat = model.predict(image)
    #     for i in range(len(yhat)):
    #         # decode the output of the network
    #         boxes += self.decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
            
    #     # correct the sizes of the bounding boxes for the shape of the image
    #     self.correct_yolo_boxes(boxes, 48, 48, input_h, input_w)
        
        
    #     print(5)
    #     # suppress non-maximal boxes
    #     self.do_nms(boxes, 0.5)  #Discard all boxes with pc less or equal to 0.5

    #     # get the details of the detected objects
    #     v_boxes, v_labels, v_scores = self.get_boxes(boxes, labels, class_threshold)
    #     print(5)

        
    #     return v_boxes, v_labels, v_scores


    # def get_frame(self):
    #     ret,test_img=self.video.read()# captures frame and returns boolean value and captured image  
    #     v_boxes, v_labels, v_scores = self.find_face_img(test_img)
    #     print(4)
    #     if len(v_boxes):
    #         for i in range(len(v_boxes)):
    #             if (not v_boxes):
    #                 x1,y1 = 0,0
    #                 x2,y2 = 0,0
    #             else:
    #                 box = v_boxes[i]
    #                 y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
    #                 cropped_image = test_img[y1:y2, x1:x2]
    #                 print(3)
    #                 if cropped_image.shape[0]==0 or cropped_image.shape[1]==0:
    #                     continue
    #                 face_1 = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    #                 new_extracted_face = zoom(face_1, (48 / cropped_image.shape[0],48 / cropped_image.shape[1]),order=3, mode='wrap')
    #                 #cast type float
    #                 new_extracted_face = new_extracted_face.astype(np.float32)
    #                 new_extracted_face /= 255.0
    #                 print(2)
    #                 f, hog_image = hog(new_extracted_face, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True)
    #                 new_combined = np.concatenate((new_extracted_face,hog_image),axis=0)
    #                 image1 = new_combined.reshape((1,96, 48,1))
    #                 print(1)
    #                 ypred=clf.predict(image1)
    #                 img = self.draw_boxes(test_img,box,ypred.argmax(1)[0])

    #     # gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  

        
    #     # # faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    #     # for (x,y,w,h) in faces_detected:  
    #     #     roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
    #     #     # roi_gray=cv2.resize(roi_gray,(48,48))  
    #     #     # img = roi_gray.reshape((1,48,48,1))
    #     #     # img = img /255.0
    #     #     new_extracted_face = zoom(roi_gray, (48 / roi_gray.shape[0],48 / roi_gray.shape[1]),order=3, mode='wrap')
    #     #     #cast type float
    #     #     new_extracted_face = new_extracted_face.astype(np.float32)

    #     #     # max_index = np.argmax(model.predict(img.reshape((1,48,48,1))), axis=-1)[0]
    #     #     f, hog_image = hog(new_extracted_face, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True)
    #     #     new_combined = np.concatenate((new_extracted_face,hog_image),axis=0)
    #     #     image1 = new_combined.reshape((1,96, 48,1))
    #     #     ypred=clf.predict(image1)
    #     #     max_index = ypred.argmax(1)[0]
            
    #     #     predicted_emotion = labels.get(max_index, "Invalid emotion")
    #     #     cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=3)  
    #     #     cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    

    #     resized_img = cv2.resize(test_img, (1000, 600))

    #     _, jpeg = cv2.imencode('.jpg', resized_img)

    #     return jpeg.tobytes()