from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import math
import numpy as np
from mtcnn import MTCNN

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

faceProto = "content/opencv_face_detector.pbtxt"
faceModel = "content/opencv_face_detector_uint8.pb"
ageProto = "content/age_deploy.prototxt"
ageModel = "content/age_net.caffemodel"
genderProto = "content/gender_deploy.prototxt"
genderModel = "content/gender_net.caffemodel"
padding=20

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

#Using below user defined function we get the coordinates for bounding boxes or we can say location of face in image
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)   
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

def age_gender_detector(frame):
    # Read frame
    frameFace, bboxes = getFaceBox(faceNet, frame)
    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1), max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        # print("Gender Output: {}".format(genderPreds))
        print("Gender: {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
        
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print("Age: {}, conf = {:.3f}".format(age, agePreds[0].max()))
        
        label = "{},{}".format(gender, age)
        cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    return frameFace

class Detector:
    def __init__(self, model_type="OD"):
        self.cfg = get_cfg()
        self.mtcnn=MTCNN()

        # Load Model config and pretrained model
        if model_type == "OD":    # object detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model_type == "IS":    # instance segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "KP":    # keypoint detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"  # cpu or cuda

        self.predictor = DefaultPredictor(self.cfg)
        
    def extract_keypoints(self,instances):
        keypoints=instances.pred_keypoints.numpy()[:,:,:2]
        return keypoints

    def onVideo(self, videoPath):
        cap = cv2.VideoCapture(videoPath)

        if not cap.isOpened():
            print("Error opening the file...")
            return

        success, image = cap.read()
        frame_count = 0
        skip_frames = 2

        tracked_keypoints = {}
        track_length_threshold = 10
        presence_duration = {}

        while success:
            if frame_count % skip_frames == 0:
                predictions = self.predictor(image)
                instances = predictions["instances"].to("cpu")

                keypoints = instances.pred_keypoints.numpy()
                person_ids = instances.pred_classes.numpy()

                for i in range(len(person_ids)):
                    person_id = person_ids[i]
                    keypoints_person = keypoints[i]

                    if person_id not in tracked_keypoints:
                        tracked_keypoints[person_id] = []

                    tracked_keypoints[person_id].append(keypoints_person)
                    if len(tracked_keypoints[person_id]) > track_length_threshold:
                        tracked_keypoints[person_id].pop(0)

                    # Check if person_id exists in the presence duration dictionary
                    if person_id in presence_duration:
                        presence_duration[person_id] += 1  # Increment duration
                    else:
                        presence_duration[person_id] = 1  # Initialize duration

                v = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                                instance_mode=ColorMode.IMAGE)
                output = v.draw_instance_predictions(instances)
                output_image = output.get_image()[:, :, ::-1].copy()
                output_image=age_gender_detector(output_image)

                cv2.imshow("Result", output_image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                print("Presence duration for each person:")
                for person_id, duration in presence_duration.items():
                    print("Person ID:", person_id, "Duration:", duration)

            frame_count += 1
            success, image = cap.read()

        cap.release()
        cv2.destroyAllWindows()


