from Detector import *

detector=Detector(model_type="KP")  #OD,IS,KP

#detector.onImage("C:\\Users\\SAMSUNG\\detectron2\\detectron2\\images\\1.jpg")

detector.onVideo("people.mp4")
