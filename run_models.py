import logging
import os

logging.basicConfig(level=logging.INFO)

runYolov5 = True
runYolov7 = True
runOpenCv = True

yolov5Params = "python ./yolov5/detect.py --source ./tpProf_cars_dataset/images --save-txt --agnostic-nms --classes 2 3 5 7 --weights ./yolov5/"
yolov5Models = ["yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]
name = " --name "

# * run yolov5 models
if runYolov5:
    for model in yolov5Models:
        modelParams = yolov5Params + model + name + model

        logging.info("#### Running: %s ####", model)
        os.system(modelParams)
        logging.info("#### Finished Running: %s ####", model)

yolov7Params = "python ./yolov7/detect.py --source ./tpProf_cars_dataset/images --conf 0.25 --img-size 640 --save-txt --agnostic-nms --class 2 3 5 7 --weights ./yolov7/"
yolov7Models = ["yolov7-tiny.pt", "yolov7.pt", "yolov7x.pt"]

# * run yolov7 models
if runYolov7:
    for model in yolov7Models:
        modelParams = yolov7Params + model + name + model

        logging.info("#### Running: %s ####", model)
        os.system(modelParams)
        logging.info("#### Finished Running: %s ####", model)

# * models running with OpenCV
yolov3 = {
    "name": "yolov3",
    "weights": "./models_openCV/yolov3/yolov3.weights",
    "config": "./models_openCV/yolov3/yolov3.cfg",
}
yolov4 = {
    "name": "yolov4",
    "weights": "./models_openCV/yolov4/yolov4.weights",
    "config": "./models_openCV/yolov4/yolov4.cfg",
}
ssdMobileNetV3 = {
    "name": "ssdMobileNetV3",
    "weights": "./models_openCV/SSD_MobileNet_V3/frozen_inference_graph.pb",
    "config": "./models_openCV/SSD_MobileNet_V3/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt",
}
openCvModels = [yolov3, yolov4, ssdMobileNetV3]

# running commands
openCvParams = "python detection_openCV.py --dataSetPath "
openCvDataSetPath = ".\\tpProf_cars_dataset"
openCvWeights = " --weights "
openCvConfig = " --config "
openCvDirName = " --dirName "

if runOpenCv:
    for model in openCvModels:
        modelParams = (
            openCvParams
            + openCvDataSetPath
            + openCvWeights
            + model["weights"]
            + openCvConfig
            + model["config"]
            + openCvDirName
            + model["name"]
        )

        logging.info("#### Running: %s ####", model)
        os.system(modelParams)
        logging.info("#### Finished Running: %s ####", model)
