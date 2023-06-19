import logging
import os
import sys

i = sys.argv[1] #este argumento es para poder hacer multiples corridas desde otro script llamando a este, con el número de corrida, para dejar los resultados en carpetas distintas numeradas

sys.path.insert(0, "ultralytics")

from ultralytics import YOLO
logging.basicConfig(level=logging.INFO)

runYolov5 = True
runYolov6 = True
runYolov7 = True
runYolov8 = True
runOpenCv = False

source_path = "./YOLOv6/data/images"
source_path = "./tpProf_cars_dataset/images"
project_name = "./runs/detect"

################################
############ YOLOV5 ############
################################


yolov5Params = "python3 ./yolov5/detect.py --source " + source_path + " --save-conf --save-txt --agnostic-nms --classes 2 3 5 7 --project " + project_name + " --weights ./yolov5/"
yolov5Models = ["yolov5n.pt", "yolov5s.pt"]#, "yolov5m.pt"]#, "yolov5l.pt"]#, "yolov5x.pt"] #Descartamos estos modelos porque tiraban problema de memoria en al Jetson y relantizaban el procesamiento (a veces crasheaba directamente)
name = " --name "

# * run yolov5 models
if runYolov5:
    for model in yolov5Models:
        modelParams = yolov5Params + model + name + model.replace(".pt","") + "_" + str(i)

        logging.info("#### Running: %s ####", model)
        os.system(modelParams)
        logging.info("#### Finished Running: %s ####", model)


        
################################
############ YOLOV6 ############
################################
       
yolov6Params = "python3 ./YOLOv6/tools/infer.py --source "+source_path+" --conf-thres 0.25 --save-txt --agnostic-nms --classes 2 3 5 7 --project " + project_name + " --weights ./YOLOv6/"
yolov6Models = ["yolov6n.pt"]#, "yolov6s.pt", "yolov6m.pt"]#, "yolov6l.pt"]#, "yolov6x.pt"] #Descartamos estos modelos porque tiraban problema de memoria en al Jetson y relantizaban el procesamiento (a veces crasheaba directamente)
name = " --name "

# * run yolov6 models
if runYolov6:
    for model in yolov6Models:
        modelParams = yolov6Params + model + name + model.replace(".pt","") + "_" + str(i)

        logging.info("#### Running: %s ####", model)
        os.system(modelParams)
        logging.info("#### Finished Running: %s ####", model)

################################
############ YOLOV7 ############
################################

yolov7Params = "python3 ./yolov7/detect.py --source "+source_path+" --conf 0.25 --img-size 640 --save-conf --save-txt --agnostic-nms --class 2 3 5 7 --project " + project_name + " --weights ./yolov7/"
yolov7Models = ["yolov7-tiny.pt"]#, "yolov7.pt"]#, "yolov7x.pt"] #Descartamos estos modelos porque tiraban problema de memoria en al Jetson y relantizaban el procesamiento (a veces crasheaba directamente)

# * run yolov7 models
if runYolov7:
    for model in yolov7Models:
        modelParams = yolov7Params + model + name + model.replace(".pt","") + "_" + str(i)

        logging.info("#### Running: %s ####", model)
        os.system(modelParams)
        logging.info("#### Finished Running: %s ####", model)

################################
############ YOLOV8 ############
################################

yolov8Models = ["yolov8n.pt"]#, "yolov8s.pt", "yolov8m.pt"]#, "yolov8l.pt"]#, "yolov8x.pt"] #Descartamos estos modelos porque tiraban problema de memoria en al Jetson y relantizaban el procesamiento (a veces crasheaba directamente)

# * run yolov8 models
if runYolov8:
    for modelName in yolov8Models:
        model = YOLO("./ultralytics/" + modelName)
        modelName = modelName.replace(".pt","") + "_" + str(i)
        results = model.predict(
            source=source_path,
            save=True,
            save_txt=True,
            save_conf=True,
            agnostic_nms=True,
            classes=[2, 3, 5, 7],
            verbose=True,
            project = project_name,
            name = modelName
        )

################################
############ OPENCV ############
################################

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
openCvParams = "python3 detection_openCV.py --dataSetPath "
openCvDataSetPath = ".\\tpProf_cars_dataset"
openCvDataSetPath = ".\\YOLOv6\\data\\images"
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
