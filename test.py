import utils.intersection_over_union as iou
import sys

sys.path.insert(0, "/tpProf/utils")


path = "./yolov5/yolov5-master/runs/detect/exp"
filePredicted = path + "/labels/zidane-predicted-labels.txt"
fileTrue = path + "/labels/zidane-true-labels-test.txt"
imgPath = path + "/zidane_copy.jpg"

iouScore, qtyPredicted, qtyTrue = iou.iouScore(filePredicted, fileTrue, imgPath)
print("################")
print("iou", iouScore)
print("qtyPredicted", qtyPredicted)
print("qtyTrue", qtyTrue)
print("################")
