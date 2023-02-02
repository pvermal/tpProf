from matplotlib import pyplot as plt
import cv2
import matplotlib
import numpy as np
import os
import sys
import time
import torch

sys.path.insert(0, "C:/PV/GitFiles/tpProf/tracking")
sys.path.insert(0, "C:/PV/GitFiles/tpProf/sort")

from sort import *
from tracking_utils import box, boxXyxy, DrawLaneCoordinates, Lane

# args
display = True
iou_threshold = 0.3
isLogTimes = False
max_age = 1
min_hits = 3
number_of_lanes = 6
# video params
save_video = True
output_video = (
    "C:/PV/FIUBA/tpProf/videos/normal_fps/puente_BA_centro_lejos_1/count_vehicles.mp4"
)
fps = 30
# video_shape = (480, 854)  # (height, width)

# * load  detection model and configuration
model = torch.hub.load("ultralytics/yolov5", "yolov5n", _verbose=False)
model.agnostic = True  # NMS class-agnostic (returns only one class for each detection)
model.classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# * initialize tracking algorithm instance
tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

# load images/video/webcam
testImg = "C:/PV/FIUBA/tpProf/videos/normal_fps/puente_BA_centro_lejos_1/images/puente_BA_centro_lejos_1_0.jpg"
testFolder = "C:/PV/FIUBA/tpProf/videos/normal_fps/puente_BA_centro_lejos_1/images"

if save_video:
    video_shape = list(cv2.imread(testImg).shape)[:-1]
    video_shape.reverse()
    video = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        video_shape,
    )

# configure lanes
lanes = []
colors = [
    (255, 63, 0),
    (255, 123, 0),
    (255, 221, 0),
    (191, 255, 0),
    (97, 255, 0),
    (0, 255, 16),
]
for i in range(number_of_lanes):
    laneNumber = i + 1
    laneCoordinates = DrawLaneCoordinates(testImg, color=colors[i], thickness=2)
    lanes.append(
        Lane(
            coordinates=laneCoordinates.coordinates,
            number=laneNumber,
            color=colors[i],
            thickness=2,
        )
    )

for frameName in sorted(
    os.listdir(testFolder),
    key=lambda x: int(
        os.path.splitext(x)[0].split("_")[-1]
    ),  # works for images in the test format we have: name_i.jpg
):
    frame = cv2.imread(os.path.join(testFolder, frameName))
    frameTime = time.time()

    t0 = time.time()
    # * run detection algorithm
    detections = model(frame).xyxy[
        0
    ]  # return x1,y1,x2,y2 (the center point has to be calculated)
    t1 = time.time()

    t2 = time.time()
    # * update tracking algorithm
    t = 1 / fps
    trackedVehicles = tracker.update(detections, t)
    t3 = time.time()

    # * generate output signal
    ## for trackedVehicle in trackedVehicles:
    ## check is the detection is inside any of the lanes
    for lane in lanes:
        # check if there is any vehicle in the lane
        lane.setIsOccupiedNow(False, -1)
        ## for lane in lanes:
        for trackedVehicle in trackedVehicles:
            # calculate center points
            xCenter = (trackedVehicle[0] + trackedVehicle[2]) / 2
            yCenter = (trackedVehicle[1] + trackedVehicle[3]) / 2

            if (
                cv2.pointPolygonTest(
                    contour=np.asarray(lane.getCoordinates()),
                    pt=(xCenter, yCenter),
                    measureDist=False,
                )
                == 1
            ):
                # lane is occupied
                lane.updateVehicleList(trackedVehicle[-1])  # add vehicle ID to the list
                lane.setIsOccupiedNow(True, trackedVehicle[-1])

        lane.updateOutputSignal(frameTime)

    # * log times:
    if isLogTimes:
        # total time (t3 - t0)
        # detection time (t1 - t0)
        # tracking time (t3 - t2)
        print(
            "Total time: {} ms".format(round((t3 - t0) * 1000, 3)),
            "Detection time: {} ms".format(round((t1 - t0) * 1000, 3)),
            "Tracking time: {} ms".format(round((t3 - t2) * 1000, 3)),
        )

    # * display results and save video
    if display or save_video:
        # plot lanes
        for lane in lanes:
            cv2.polylines(
                frame,
                [np.asarray(lane.coordinates)],
                isClosed=True,
                color=lane.color,
                thickness=lane.thickness,
            )
            cv2.putText(
                img=frame,
                text="Lane {}: {}".format(lane.getNumber(), lane.getVehicleCount()),
                org=(30, lane.number * 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=lane.color,
                thickness=lane.thickness,
            )
        for trackedVehicle in trackedVehicles:
            boxXyxy(
                frame,
                x1=trackedVehicle[0],
                y1=trackedVehicle[1],
                x2=trackedVehicle[2],
                y2=trackedVehicle[3],
                id=trackedVehicle[4],
                color=(0, 0, 255),
                thickness=1,
                printXY=False,
            )
            xCenter = (trackedVehicle[0] + trackedVehicle[2]) / 2
            yCenter = (trackedVehicle[1] + trackedVehicle[3]) / 2
            cv2.circle(
                img=frame,
                center=(int(xCenter), int(yCenter)),
                radius=4,
                color=(0, 0, 255),
                thickness=-1,
            )

        if save_video:
            video.write(frame)

        if display:
            cv2.imshow("detections", frame)
            cv2.waitKey(int(1000 / fps))

matplotlib.use("TkAgg")
fig, axs = plt.subplots(number_of_lanes)
for idx, lane in enumerate(lanes):
    axs[idx].plot(
        np.array(lane.getOutputSignal())[:, 0],
        np.array(lane.getOutputSignal())[:, 1],
        label=lane.number,
    )
    axs[idx].set_title("Lane {}".format(lane.number))
plt.show()
