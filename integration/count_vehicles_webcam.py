from matplotlib import pyplot as plt
from ultralytics import YOLO
import cv2
import matplotlib
import numpy as np
import os
import sys
import time
import torch

sys.path.insert(0, "tracking")
sys.path.insert(0, "sort")
sys.path.insert(0, "videos")

from sort import *
from tracking_utils import box, boxXyxy, DrawLaneCoordinates, Lane


def count_vehicles_webcam(
    display,
    plot_dets,
    plot_tracks,
    save_video,
    output_video_path,
    showLogTimes,
    is_yolov8,
    iou_threshold_tracking,
    max_age,
    min_hits,
    number_of_lanes,
    fps,
):
    # * load  detection model and configuration
    if is_yolov8:
        model = YOLO("yolov8n.pt")
    else:
        model = torch.hub.load("ultralytics/yolov5", "yolov5n", _verbose=False)
        model.agnostic = (
            True  # NMS class-agnostic (returns only one class for each detection)
        )
        model.classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    # * initialize tracking algorithm instance
    tracker = Sort(
        max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold_tracking
    )

    # load images/video/webcam
    # testImg = "videos/5fps/puente_BA_centro_lejos_1_5fps/images/puente_BA_centro_lejos_1_5fps_0.jpg"
    # testFolder = "videos/5fps/puente_BA_centro_lejos_1_5fps/images"

    # ! fix code to save video
    # if save_video:
    #     video_shape = list(cv2.imread(testImg).shape)[:-1] # video_shape = (height, width)
    #     video_shape.reverse()
    #     video = cv2.VideoWriter(
    #         output_video_path,
    #         cv2.VideoWriter_fourcc(*"mp4v"),
    #         fps,
    #         video_shape,
    #     )

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

    # webcam stream
    cam = cv2.VideoCapture("udp://192.168.0.7:9999")
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frameNumber = 0
    videoFpsAvg = 0
    key = cv2.waitKey(1)
    while True:
        ret, frame = cam.read()

        if not ret:
            print("Can't receive frames from stream.")
            break

        frameTime = time.time()

        # press "l" to draw lanes
        if key == ord("l"):
            lanes = []
            for i in range(number_of_lanes):
                laneNumber = i + 1
                laneCoordinates = DrawLaneCoordinates(
                    frame, color=colors[i], thickness=2
                )
                lanes.append(
                    Lane(
                        coordinates=laneCoordinates.coordinates,
                        number=laneNumber,
                        color=colors[i],
                        thickness=2,
                    )
                )

        t0 = time.time()
        # * run detection algorithm
        if is_yolov8:
            detections = model.predict(
                source=frame,
                agnostic_nms=True,
                classes=[2, 3, 5, 7],
            )[0].boxes.data

        else:
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
        for lane in lanes:
            # check if there is any vehicle in the lane
            lane.setIsOccupiedNow(False, -1)

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
                    lane.updateVehicleList(
                        trackedVehicle[-1]
                    )  # add vehicle ID to the list
                    lane.setIsOccupiedNow(True, trackedVehicle[-1])

            lane.updateOutputSignal(frameTime)

        # * log times:
        if showLogTimes:
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
            if plot_dets:
                for detection in detections:
                    boxXyxy(
                        frame,
                        x1=detection[0],
                        y1=detection[1],
                        x2=detection[2],
                        y2=detection[3],
                        color=(255, 0, 0),
                        thickness=1,
                        printXY=False,
                    )
            if plot_tracks:
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

            if frameNumber < 50:
                frameNumber = frameNumber + 1

            if save_video:
                video.write(frame)

            if display:
                videoFPS = 1 / (time.time() - frameTime)
                videoFpsAvg = videoFpsAvg * (frameNumber - 1) / (
                    frameNumber
                ) + videoFPS / (frameNumber)
                cv2.putText(
                    img=frame,
                    text="FPS intant: {}".format(round(videoFPS, 2)),
                    org=(640 - 10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 171, 255),
                    thickness=2,
                )
                cv2.putText(
                    img=frame,
                    text="FPS average: {}".format(round(videoFpsAvg, 2)),
                    org=(640 - 10, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 171, 255),
                    thickness=2,
                )
                cv2.imshow("detections", frame)
                key = cv2.waitKey(1)


# matplotlib.use("TkAgg")
# fig, axs = plt.subplots(number_of_lanes)
# for idx, lane in enumerate(lanes):
#     axs[idx].plot(
#         np.array(lane.getOutputSignal())[:, 0],
#         np.array(lane.getOutputSignal())[:, 1],
#         label=lane.number,
#     )
#     axs[idx].set_title("Lane {}".format(lane.number))
# plt.show()
