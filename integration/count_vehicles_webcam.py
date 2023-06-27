import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import csv
import cv2
import numpy as np
import os
import sys
import time
import torch

sys.path.insert(0, "tracking")
sys.path.insert(0, "sort")
sys.path.insert(0, "videos")
sys.path.insert(0, "ultralytics")

from ultralytics import YOLO
from sort import *
from tracking_utils import box, boxXyxy, divideChunks, DrawLaneCoordinates, Lane, VirtualLoop

CSV_CONFIGURATION_PATH = "./configuration/lanes.csv"

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
    show_plot_live
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

    # configure lanes
    vLoops = []
    colors = [
        (255, 63, 0),
        (255, 123, 0),
        (255, 221, 0),
        (191, 255, 0),
        (97, 255, 0),
        (0, 255, 16),
    ]
    xData = []
    yData = []

    # * initialize the lanes if there's a configuration file (lanes.csv)
    if os.path.exists(CSV_CONFIGURATION_PATH):
        with open(CSV_CONFIGURATION_PATH, "r") as csvFile:
            csvLanes = csv.reader(csvFile)
            csvLanes = list(csvLanes)

            print("Initializing lanes with lanes.csv")

            if number_of_lanes == -1:
                number_of_lanes = len(csvLanes)

            for vLoopNumber, csvLane in enumerate(csvLanes, 1):
                # converts the format from the csv
                # [x1, y1, x2, y2, x3, y3, x4, y4] to
                # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                # which is the format expected by Lane class
                formattedLane = list(divideChunks([int(l) for l in csvLane], 2))
                if vLoopNumber <= number_of_lanes:
                    vLoops.append(
                        VirtualLoop(
                            coordinates=formattedLane,
                            virtualLoopId=vLoopNumber,
                            color=colors[vLoopNumber - 1],
                            thickness=2,
                        )
                    )
                    xData.append(np.empty((0,), dtype=np.float64))
                    yData.append(np.empty((0,), dtype=bool))
                else:
                    break

            csvFile.close()

    # webcam stream
    #cam = cv2.VideoCapture(1)  # ! use this this for OBS virtualCam
    #cam = cv2.VideoCapture("udp://192.168.0.11:9999?overrun_nonfatal=1&fifo_size=999999999999")
    cam = cv2.VideoCapture("videos/normal_fps/puente_BA_centro_cerca_1/puente_BA_centro_cerca_1.mp4")
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    #cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if save_video:
        print("--SAVE VIDEO--")
        ret, frame = cam.read()
        video_shape = list(frame.shape)[:-1] # video_shape = (height, width)
        video_shape.reverse()
        video = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            video_shape,
        )

    frameNumber = 0
    videoFpsAvg = 0
    key = cv2.waitKey(1)

    # Create an empty plot with number_of_lanes lines
    if show_plot_live:
        print("--SHOW PLOT LIVE--")
        matplotlib.use("TkAgg")
        plt.ion()
        fig, axs = plt.subplots(number_of_lanes)
        for idx, ax in enumerate(axs, 1):
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Lane " + str(idx))
        fig.canvas.draw()
        plt.show(block=False)

        # the size of the window will detrmine how many points we see in the plot
        plotWindowSize = 50

    while True:
        ret, frame = cam.read()

        if not ret:
            print("Can't receive frames from stream.")
            break

        frameTime = time.time()

        # press "l" to draw lanes
        if key == ord("l"):
            vLoops = []
            xData = []
            yData = []
            for i in range(number_of_lanes):
                # delete the old configuration file at start
                if i == 0 and os.path.exists(CSV_CONFIGURATION_PATH):
                    os.remove(CSV_CONFIGURATION_PATH)

                vLoopNumber = i + 1
                laneCoordinates = DrawLaneCoordinates(
                    frame, color=colors[i], thickness=2
                )
                vLoops.append(
                    VirtualLoop(
                        coordinates=laneCoordinates.coordinates,
                        virtualLoopId=vLoopNumber,
                        color=colors[i],
                        thickness=2,
                    )
                )
                xData.append(np.empty((0,), dtype=np.float64))
                yData.append(np.empty((0,), dtype=bool))
                # save the lane configuration in a csv file
                with open(CSV_CONFIGURATION_PATH, "a", newline="") as csvFile:
                    csvWriter = csv.writer(csvFile)
                    # flatten [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    # to [x1, y1, x2, y2, x3, y3, x4, y4]
                    # and save that to the csv file
                    csvWriter.writerow(
                        [
                            coordinate
                            for point in laneCoordinates.coordinates
                            for coordinate in point
                        ]
                    )

            csvFile.close()

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
        for virtualLoop in vLoops:
            # check if there is any vehicle in the lane
            actualIsOccupied = False
            for trackedVehicle in trackedVehicles:
                # calculate center points
                xCenter = (trackedVehicle[0] + trackedVehicle[2]) / 2
                yCenter = (trackedVehicle[1] + trackedVehicle[3]) / 2

                if (
                    cv2.pointPolygonTest(
                        contour=np.asarray(virtualLoop.getCoordinates()),
                        pt=(xCenter, yCenter),
                        measureDist=False,
                    )
                    == 1
                ):
                    # lane is occupied
                    actualIsOccupied = True
                    actualDetectedId = trackedVehicle[-1] # the last column contains the ID of the SORT tracking
                    lastDetectedId = virtualLoop.getLastDetectedId()
                    lastIsOccupied = virtualLoop.getLastIsOccupied()
                    if lastDetectedId == actualDetectedId and lastIsOccupied == False:
                        virtualLoop.correctBackwards(actualDetectedId)
                    virtualLoop.updateIsOccupied(True, actualDetectedId, frameTime)
                    lastIsOccupied = True
                    break
            if actualIsOccupied == False:
                virtualLoop.updateIsOccupied(False, -1, frameTime)

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

        if vLoops[0].lane.buffer.isFull():
                for idx, virtualLoop in enumerate(vLoops):
                    # dequeue values from the buffer
                    auxValue = virtualLoop.popFirstValue()

                    if show_plot_live:
                        xData[idx] = np.append(
                            xData[idx][-plotWindowSize + 1 :], auxValue["timeStamp"]
                        )
                        yData[idx] = np.append(
                            yData[idx][-plotWindowSize + 1 :],
                            auxValue["isOccupied"],
                        )
                        axs[idx].cla()
                        axs[idx].plot(
                            xData[idx][-plotWindowSize:],
                            yData[idx][-plotWindowSize:],
                            label=virtualLoop.getVirtualLoopId(),
                        )
                        axs[idx].set_xlabel("Time (s)")
                        axs[idx].set_ylabel("Lane " + str(virtualLoop.getVirtualLoopId()))

                if show_plot_live:
                    plt.pause(0.00001)

        # * display results and save video
        if display or save_video:
            # plot lanes
            for virtualLoop in vLoops:
                cv2.polylines(
                    frame,
                    [np.asarray(virtualLoop.lane.coordinates)],
                    isClosed=True,
                    color=virtualLoop.lane.color,
                    thickness=virtualLoop.lane.thickness,
                )
                cv2.putText(
                    img=frame,
                    text="Lane {}: {}".format(
                        virtualLoop.getVirtualLoopId(), virtualLoop.getVehicleListCount()
                    ),
                    org=(30, virtualLoop.getVirtualLoopId() * 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=virtualLoop.lane.color,
                    thickness=virtualLoop.lane.thickness,
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
                    text="FPS instant:  {:.2f}".format(round(videoFPS, 2)),
                    org=(640 - 200, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1 * 0.6,
                    color=(0, 171, 255),
                    thickness=2,
                )
                cv2.putText(
                    img=frame,
                    text="FPS average: {:.2f}".format(round(videoFpsAvg, 2)),
                    org=(640 - 200, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1 * 0.6,
                    color=(0, 171, 255),
                    thickness=2,
                )
                cv2.imshow("detections", frame)
                key = cv2.waitKey(1)
