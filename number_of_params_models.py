import numpy as np
import matplotlib.pyplot as plt
from numerize import numerize
import pandas as pd

# yoloX
yoloX = ["YOLOX-s", "YOLOX-m", "YOLOX-l", "YOLOX-x", "YOLOX-Nano", "YOLOX-Tiny"]
paramsYoloX = [9.0e6, 25.3e6, 54.2e6, 99.1e6, 0.91e6, 5.06e6]
yoloXDf = pd.DataFrame(
    {"models": yoloX, "numberOfParams": paramsYoloX, "family": "yoloX"}
)

# yolov5
yolov5 = ["YOLOv5n", "YOLOv5s", "YOLOv5m", "YOLOv5l", "YOLOv5x"]
paramsYolov5 = [1.9e6, 7.2e6, 21.2e6, 46.5e6, 86.7e6]
yolov5Df = pd.DataFrame(
    {"models": yolov5, "numberOfParams": paramsYolov5, "family": "yolov5"}
)

# yolov6
yolov6 = ["YOLOv6n", "YOLOv6s", "YOLOv6m", "YOLOv6l"]
paramsYolov6 = [4.7e6, 18.5e6, 34.9e6, 59.6e6]
yolov6Df = pd.DataFrame(
    {"models": yolov6, "numberOfParams": paramsYolov6, "family": "yolov6"}
)

# yolov7
yolov7 = ["YOLOv7", "YOLOv7x"]
paramsYolov7 = [36.9e6, 71.3e6]
yolov7Df = pd.DataFrame(
    {"models": yolov7, "numberOfParams": paramsYolov7, "family": "yolov7"}
)

# yolov8
yolov8 = ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"]
paramsYolov8 = [3.2e6, 11.2e6, 25.9e6, 43.7e6, 68.2e6]
yolov8Df = pd.DataFrame(
    {"models": yolov8, "numberOfParams": paramsYolov8, "family": "yolov8"}
)

df = pd.concat([yoloXDf, yolov5Df, yolov6Df, yolov7Df, yolov8Df])
print(df)

fig, ax = plt.subplots()
sortedDf = df.sort_values(by="numberOfParams")
ax.barh(
    sortedDf["models"], sortedDf["numberOfParams"] / 1e6
)  # scaled to be shown in millions

# add grid
ax.grid(visible=True, color="grey", linestyle="-.", linewidth=0.5, alpha=0.2)

# add annotation to bars
for i in ax.patches:
    plt.text(
        i.get_width() + 0.5e6,
        i.get_y() + 0.3125,
        str(numerize.numerize(i.get_width())),
        fontsize=10,
        fontweight="bold",
        color="grey",
    )

# plot title
ax.set_title("# Parameters vs Models")

# set labels
ax.set_xlabel("# Parameters [M]")
ax.set_ylabel("Models")

# show plot
plt.show()
