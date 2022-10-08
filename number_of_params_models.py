import numpy as np
import matplotlib.pyplot as plt
from numerize import numerize
import pandas as pd

# yolov5
yolov5 = ["YOLOv5n", "YOLOv5s", "YOLOv5m", "YOLOv5l", "YOLOv5x"]
paramsYolov5 = [1.9e6, 7.2e6, 21.2e6, 46.5e6, 86.7e6]
yolov5Df = pd.DataFrame(
    {"models": yolov5, "numberOfParams": paramsYolov5, "family": "yolov5"}
)

# yolov7
yolov7 = ["YOLOv7", "YOLOv7-X"]
paramsYolov7 = [36.9e6, 71.3e6]
yolov7Df = pd.DataFrame(
    {"models": yolov7, "numberOfParams": paramsYolov7, "family": "yolov7"}
)

# yoloX
yoloX = ["YOLOX-s", "YOLOX-m", "YOLOX-l", "YOLOX-x", "YOLOX-Nano", "YOLOX-Tiny"]
paramsYoloX = [9.0e6, 25.3e6, 54.2e6, 99.1e6, 0.91e6, 5.06e6]
yoloXDf = pd.DataFrame(
    {"models": yoloX, "numberOfParams": paramsYoloX, "family": "yoloX"}
)

df = pd.concat([yolov5Df, yolov7Df, yoloXDf])
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
