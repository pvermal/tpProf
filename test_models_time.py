import csv
import logging
from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
MODELS = os.path.join(ROOT, r"tpProf_cars_dataset")

modelTimes = {}
for model in os.listdir(MODELS):
    # ignore the folders containing images and labels
    if model not in ["images", "labels"]:
        logging.info("Getting time for model {}.".format(model))
        modelTimes.update({model: []})
        PREDICTIONS = os.path.join(MODELS, model, "predictions")

        for file in os.listdir(PREDICTIONS):
            FILE = os.path.join(PREDICTIONS, file)
            with open(FILE) as csvFilePredicted:
                logging.info("Getting time for file {}.".format(FILE))
                for row in csv.reader(csvFilePredicted, delimiter=" "):
                    # append the time to the list. Only the first value is needed, as all
                    # the rows have the same time value for an image
                    modelTimes[model].append(float(row[-1]))
                    break

# average time for each model
averageTimes = {}
print("Average processing time:")
for model, values in modelTimes.items():
    mean = np.mean(np.asarray(values))
    averageTimes.update({model: mean})
    print("{}: {} ms".format(model, mean))

# plot times for each model
plt.figure()
for model, times in modelTimes.items():
    plt.plot(times, label=model)

plt.title("Time taken for each image")
plt.xlabel("Image n°")
plt.ylabel("Time")
plt.legend()
plt.show()
