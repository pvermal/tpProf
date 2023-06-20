import csv
import logging
from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
#MODELS = os.path.join(ROOT, r"tpProf_cars_dataset")
MODELS = os.path.join(ROOT, r"runs/detect")

modelNamesRun = ["yolov5n", "yolov5s","yolov6n","yolov7-tiny","yolov8n"]

modelNamesRunMean = {}
modelNamesRunStd = {}

modelTimes = {}
for model in os.listdir(MODELS):
    # ignore the folders containing images and labels
    if model not in ["images", "labels"]:
        logging.info("Getting time for model {}.".format(model))
        modelTimes.update({model: []})
        #PREDICTIONS = os.path.join(MODELS, model, "predictions")
        PREDICTIONS = os.path.join(MODELS, model, "labels")
        
        for file in os.listdir(PREDICTIONS):
            FILE = os.path.join(PREDICTIONS, file)
            with open(FILE) as csvFilePredicted:
                #logging.info("Getting time for file {}.".format(FILE))
                for row in csv.reader(csvFilePredicted, delimiter=" "):
                    # append the time to the list. Only the first value is needed, as all
                    # the rows have the same time value for an image
                    modelTimes[model].append(float(row[-1]))
                    break

# average time for each run
averageTimes = {}
print("Average processing time:")
for model, values in modelTimes.items():
    mean = np.mean(np.asarray(values))
    #var = np.var(np.asarray(values))
    var = np.std(np.asarray(values))
    averageTimes.update({model: mean})
    print("{}: {} +- {} ms".format(model, mean, var))

# group by model name
for modelName in modelNamesRun:
    modelNamesRunMean.update({modelName: []})
    modelNamesRunStd.update({modelName: []})
    for model, values in modelTimes.items():
        if modelName in model: 
            modelNamesRunMean[modelName].append( np.mean( np.asarray(values) ) )
            modelNamesRunStd[modelName].append( np.std( np.asarray(values) ) )
            
# average time for each model
averageTimes = {}
print("Average processing time:")
for model, values in modelNamesRunMean.items():
    mean = np.mean(np.asarray(values))
    #var = np.var(np.asarray(values))
    var = np.std(np.asarray(values))
    averageTimes.update({model: mean})
    #print("{}: {} +- {} ms".format(model, mean, var))
    #print("{}: ({} +- {}) +- ({} +- {}) ms".format(model, mean_1, var_1,mean_2,var_2))
    
# std time for each model
stdTimes = {}
print("Std processing time:")
for model, values in modelNamesRunStd.items():
    mean = np.mean(np.asarray(values))
    #var = np.var(np.asarray(values))
    var = np.std(np.asarray(values))
    stdTimes.update({model: mean})
    for model_mean , values_mean in modelNamesRunMean.items():
        if model_mean in model:
            mean2 = np.mean(np.asarray(values_mean))
            var2 = np.std(np.asarray(values_mean))
    #print("{}: {} +- {} ms".format(model, mean, var))
    print("{}: ({:.3f} +- {:.3f}) +- ({:.3f} +- {:.3f}) ms".format(model, mean2, var2, mean, var))

# plot times for each model
plt.figure()
for model, times in modelNamesRunMean.items():
    plt.plot(times, label=model)
plt.title("Time taken for each run")
plt.xlabel("Run n°")
plt.ylabel("Time")
plt.legend()
