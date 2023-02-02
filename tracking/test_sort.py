import os
from tracking_utils import detections2Sort, getImgShape, tracking2Video

"""
Convierte los archivos de salida del algoritmo de deteccion (predictions) en un archivo
con el formato esperado por el algoritmo de tracking SORT. Lo hace para cada carpeta con
imagenes de prueba que tenga el formato adecuado:
folderWithCorrectFormat
          |
          -> images
          |
          -> predictions
El path inicial para correrlo es C:/PV/GitFiles/tpProf
"""
basePath = "./videos/normal_fps/"
# basePath = "/content/tpProf/videos/5fps/"
sortTrainPath = "./sort/data/train/"
sortFormatName = "det.txt"
# for dir in sorted(os.listdir(basePath)):
for dir in ["miDataMulti"]:
    dirPath = os.path.join(basePath, dir)
    predictionsPath = os.path.join(dirPath, "predictions")
    imagesPath = os.path.join(dirPath, "images")
    if os.path.exists(os.path.join(dirPath, "images")):
        imgShape = getImgShape(imagesPath)
        # convert the output files of the prediction algorithm into the input format file for SORT algorithm
        sortFormatPath = os.path.join(sortTrainPath, dir, "det")
        detections2Sort(predictionsPath, sortFormatPath, sortFormatName, imgShape)
    else:
        print("Couldn't find path: ", imagesPath)

# ejecuta el algoritmo de tracking SORT
os.system("python ../sort/sort.py")

"""
Crea un video para cada carpeta a partir de las imagenes de prueba y del archivo de tracking
obtenido luego de aplicar el algoritmo detracking SORT.
"""
basePath = "./tpProf/videos/normal_fps/"
# basePath = "/content/tpProf/videos/5fps/"
outputVideoBasePath = "./trackingVideos/"
sortOutputBasePath = "./sort/output/"
# for dir in sorted(os.listdir(basePath)):
for dir in ["miDataMulti"]:
    dirPath = os.path.join(basePath, dir)
    imagesPath = os.path.join(dirPath, "images")
    if os.path.exists(os.path.join(dirPath, "images")):
        outputVideo = os.path.join(outputVideoBasePath, dir + ".avi")
        fps = (
            5 if "5fps" in dir else 25
        )  # no todos los videos tienen 25 fps, pero es un buen valor de referencia
        detectionsPath = os.path.join(sortOutputBasePath, dir + ".txt")
        print("fps: ", fps)
        print("Creating video: ", outputVideo)
        tracking2Video(outputVideo, fps, imagesPath, detectionsPath)
    else:
        print("Couldn't find path: ", imagesPath)
