import cv2
import os


# * el nombre de la carpeta debe ser el mismo que el del archivo de video
folderName = "roadTraffic_p6_5fps"
# * normal / 5fps
frames = "5fps"

basePath = "C:\\PV\\FIUBA\\tpProf\\videos\\{}\\{}".format(frames, folderName)
videoPath = basePath + "\\{}.mp4".format(folderName)
imgPath = basePath + "\\images"

# Read the video from specified path
cam = cv2.VideoCapture(videoPath)

try:
    # creating a folder named images
    if not os.path.exists(imgPath):
        os.makedirs(imgPath)

# if not created then raise error
except OSError:
    print("Error: Creating directory of data")

# frame
currentframe = 0

while True:
    # reading from frame
    ret, frame = cam.read()

    if ret:
        # if video is still left continue creating images
        # name = folderName + "_" + str(currentframe) + ".jpg"
        name = imgPath + "\\" + folderName + "_" + str(currentframe) + ".jpg"
        print("Creating..." + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
