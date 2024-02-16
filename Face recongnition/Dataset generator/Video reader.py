import cv2
from pathlib import Path
import time
import multiprocessing
import os
import sys

parentDirectory = Path(__file__).parent
inputDirectory = parentDirectory.joinpath("Video")
outputDirectory = parentDirectory.joinpath("Images")
supportsExtension = ["*/*.mp4", "*/*.mov"]
processes = []
processesCount = 1

def getFilePATHS(directory):
    videoPATHs = []
    for extension in supportsExtension: #Collect file that has mp4 and mov file extension
        for file in directory.glob(extension):
            videoPATHs.append(file)
    print(f"total files: {len(videoPATHs)}")
    return videoPATHs

def splitList(list):
    step = len(list) // processesCount
    remain = len(list) % processesCount
    for i in range(0, len(list), step):
        yield list[i:i + step + remain] #multiple return 1D list
        remain = 0

def save_frames(videoPATHs):
    try:    
        for videoPATH in videoPATHs:
            fileName = videoPATH.name.split(".")[0]
            cap = cv2.VideoCapture(str(videoPATH))
            success, frame = cap.read()
            frame_number = 0
            labelDirectory = outputDirectory.joinpath(videoPATH.parent.name)
            labelDirectory.mkdir(parents=True, exist_ok=True) #Create directory if not exist

            while success:
                image_path = str(labelDirectory.joinpath(f"{fileName}_{str(frame_number)}.png"))
                cv2.imwrite(image_path, frame)
                del frame
                del success
                success, frame = cap.read()
                frame_number += 1
            cap.release()
            del frame_number
            del cap
    except:
        pass

startTime = time.perf_counter()

videoPATHs = getFilePATHS(inputDirectory)
videoPATHsList = list(splitList(videoPATHs))
del videoPATHs #hopefully it will freeup some memory

for i in range(processesCount): #initiate process
    p = multiprocessing.Process(target=save_frames, args=[videoPATHsList[i]])
    p.start()
    processes.append(p)
for process in processes: #wait for process to end
    process.join()

del videoPATHsList
del processes

finishTime = time.perf_counter()
print(f"total time: {finishTime - startTime}")
