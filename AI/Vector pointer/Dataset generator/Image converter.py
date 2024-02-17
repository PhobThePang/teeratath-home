import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import os
import time

#PATH variable
parentPath = Path(__file__).parent
print("Parent directory: " + str(parentPath))
inputDirectory = parentPath.joinpath("Images")
outputFile = parentPath.joinpath("Dataset" + ".csv")
modelDirectory = parentPath.joinpath("Model")
model = modelDirectory.joinpath("hand_landmarker.task")
columnNameList = ["wrist", "thumb cmc", "thumb mcp", "thumb ip", "thumb tip",
                  "index finger mcp", "index finger pip", "index finger dip", "index finger tip", "middle finger mcp",
                  "middle finger pip", "middle finger dip", "middle finger tip", "ring finger mcp", "ring finger pip",
                  "ring finger dip", "ring finger tip", "pinky mcp", "pinky pip", "pinky dip",
                  "pinky tip"]
#list for converting label into a number
labelList = {"Pointing": 0,
             "Uncatagorize": 1}
#Config
removeUnusableImage = True
minConfidence = 0.5
BaseOptions = mp.tasks.BaseOptions
handLandmarker = vision.HandLandmarker
handLandmarkerOption = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

#create the landmarker object
options = handLandmarkerOption(base_options=BaseOptions(model_asset_path=model),
                               running_mode=VisionRunningMode.IMAGE,
                               min_hand_detection_confidence=minConfidence,
                               num_hands=1)
with handLandmarker.create_from_options(options) as landmarker:
    #initiate dataframe
    columns = ["Label"]
    for name in columnNameList:
        columns.append(f"{name}")
    #for i in ["left", "right"]:
    #    for name in columnNameList:
    #        columns.append(f"{i} {name}")
    df = pd.DataFrame(columns=columns)

    def toDataFrame(imagePATH, label): #convert image path to be added to dataframe
        image = mp.Image.create_from_file(str(imagePATH))
        landmarkResult = landmarker.detect(image)
        landmarkCoordinates = landmarkResult.hand_landmarks

            #Check for empty list           Check for empty coordinate list ([[coordinat1, coordinat...]])
        if len(landmarkCoordinates) > 0 and len(landmarkCoordinates[0]) > 0:
            matrix = [[0, 0, 0]] * len(columns)
            matrix[0] = labelList[label]
            i = 1
            #Add landmarks to M A T R I X
            for landmark in landmarkCoordinates[0]:
                matrix[i] = [landmark.x, landmark.y, landmark.z]
                i += 1
            #Add M A T R I X to dataframe
            df.loc[len(df)] = matrix
        else:
            #if removeUnusableImage:
            #    os.remove(imagePATH)
            pass

    #"g o o d s t u f f"
    startTime = time.perf_counter()
    imageSubdirectory = inputDirectory.iterdir()
    for childDirectory in imageSubdirectory:
        if childDirectory.is_dir():
            for image in childDirectory.glob("**/*.*"): #reading all image in input directory
                label = childDirectory.name #saving directory name to use as index name
                toDataFrame(image, label)

    df = df.sample(frac=1) #Shuffle dataframe
    print("Output dataframe:")
    print(df)
    df.to_csv(outputFile, index=False)

    finishTime = time.perf_counter()
    print(f"total time: {finishTime - startTime}")