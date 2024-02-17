import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
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
model = modelDirectory.joinpath("face_landmarker.task")
#list for converting label into a number
labelList = {"Opang": 0,
             "Jai": 1,
             "Max": 2,
             "Achi": 3,
             "PP": 4}
#Config
removeUnusableImage = True
minConfidence = 0.5
BaseOptions = mp.tasks.BaseOptions
faceLandmarker = vision.FaceLandmarker
faceLandmarkerOption = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

#create the landmarker object
options = faceLandmarkerOption(base_options=BaseOptions(model_asset_path=model),
                               running_mode=VisionRunningMode.IMAGE,
                               min_face_detection_confidence=minConfidence)
with faceLandmarker.create_from_options(options) as landmarker:
    #initiate dataframe
    columns = ["Label"]
    for i in range(478):
        columns.append(str(i))
    df = pd.DataFrame(columns=columns)

    def toDataFrame(imagePATH, label): #convert image path to be added to dataframe
        image = mp.Image.create_from_file(str(imagePATH))
        landmarkResult = landmarker.detect(image)
        landmarkCoordinates = landmarkResult.face_landmarks

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
            if removeUnusableImage:
                os.remove(imagePATH)

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