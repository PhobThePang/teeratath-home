import tensorflow as tf
import keras
from pathlib import Path
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import math

parentDirectory = Path(__file__).parent
minConfidence = 0.5 #Use for both landmarker and gesture recognition model
#Gesture recognition model config
gestureRecognition = keras.models.load_model(parentDirectory.joinpath("Gesture recognition model"))
keras.mixed_precision.set_global_policy(keras.mixed_precision.Policy('mixed_float16'))
labelList = {0: "Pointing",
             1: "Uncatagorize"}

#Hand landmark model config
handLandmarkModel = parentDirectory.joinpath("hand_landmarker.task")
BaseOptions = mp.tasks.BaseOptions
handLandmarker = vision.HandLandmarker
handLandmarkerOption = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode
#create the landmarker object
options = handLandmarkerOption(base_options=BaseOptions(model_asset_path=handLandmarkModel),
                               running_mode=VisionRunningMode.IMAGE,
                               min_hand_detection_confidence=minConfidence,
                               num_hands=1)
with handLandmarker.create_from_options(options) as landmarker:
    def imageToMatrix(imagePATH):
        image = mp.Image.create_from_file(str(imagePATH))
        landmarkResult = landmarker.detect(image)
        landmarkCoordinates = landmarkResult.hand_landmarks

        if len(landmarkCoordinates) > 0 and len(landmarkCoordinates[0]) > 0:
            matrix = [[0, 0, 0]] * 21
            i = 0
            #Add landmarks to M A T R I X
            for landmark in landmarkCoordinates[0]:
                matrix[i] = [landmark.x, landmark.y, landmark.z]
                i += 1
            return matrix
        else:
            return None

    def predictMatrix(matrix):
        #Preprocess matrix and get the prediction
        matrix = np.array(matrix).flatten()
        matrix = matrix.reshape((-1, 67*3, 1))
        matrix = tf.convert_to_tensor(matrix, dtype=tf.float16)

        prediction = gestureRecognition.predict(matrix, verbose=3)
        if labelList[np.argmax(prediction)] == "Pointing" and max(prediction[0]) * 100 > minConfidence:
            return matrix[5], matrix[8] #return position of base of the index finger and the tip of index finger
        else:
            return None

    def getDegree(startPoint, endPoint):
        ax, ay, az = startPoint[0], startPoint[1], startPoint[2]
        bx, by, bz = endPoint[0], endPoint[1], endPoint[2]
        
        #Get length of each axis
        xDist = abs(bx-ax)
        yDist = abs(by-ay)
        zDist = abs(bz-az)
        print(f"x dist:{xDist}, y dist:{yDist}, z dist:{zDist}")
    
        #Pythgorean moment
        upEuclidean = math.hypot(xDist, yDist)
        floorEuclidean = math.hypot(xDist, zDist)
        print(f"up Euclidean:{upEuclidean}, floor Euclidean:{floorEuclidean}")
    
        #Use law of cosine to get radiant then convert the radiant to degree
        upAngle = math.degrees(math.acos((xDist**2 + upEuclidean**2 - yDist**2) / (2 * xDist * upEuclidean)))
        floorAngle = math.degrees(math.acos((xDist**2 + floorEuclidean**2 - zDist**2) / (2 * xDist * floorEuclidean)))
        
        return upAngle, floorAngle
    
    imagePath = ""
    imageMatrix = imageToMatrix(imagePath)
    if imageMatrix != None:
        basePosition, tipPosition = predictMatrix(imageMatrix)
        upAngle, floorAngle = getDegree(basePosition, tipPosition)
    print(f"Up angle: {upAngle}, Floor angle: {floorAngle}")