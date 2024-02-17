import tensorflow as tf
import keras
from keras import layers
from tensorflow import nn
import pandas as pd
from pathlib import Path
import numpy as np
import ast
import time

keras.mixed_precision.set_global_policy(keras.mixed_precision.Policy('mixed_float16'))
batchSize = 256
parentDirectory = Path(__file__).parent
labelList = {"Opang": 0,
             "Jai": 1,
             "Max": 2,
             "Achi": 3,
             "PP": 4}

def preprocessData(dataset):
    label = tf.convert_to_tensor(dataset["Label"].values, dtype=tf.int8)
    stringData = dataset.drop(["Label"], axis=1)
    columnNames = [f'{i}' for i in range(len(stringData.columns))]
    data = pd.DataFrame(columns=columnNames)
    for row in stringData.values: #pandas decided to convert all matrix to string when save the dataset as xlsx
        rowData = [0] * len(stringData.columns)
        i = 0
        for matrix in row:
            rowData[i] = ast.literal_eval(matrix) #turn string into matrix
            i += 1
        data.loc[len(data)] = rowData
    #convert dataframe to ndarray
    #convert the 3D ndarray into 2D ndarray or converting [[[x,y,z],[x,y,z],...],[[x,y,z],[x,y,z],...],...] to [[x,y,z],[x,y,z],[x,y,z],...] with .flatten() function
    #convert the 2D ndarray into 1D ndarray or converting [[x,y,z],[x,y,z],[x,y,z],...] to [x,y,z,x,y,z,x,y,z,...] with np.concatenate() function
    data = np.concatenate(data.to_numpy().flatten())
    data = data.reshape((-1, len(stringData.columns)*3, 1)) #sperate ndarray into multiple one corresponding to its label
    data = tf.convert_to_tensor(data, dtype=tf.float16)
    return data, label

#Preparing dataset
dataset = pd.read_csv(parentDirectory.joinpath("Dataset.csv"))
loadStart = time.perf_counter()
data, label = preprocessData(dataset)
loadFinish = time.perf_counter()
#seed = random.randint(0, 1000) #Set seed to make sure the data and label will be shuffle in the same order
#data = tf.random.shuffle(data, seed=seed)
#label = tf.random.shuffle(label, seed=seed)
print(data)
print(label)
print(f"Dataset load time: {loadFinish - loadStart}")
del dataset 

model = keras.models.Sequential([
    layers.Flatten(input_shape=(1434, 1)),
    layers.Dense(256, activation=nn.relu),
    layers.Dense(512, activation=nn.relu),
    layers.Dense(256, activation=nn.relu),
    layers.Dense(128, activation=nn.leaky_relu),
    layers.Dropout(0.3),
    layers.Dense(len(labelList), activation=nn.softmax) 
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, label, epochs=50, batch_size=batchSize)
model.evaluate(data, label, batch_size=batchSize)
model.save(str(parentDirectory.joinpath("Face recognition model")))