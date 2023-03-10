import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def representation_initialization():
  # representation function
  # input - 32 most recent frames and actions
  # output - hidden state

  global representation_model
  if name == "atari":
    inputs = keras.Input(shape = (96,96,96)) # 3 * 32 = 96
    conv = layers.Conv2D(128, (3,3), 2, activation = "relu", padding = "same")
    x = conv(inputs)

    # 2 residual blocks
    kernel_size = (3,3)
    filters = 128
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    x = layers.Conv2D(256, (3,3), 2, activation = "relu", padding = "same")(x)

    # 3 residual blocks
    filters = 256
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    x = layers.AveragePooling2D((2,2), 2)(x)

    # 3 residual blocks
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    outputs = layers.AveragePooling2D((2,2), 2)(x)

  elif name == "chess":
   shape_input = (8,8,2000) # 20 * 100 = 2000
   inputs = keras.Input(shape = shape_input)
   conv = layers.Conv2D(1500, (3,3), activation = "relu", padding = "same")
   x = conv(inputs)
   x = layers.Conv2D(1000, (3,3), activation = "relu", padding = "same")
   x = layers.Conv2D(256, (3,3), activation = "relu", padding = "same")

   # 16 residual blocks
   filters = 256
   kernel_size = (3,3)

   # 1
   fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
   fx = layers.BatchNormalization()(fx)
   fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
   out = layers.Add()([x,fx])
   out = layers.ReLU()(out)
   x = layers.BatchNormalization()(out)

   # 2
   fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
   fx = layers.BatchNormalization()(fx)
   fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
   out = layers.Add()([x,fx])
   out = layers.ReLU()(out)
   x = layers.BatchNormalization()(out)

   # 3
   fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
   fx = layers.BatchNormalization()(fx)
   fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
   out = layers.Add()([x,fx])
   out = layers.ReLU()(out)
   x = layers.BatchNormalization()(out)

   # 4
   fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
   fx = layers.BatchNormalization()(fx)
   fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
   out = layers.Add()([x,fx])
   out = layers.ReLU()(out)
   x = layers.BatchNormalization()(out)

   # 5
   fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
   fx = layers.BatchNormalization()(fx)
   fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
   out = layers.Add()([x,fx])
   out = layers.ReLU()(out)
   x = layers.BatchNormalization()(out)

   # 6
   fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
   fx = layers.BatchNormalization()(fx)
   fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
   out = layers.Add()([x,fx])
   out = layers.ReLU()(out)
   x = layers.BatchNormalization()(out)

   # 7
   fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
   fx = layers.BatchNormalization()(fx)
   fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
   out = layers.Add()([x,fx])
   out = layers.ReLU()(out)
   x = layers.BatchNormalization()(out)

   # 8
   fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
   fx = layers.BatchNormalization()(fx)
   fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
   out = layers.Add()([x,fx])
   out = layers.ReLU()(out)
   x = layers.BatchNormalization()(out)

   # 9
   fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
   fx = layers.BatchNormalization()(fx)
   fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
   out = layers.Add()([x,fx])
   out = layers.ReLU()(out)
   x = layers.BatchNormalization()(out)

   # 10
   fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
   fx = layers.BatchNormalization()(fx)
   fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
   out = layers.Add()([x,fx])
   out = layers.ReLU()(out)
   x = layers.BatchNormalization()(out)

   # 11
   fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
   fx = layers.BatchNormalization()(fx)
   fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
   out = layers.Add()([x,fx])
   out = layers.ReLU()(out)
   x = layers.BatchNormalization()(out)

   # 12
   fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
   fx = layers.BatchNormalization()(fx)
   fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
   out = layers.Add()([x,fx])
   out = layers.ReLU()(out)
   x = layers.BatchNormalization()(out)

   # 13
   fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
   fx = layers.BatchNormalization()(fx)
   fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
   out = layers.Add()([x,fx])
   out = layers.ReLU()(out)
   x = layers.BatchNormalization()(out)

   # 14
   fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
   fx = layers.BatchNormalization()(fx)
   fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
   out = layers.Add()([x,fx])
   out = layers.ReLU()(out)
   x = layers.BatchNormalization()(out)

   # 15
   fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
   fx = layers.BatchNormalization()(fx)
   fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
   out = layers.Add()([x,fx])
   out = layers.ReLU()(out)
   x = layers.BatchNormalization()(out)

   # 16
   fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
   fx = layers.BatchNormalization()(fx)
   fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
   out = layers.Add()([x,fx])
   out = layers.ReLU()(out)
   x = layers.BatchNormalization()(out)

  representation_model = keras.Model(inputs = inputs, outputs = outputs, name = "rep")


def dynamics_initialization():
  # dynamics function
  # input - hidden state, action
  # output - next hidden state, reward
  global dynamics_model
  if name == "atari":
    shape_input = (6,6,257) # 256 + 1 = 257
    inputs = keras.Input(shape = shape_input)
    conv = layers.Conv2D(257, (3,3), activation = "relu", padding = "same")
    x = conv(inputs)

    # 2 residual blocks
    kernel_size = (3,3)
    filters = 257
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    x = layers.Conv2D(257, (3,3), activation = "relu", padding = "same")(x)

    # 3 residual blocks
    filters = 257
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    # x = layers.AveragePooling2D((2,2), 2)(x)

    # 3 residual blocks
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    outputs = layers.BatchNormalization()(out)

    # outputs = layers.AveragePooling2D((2,2), 2)(x)
    # output shape: (6, 6, 257), state + reward

  elif name == "chess":
    shape_input = (8,8,263) # 256 + 7 = 263
    inputs = keras.Input(shape = shape_input)
    conv = layers.Conv2D(257, (3,3), activation = "relu", padding = "same")
    x = conv(inputs)
    filters = 257
    kernel_size = (3,3)

    # 1
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    # 2
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    # 3
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    # 4
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    # 5
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    # 6
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    # 7
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out) 

    # 8
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out) 

    # 9
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out) 

    # 10
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    # 11
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    # 12
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    # 13
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    # 14
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

    # 15
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out) 

    # 16
    fx = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
    out = layers.Add()([x,fx])
    out = layers.ReLU()(out)
    x = layers.BatchNormalization()(out)

  dynamics_model = keras.Model(inputs = inputs, outputs = outputs, name = "dyn")


def prediction_initialization():
  # prediction function
  # input - hidden state
  # output - value, policy

  global prediction_model

  if name == "atari":
    inputs = keras.Input(shape = (6,6,256))
    conv = layers.Conv2D(128, (3,3), activation = "relu", padding = "same")
    x = conv(inputs)
    x = layers.Conv2D(64, (3,3), activation = "relu", padding = "same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1000)(x)
    x = layers.Dense(200)(x)
    outputs = layers.Dense(4)(x) # action_space_size + 1 

  elif name == "chess":
    shape_input = (8,8,256) # 256 + 7 = 263
    inputs = keras.Input(shape = shape_input)
    conv = layers.Conv2D(256, (3,3), activation = "relu", padding = "same")
    x = conv(inputs)
    
    x = layers.Conv2D(200, (3,3), activation = "relu", padding = "same")(x)
    x = layers.Conv2D(128, (3,3), activation = "relu", padding = "same")(x)
    x = layers.Conv2D(64, (3,3), activation = "relu", padding = "same")(x)
    x = layers.Conv2D(32, (3,3), activation = "relu", padding = "same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1000)(x)
    x = layers.Dense(200)(x)
    outputs = layers.Dense(4)(x) # action_space_size + 1 

  prediction_model = keras.Model(inputs = inputs, outputs = outputs, name = "pre")

def representation(observation, predS):
  try:
    representation_model
  except NameError:
    representation_initialization()
  
  weights = predS.get()
  predS.put(weights)
  
  for i in range(len(prediction_model.trainable_variables)):
    prediction_model.trainable_variables.assign[i](weights[i])
  
  hidden_state = representation_model.predict(observation)
  return hidden_state


# the hidden state as input has action plane as well
def dynamics(hidden_state, action, dynS):
  try:
    dynamics_model
  except NameError:
    dynamics_initialization(name)

  weights = dynS.get()
  dynS.put(weights)
  
  for i in range(len(prediction_model.trainable_variables)):
    prediction_model.trainable_variables.assign[i](weights[i])
  
  #### ATARI SPECIFIC ####
  arr_a = np.fill((6,6), action)
  input_a = np.vstack((hidden_state, arr_a))
  
  output_a = dynamics_model.predict(input_a)
  reward = output_a[0][0][256]
  next_hidden_state = np.fill((6,6,256), 0)
  for r in range(6):
    for c in range(6):
      for h in range(256):
        next_hidden_state[r][c][h] = output_a[r][c][h]
  
  return next_hidden_state, reward


def prediction(hidden_state):
  try:
    prediction_model
  except NameError:
    prediction_initialization()
    
  p1, p2, p3, v = prediction_model.predict(hidden_state)
  return p1, p2, p3, v 

  
def rep_init_w():
  representation_initialization()
  n = len(representation_model.trainable_variables)
  arr = []
  for i in range(n):
    arr.append(representation_model.trainable_variables[i])
    
  return arr  

def dyn_init_w():
  dynamics_initialization()
  n = len(dynamics_model.trainable_variables)
  arr = []
  
  for i in range(n):
    arr.append(dynamics_model.trainable_variables[i])
    
  return arr  
    
def pred_init_w():
  prediction_initialization()
  n = len(prediction_model.trainable_variables)
  arr = []
  
  for i in range(n):
    arr.append(prediction_model.trainable_variables[i])
   
  return arr


global name
name = "atari"
rep_init_w()
