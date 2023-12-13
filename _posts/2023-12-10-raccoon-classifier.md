---
title: Raccoon Image Classifier
date: 2023-12-12
categories: [Python, Streamlit, Neural-Networks, Raccoons]
tags: [python, neural-networks]
toc: true
math: true
image:
  path: /assets/images/2023-12-12-raccoon-classifier/mouth.jpg
---

# App

<iframe
  src="https://raccoonclassifier1.streamlit.app/?embed=true"
  height="450"
  style="width:100%;border:none;"
></iframe>

# Introduction

So, fun fact about me, I have never deployed a Neural Network (NN) before.

There's never been a need to in my line of work, nor in my hobby data projects. The data I work with is overwhelmingly in tabular format, thus tree-based methods or linear models are almost always preferred. They're 'simpler', faster, and the performance differences are negligible to the point at which I don't bother to learn one of `Tensorflow` or `Pytorch`. Further, I really don't enjoy fitting something I don't understand, and NNs are very complicated. Despite still not understanding the finer details of NNs, I've finally fit one... to classify pictures of raccoons.



Why raccoons? Because they're cool little fellas and my favorite animal :tm: (as you can probably ascertain from the theming on this website).

# The Code

Before getting started, please shove this at the top of your script/notebook and ensure you have these packages installed:

```python
import os # This import was my doing
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

Anyway, this process is a near copy-paste walkthrough from `Tensorflow`'s image classification tutorial found [here](https://www.tensorflow.org/tutorials/images/classification). The only things which I altered were the training and testing data to accommodate my goal of classifying raccoons rather than flowers. I would argue that the data procurement step is the most tedious/challenging part of this exercise. Fortunately, once data is secured, the process for ingestion is quired straightforward thanks to `keras`. We'll want to set up folders somewhere on our computer containing the images for training, with each folder being named to the class of the images contained inside. For example, my folder looked like this:

```text
images/
...raccoon/
......raccoon_1.jpg
......raccoon_2.jpg
...dog/
......dog_1.jpg
......dog2_2.jpg
.
.
.
...something_else/
......human_1.jpg
......tiger_1.jpg
......car_1.jpg
```



Thankfully, `keras` can magically take this directory and split them into train/test sets for us by using the following code:

```python
# Will resize images in train/test data to 180x180:
img_height = 180
img_width = 180

# Training set:
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width)
  )

  # Testing set:
  val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width)
  )
```
If you want to ensure that the code did its job properly and identified the appropriate folders, you may print the class names the model will be trained for:

```python
# Print to user what the class names are:
class_names = train_ds.class_names
num_classes = len(class_names)
for name in class_names:
    print(f"Class identified: {name}")
```
```text
Class identified: cat
Class identified: dog
Class identified: possum
Class identified: raccoon
Class identified: something else
```
The next thing is telling `tensorflow`/`keras` how we want the ingestion process to be handled. If we use the `AUTOTUNE` method on `tf.data`, it will instruct the program to dynamically pull the data in preparation for training while the model is already running prior data fetches. This will speed the overall training process up.

```python
# TF Autotuner:
AUTOTUNE = tf.data.AUTOTUNE

# More options which cache data in memory for speed and add overfitting precautions.
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```
Next we can actually write our model by specifying the layers of our Convolutional Neural Network

```python
# Set model layers and overfitting precautions:
model = Sequential([
  layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
```

So, things to go over here: what are each of the methos being called doing?

* `RandomFlip` does exactly what it sounds like, it's randomly flipping our images horizontally. This is a measure to help prevent overfitting.
* `RandomRotation` same logic as above, but with a rotation transformation.
* `RandomZoom`... you get the idea.
* `Rescaling` normalizes the data by dividing by the maximum value that a color channel's 'intensity', or brightness takes on. This is, for reasons I don't entirely understand, helpful.
* `Conv2D`: This sets the number of neurons/nodes in your NN. The number of neurons are determined by the number of filters you place onto the image (in the first call of this method, 16). Filters are effectively windows capturing a portion of your image, set to a given kernal size (in this case, a 3x3 matrix, or 'window') which convolutes across your image to create a feature map.
* `MaxPooling`: This downsamples the spatial dimension of the feature map by selecting the maximum value within your kernel window.
* `Dropout`: Will randomly drop input data from training to prevent overfitting.
* `Flatten`: Flattens the input data into a one-dimensional array.
* `Dense`: This is the dense layer of neurons that take the image processing outputs of prior steps to spit out a final prediction.

Next, we want to set the metric we wish to optimize and which loss function we wish to minimize:

```python
# Configure model:
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```
Now we're finally able to fit and save our model!

```python
# Fit NN:
epochs=15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Save the model as a TFLite file:
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('raccoon.tflite', 'wb') as f:
  f.write(tflite_model)
```

# Results.

After running the NN through 15 epochs, we've achieve about a 60% test set accuracy marginally. My next posts on this subject might be about improving this model through tuning techniques or feature engineering.



