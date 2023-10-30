# Import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", 
    default="F:/AI Learning/Deep learning/helmet_detection/dataset",
    help="path to input dataset")  # Update the dataset path
ap.add_argument("-m", "--model", type=str, 
    default="helmet_detector.model", 
    help="path to trained helmet detector model")  # Update the model path
ap.add_argument("-c", "--confidence", type=float, 
    default=0.5, 
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

from imutils import paths

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# Check if there are image files in the dataset directory
if len(imagePaths) == 0:
    print("[ERROR] No images found in the dataset directory.")
    exit()

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the input image (224x224) and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# Check if any labels are found
if len(labels) == 0:
    print("[ERROR] No labels found for the images.")
    exit()

# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.20, stratify=labels, random_state=42)

# Check if any training or testing data is available
if len(trainX) == 0 or len(testX) == 0:
    print("[ERROR] Not enough data for training/testing.")
    exit()

# The rest of the code remains unchanged.

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(1, activation="sigmoid")(headModel)  # Change to 1 output neuron for binary classification

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# Define your learning rate and other parameters
INIT_LR = 0.001  # You can adjust this value
EPOCHS = 10  # You can adjust this value
BS = 10

# Compile your model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
print("[INFO] saving helmet detector model...")
model.save(args["model"], save_format="h5")

# Train the head of the network
print("[INFO] training head....")
# Use steps_per_epoch when calling model.fit()
num_samples = len(trainX)
batch_size = BS

# Now, you can use steps_per_epoch when calling model.fit()
H = model.fit(
    aug.flow(trainX, trainY, batch_size=batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // batch_size,
    epochs=EPOCHS,
)

# Make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = (predIdxs > 0.5).astype("int32")  # Convert probabilities to binary predictions

# Show a nicely formatted classification report
print(classification_report(testY.astype("int"), predIdxs,
    target_names=lb.classes_))

# Serialize the model to disk (save it)
model.save(args["model"], save_format="h5")

# Plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")  # Changed from "acc"
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")  # Changed from "val_acc"
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()

# Show a summary of the model
model.summary()
