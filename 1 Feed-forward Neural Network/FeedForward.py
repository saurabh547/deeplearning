#pip install tensorflow scikit-learn matplotlib numpy

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, Adadelta
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# Grab the MNIST dataset
print("[INFO] accessing MNIST...")
((trainX, trainY), (testX, testY)) = mnist.load_data()

# Flatten the images
trainX = trainX.reshape((trainX.shape[0], 28 * 28 * 1))
testX = testX.reshape((testX.shape[0], 28 * 28 * 1))

# Scale data to the range of [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# Convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Define the 784-256-128-10 architecture using Keras
def create_model():
    model = Sequential()
    model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(10, activation="softmax"))
    return model

# List of optimizers to train the model with
optimizers = {
    "SGD": SGD(0.01),
    "Adam": Adam(),
    "Adadelta": Adadelta()
}

# Dictionary to store the history of models
histories = {}

# Loop over the optimizers and train the model
for key, opt in optimizers.items():
    print(f"[INFO] training network with {key} optimizer...")
    model = create_model()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)
    histories[key] = H.history

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure(figsize=(12, 8))

for key in histories.keys():
    plt.plot(np.arange(0, 100), histories[key]["loss"], label=f"{key} train_loss")
    plt.plot(np.arange(0, 100), histories[key]["val_loss"], label=f"{key} val_loss")
    plt.plot(np.arange(0, 100), histories[key]["accuracy"], label=f"{key} train_acc")
    plt.plot(np.arange(0, 100), histories[key]["val_accuracy"], label=f"{key} val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

# Evaluate the network with each optimizer
for key, opt in optimizers.items():
    print(f"[INFO] evaluating network with {key} optimizer...")
    model = create_model()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.fit(trainX, trainY, epochs=100, batch_size=128, verbose=0)
    predictions = model.predict(testX, batch_size=128)
    print(f"Results for {key}:")
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))





