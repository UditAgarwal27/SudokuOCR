from modelCode.SudokuNetwork import SudokuNet
from keras.optimizers import Adam
from keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to output model aafter training")
args = vars(ap.parse_args())

initial_learning_rate = 1e-3
epochs = 10
batch_size = 128

((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

trainData = trainData.astype("float32")/256.0
testData = testData.astype("float32")/256.0

le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.fit_transform(testLabels)

opt = Adam(lr=initial_learning_rate)
model = SudokuNet.buildmodel(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(trainData, trainLabels, validation_data=(testData, testLabels), batch_size=batch_size, epochs=epochs,
              verbose=1)

predictions = model.predict(testData)
print(classification_report(
    testLabels.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=[str(x) for x in le.classes_]
))

model.save(args["model"], save_format="h5")
