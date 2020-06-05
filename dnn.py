import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
import ROOT
# import matplotlib.pyplot as plt
import datetime
import numpy as np
import data
import matplotlib.pyplot as plt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


# lep1 *4
# lep2 *4
# jet1 *3
# jet2 *3
# jet3 *3
# jet4 *3
# fatjet1 *4
# fatjet2 *4
# largest_nonW_mjj *1
# PuppiMET_pt *1
# PuppiMET_phi *1

def getModel():
    model = keras.Sequential()
    model.add(Dense(75, activation='elu', input_shape=(31,)))
    model.add(LayerNormalization())
    model.add(Dropout(0.05))
    for _ in range(9):
        model.add(Dense(100, activation='elu'))
        model.add(LayerNormalization())
        model.add(Dropout(0.05))
    model.add(Dense(2, activation='softmax'))
    return model




# Define loass function and optimizer
loss_func = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

train_batch_size = 128
val_batch_size = 2e5
test_batch_size = 1e5

generator = data.dataGenerator(batch_size=64)
val_gen = data.dataGenerator(batch_size=val_batch_size, mode="validate")
test_gen = data.dataGenerator(batch_size=test_batch_size, mode="test")

val_data = next(val_gen)
val_gen.close()



model = getModel()
model.compile(
    optimizer=optimizer,
    loss=loss_func,
    metrics=['accuracy']
)

model.summary()
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="training1/cp-{epoch:04d}.ckpt",
    save_weights_only=True,
    period=10)

reduceLR_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=5)

earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)


# a = next(generator)
# print(type(a))
# b, c = a
# print(type(b), b.shape)
# print(type(c), c.shape)
H = model.fit(
    x = generator,
    steps_per_epoch=5e4,
    epochs=100,
    validation_data=val_data,
    callbacks=[
        tensorboard_callback,
        cp_callback,
        reduceLR_callback,
        earlyStopping_callback,
    ],
)

model.save_weights("training1.end")
# evaluate final model on test data

# loss, accuracy = model.evaluate(x=test_gen,steps=1e3)


total_test = 1e6

labels = np.empty(int(total_test), dtype=int)
predictions = np.empty(int(total_test), dtype=float)

for i in range(int(total_test/test_batch_size)):
    x_test, y_test = next(test_gen)
    pred = model.predict(x_test)

    begin = i*int(test_batch_size)
    end = (i+1)*int(test_batch_size)

    labels[begin:end] = tf.argmax(y_test, axis=1)
    predictions[begin:end] = tf.argmax(pred, axis=1)

conf_matrix = tf.math.confusion_matrix(labels=labels, predictions=predictions)
conf_matrix = conf_matrix.numpy()
print(conf_matrix)

classes = ["background", "signal"]

fig, ax = plt.subplots()
im = ax.matshow(conf_matrix)

ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
ax.set_ylabel("True")
ax.set_xlabel("Predicted")


# print numbers in fiels
for i in range(len(classes)):
    for j in range(len(classes)):
        test = ax.text(j, i, conf_matrix[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Confusion matrix")
fig.tight_layout()

plt.savefig("conf_matrix.png")


# print("Final test loss: {}, accuracy: {}".format(loss, accuracy))
