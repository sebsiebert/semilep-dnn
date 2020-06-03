import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
import ROOT
# import matplotlib.pyplot as plt
import datetime

import data

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


# import wandb
# wandb.init(project="physics")

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
    model.add(Dropout(0.1))
    for _ in range(9):
        model.add(Dense(100, activation='elu'))
        model.add(LayerNormalization())
        model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))
    return model




# Define loass function and optimizer
loss_func = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

# Average the loss across the batch size within an epoch
train_loss = tf.keras.metrics.Mean(name="train_loss")
valid_loss = tf.keras.metrics.Mean(name="test_loss")

# Specify the performance metric
train_acc = tf.keras.metrics.CategoricalAccuracy(name="train_acc")
valid_acc = tf.keras.metrics.CategoricalAccuracy(name="valid_acc")


generator = data.dataGenerator(batch_size=64)

# @tf.function
# def model_train(features, labels):
#     with tf.GradientTape() as tape:
#         predictions = model(features)
#         loss = loss_func(labels, predictions)
    
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#     train_loss(loss)
#     train_acc(labels, predictions)


# @tf.function
# def model_validate(features, labels):
#     predictions = model(features)
#     v_loss = loss_func(labels, predictions)
#     valid_loss(v_loss)
#     valid_acc(labels, predictions)




model = getModel()
model.compile(
    optimizer=optimizer,
    loss=loss_func,
    metrics=['accuracy']
)

model.summary()
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

a = next(generator)
print(type(a))
b, c = a
print(type(b), b.shape)
print(type(c), c.shape)
H = model.fit(
    x = generator,
    steps_per_epoch=1000,
    epochs=50,
    callbacks=[tensorboard_callback]
)






# # Grab random images from the test and make predictions using 
# # the model *while it is training* and log them using WnB
# def get_sample_predictions():
#     predictions = []
#     images = []
#     random_indices = np.random.choice(X_test.shape[0], 25)
#     for index in random_indices:
#         image = X_test[index].reshape(1, 28, 28, 1)
#         prediction = np.argmax(model(image).numpy(), axis=1)
#         prediction = CLASSES[int(prediction)]
        
#         images.append(image)
#         predictions.append(prediction)
    
#     wandb.log({"predictions": [wandb.Image(image, caption=prediction) 
#                                for (image, prediction) in zip(images, predictions)]})


# # Train the model for 5 epochs
# for epoch in range(5):
#     # Run the model through train and test sets respectively
#     for (features, labels) in train_ds:
#         model_train(features, labels)

#     for test_features, test_labels in test_ds:
#         model_validate(test_features, test_labels)
        
#     # Grab the results
#     (loss, acc) = train_loss.result(), train_acc.result()
#     (val_loss, val_acc) = valid_loss.result(), valid_acc.result()
    
#     # Clear the current state of the metrics
#     train_loss.reset_states(), train_acc.reset_states()
#     valid_loss.reset_states(), valid_acc.reset_states()
    
#     # Local logging
#     template = "Epoch {}, loss: {:.3f}, acc: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}"
#     print (template.format(epoch+1,
#                          loss,
#                          acc,
#                          val_loss,
#                          val_acc))
    
#     # Logging with WnB
#     wandb.log({"train_loss": loss.numpy(),
#                "train_accuracy": acc.numpy(),
#                "val_loss": val_loss.numpy(),
#                "val_accuracy": val_acc.numpy()
#     })
#     get_sample_predictions()




# for layer in model.layers:
#     if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
#         print(layer.get_weights()[0].shape)
#         wandb.log({"weights": wandb.Histogram(layer.get_weights()[0])})
#         wandb.run.summary.update({"weights": wandb.Histogram(layer.get_weights()[0])})

