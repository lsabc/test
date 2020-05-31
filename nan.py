import os
import tensorflow as tf
import numpy as np
import pdb

BATCH_SIZE = 100

def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((28,28,1)))
    #model.add(tf.keras.layers.Conv2D(32, [5, 5], activation='relu', padding='SAME'))
    model.add(tf.keras.layers.Conv2D(32, [5, 5], activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, [5, 5], activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, [3, 3], activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

def loss_fn(y_true, y_pred):
    num_classes = y_pred.shape[1]
    y_true = tf.reshape(y_true, [-1])
    y_true = tf.cast(y_true, tf.int32)
    loss = tf.reduce_sum(tf.one_hot(y_true, num_classes)*tf.math.log(y_pred)) # faulty statement
    loss = -loss/BATCH_SIZE
    tf.debugging.assert_all_finite(loss, "NAN Loss...")
    return loss

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE).repeat()

strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

with strategy.scope():
    model = create_model()
    optimizer=tf.keras.optimizers.Adam(1e-4)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['sparse_categorical_accuracy'])

model.fit(train_ds, steps_per_epoch=60000/BATCH_SIZE, epochs=50, verbose=2)

model.evaluate(x_test, y_test, verbose=2)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(......)

# Model weights are saved at the end of every epoch, if it's the best seen
# so far.
model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])
