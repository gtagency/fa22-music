import collections
import datetime
import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf

from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple

def simple_train(model, loss, optimizer, train_dataset):
    model.compile(
        loss=loss,
        loss_weights={
            'pitch': 0.05,
            'step': 1.0,
            'duration': 1.0,
        },
        optimizer=optimizer,
    )

    model.evaluate(train_dataset, return_dict=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./training_checkpoints/ckpt_{epoch}',
            save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            verbose=1,
            restore_best_weights=True),
    ]

    epochs = 50

    history = model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    return history

def plot_history(history):
    plt.plot(history.epoch, history.history['loss'], label='total loss')
    plt.show()

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)


def get_loss_and_opt(learning_rate):
    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return loss, optimizer