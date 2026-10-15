"""
Train on one dataset (numpy files) and evaluate on another (cross-dataset).

Usage examples:
  python -m pa_csi.cross_eval --train-amp datasets/our_data_amp.npy --train-phase datasets/our_data_phase.npy --train-label datasets/our_data_label.npy \
      --test-amp data_6c_2_hang_750.npy --test-phase data_6c_2_hang_750.npy --test-label label_6c_2_hang_750.npy

The script infers input shapes from the numpy arrays and prints precision/recall/F1 and confusion matrix.
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time

from .models import Two_Stream_Model
from .grn import GatesResidualNetwork


def build_model(input_shape, hlayers=5, vlayers=1, hheads=9, vheads=50, K=10, sample=2, num_class=6):
    inputs = layers.Input(shape=input_shape)
    inputs_1 = layers.Input(shape=input_shape)
    x = Two_Stream_Model(hlayers,vlayers,hheads,vheads,K,sample,num_class,input_shape[0])(inputs)
    x_1 = Two_Stream_Model(hlayers,vlayers,hheads,vheads,K,sample,num_class,input_shape[0])(inputs_1)
    gate = GatesResidualNetwork(256)(x, x_1)
    outputs = layers.Dense(num_class,activation='softmax')(gate)
    model = keras.Model(inputs=[inputs,inputs_1], outputs=outputs)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-amp', required=True)
    p.add_argument('--train-phase', required=True)
    p.add_argument('--train-label', required=True)
    p.add_argument('--test-amp', required=True)
    p.add_argument('--test-phase', required=True)
    p.add_argument('--test-label', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()

    print('Loading training data...')
    x_train_amp = np.load(args.train_amp, allow_pickle=True)
    x_train_phase = np.load(args.train_phase, allow_pickle=True)
    y_train = np.load(args.train_label, allow_pickle=True)

    print('Loading test data...')
    x_test_amp = np.load(args.test_amp, allow_pickle=True)
    x_test_phase = np.load(args.test_phase, allow_pickle=True)
    y_test = np.load(args.test_label, allow_pickle=True)

    print('Train shapes:', x_train_amp.shape, x_train_phase.shape, y_train.shape)
    print('Test shapes :', x_test_amp.shape, x_test_phase.shape, y_test.shape)

    # infer input shape (timesteps, features)
    input_shape = x_train_amp.shape[1:]
    num_class = int(np.max(y_train)) + 1

    model = build_model(input_shape, num_class=num_class)

    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0001,
        decay_steps=10000,
        decay_rate=0.9
    )

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
                    loss="sparse_categorical_crossentropy", 
                    metrics=["accuracy"])

    model.summary()

    # Train
    model.fit([x_train_amp, x_train_phase], y_train, 
              validation_split=0.1, 
              epochs=args.epochs, 
              batch_size=args.batch_size)

    # Evaluate on test set
    print('\nEvaluating on test set...')
    t0 = time.time()
    loss, acc = model.evaluate([x_test_amp, x_test_phase], y_test, batch_size=args.batch_size)
    preds = model.predict([x_test_amp, x_test_phase], batch_size=args.batch_size)
    preds = np.argmax(preds, axis=1)
    t1 = time.time()

    precision = precision_score(y_test, preds, average='weighted')
    recall = recall_score(y_test, preds, average='weighted')
    f1 = f1_score(y_test, preds, average='weighted')
    cm = confusion_matrix(y_test, preds)

    print(f'Test loss: {loss:.4f}, Test acc: {acc:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    print('Confusion Matrix:')
    print(cm)
    print('Time to evaluate: {:.2f}s'.format(t1 - t0))


if __name__ == '__main__':
    main()
