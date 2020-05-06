import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from rbm_model import RBM

train, test = keras.datasets.mnist.load_data()

x_train, _ = train
x_test, _ = test

# Reformat dataset
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model = RBM(visible_units=784, hidden_units=64)

# Shuffle and split dataset into batches
train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(64)

# Hyperparameters
epochs = 3
alpha = 0.01

def run(dataset, train=True):
    for step, batch in enumerate(dataset):
        loss = model.train(batch, alpha) if train else model.evaluate(batch)
        
        if step % 100 == 0:
            print('step %s: loss = %s' % (step, loss))

for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))
    run(train_dataset)
print('')
# Evaluation
run(test_dataset, train=False)