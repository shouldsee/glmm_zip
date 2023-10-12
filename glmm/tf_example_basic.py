#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2020 The TensorFlow Authors.

# In[ ]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# In the previous guides, you have learned about [tensors](./tensor.ipynb), [variables](./variable.ipynb), [gradient tape](autodiff.ipynb), and [modules](./intro_to_modules.ipynb).  In this guide, you will fit these all together to train models.
# 
# TensorFlow also includes the [tf.Keras API](https://www.tensorflow.org/guide/keras/overview), a high-level neural network API that provides useful abstractions to reduce boilerplate.  However, in this guide, you will use basic classes.

# ## Setup

# In[ ]:


import tensorflow as tf
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# ## Solving machine learning problems
# 
# Solving a machine learning problem usually consists of the following steps:
# 
#  - Obtain training data.
#  - Define the model.
#  - Define a loss function.
#  - Run through the training data, calculating loss from the ideal value
#  - Calculate gradients for that loss and use an *optimizer* to adjust the variables to fit the data.
#  - Evaluate your results.
# 
# For illustration purposes, in this guide you'll develop a simple linear model, $f(x) = x * W + b$, which has two variables: $W$ (weights) and $b$ (bias).
# 
# This is the most basic of machine learning problems:  Given $x$ and $y$, try to find the slope and offset of a line via  [simple linear regression](https://en.wikipedia.org/wiki/Linear_regression#Simple_and_multiple_linear_regression).

# ## Data
# 
# Supervised learning uses *inputs* (usually denoted as *x*) and *outputs* (denoted *y*, often called *labels*).  The goal is to learn from paired inputs and outputs so that you can predict the value of an output from an input.
# 
# Each input of your data, in TensorFlow, is almost always represented by a tensor, and is often a vector. In supervised training, the output (or value you'd like to predict) is also a tensor.
# 
# Here is some data synthesized by adding Gaussian (Normal) noise to points along a line.


# The actual line
TRUE_W = 3.0
TRUE_B = 2.0
NUM_EXAMPLES = 201
def f(x):
  return x * TRUE_W + TRUE_B
  
def load_data():


  # A vector of random x values
  x = tf.linspace(-2,2, NUM_EXAMPLES)
  x = tf.cast(x, tf.float32)



  # Generate some noise
  noise = tf.random.normal(shape=[NUM_EXAMPLES])

  # Calculate y
  y = f(x) + noise
  return x,y




class MyModel(tf.Module):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.w = tf.Variable(5.0)
    self.b = tf.Variable(0.0)

  def __call__(self, x):
    return self.w * x + self.b

def loss(target_y, predicted_y):
  return tf.reduce_mean(tf.square(target_y - predicted_y))



# Given a callable model, inputs, outputs, and a learning rate...
def train(model, x, y, learning_rate):

  with tf.GradientTape() as t:
    # Trainable variables are automatically tracked by GradientTape
    current_loss = loss(y, model(x))

  # Use GradientTape to calculate the gradients with respect to W and b
  dw, db = t.gradient(current_loss, [model.w, model.b])

  # Subtract the gradient scaled by the learning rate
  model.w.assign_sub(learning_rate * dw)
  model.b.assign_sub(learning_rate * db)


def report(model, loss):
  return f"W = {model.w.numpy():1.2f}, b = {model.b.numpy():1.2f}, loss={loss:2.5f}"



def main():
  x,y = load_data()

  model = MyModel()

  # Collect the history of W-values and b-values to plot later
  weights = []
  biases = []

  # Define a training loop
  def training_loop(model, x, y, num_epochs):

    for epoch in range(num_epochs):
      # Update the model with the single giant batch
      train(model, x, y, learning_rate=0.1)

      # Track this before I update
      weights.append(model.w.numpy())
      biases.append(model.b.numpy())
      current_loss = loss(y, model(x))

      print(f"Epoch {epoch:2d}:")
      print("    ", report(model, current_loss))


  current_loss = loss(y, model(x))

  print(f"Starting:")
  print("    ", report(model, current_loss))

  training_loop(model, x, y, 10)



  # Before training the model, you can visualize the loss value by plotting the model's predictions in red and the training data in blue:

  plt.plot(x, y, '.', label="Data")
  plt.plot(x, f(x), label="Ground truth")
  plt.plot(x, model(x), label="Predictions")
  plt.legend()
  plt.show()
  print("Current loss: %1.6f" % loss(y, model(x)).numpy())


main()
assert 0

# Plot the evolution of the weights over time:

# In[ ]:


plt.plot(epochs, weights, label='Weights', color=colors[0])
plt.plot(epochs, [TRUE_W] * len(epochs), '--',
         label = "True weight", color=colors[0])

plt.plot(epochs, biases, label='bias', color=colors[1])
plt.plot(epochs, [TRUE_B] * len(epochs), "--",
         label="True bias", color=colors[1])

plt.legend()
plt.show()


# Visualize how the trained model performs

# In[ ]:


plt.plot(x, y, '.', label="Data")
plt.plot(x, f(x), label="Ground truth")
plt.plot(x, model(x), label="Predictions")
plt.legend()
plt.show()

print("Current loss: %1.6f" % loss(model(x), y).numpy())


# ## The same solution, but with Keras
# 
# It's useful to contrast the code above with the equivalent in Keras.
# 
# Defining the model looks exactly the same if you subclass `tf.keras.Model`.  Remember that Keras models inherit ultimately from module.

# In[ ]:


class MyModelKeras(tf.keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # Initialize the weights to `5.0` and the bias to `0.0`
    # In practice, these should be randomly initialized
    self.w = tf.Variable(5.0)
    self.b = tf.Variable(0.0)

  def call(self, x):
    return self.w * x + self.b

keras_model = MyModelKeras()

# Reuse the training loop with a Keras model
training_loop(keras_model, x, y)

# You can also save a checkpoint using Keras's built-in support
keras_model.save_weights("my_checkpoint")


# Rather than write new training loops each time you create a model, you can use the built-in features of Keras as a shortcut.  This can be useful when you do not want to write or debug Python training loops.
# 
# If you do, you will need to use `model.compile()` to set the parameters, and `model.fit()` to train.  It can be less code to use Keras implementations of L2 loss and gradient descent, again as a shortcut.  Keras losses and optimizers can be used outside of these convenience functions, too, and the previous example could have used them.

# In[ ]:


keras_model = MyModelKeras()

# compile sets the training parameters
keras_model.compile(
    # By default, fit() uses tf.function().  You can
    # turn that off for debugging, but it is on now.
    run_eagerly=False,

    # Using a built-in optimizer, configuring as an object
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),

    # Keras comes with built-in MSE error
    # However, you could use the loss function
    # defined above
    loss=tf.keras.losses.mean_squared_error,
)


# Keras `fit` expects batched data or a complete dataset as a NumPy array.  NumPy arrays are chopped into batches and default to a batch size of 32.
# 
# In this case, to match the behavior of the hand-written loop, you should pass `x` in as a single batch of size 1000.

# In[ ]:


print(x.shape[0])
keras_model.fit(x, y, epochs=10, batch_size=1000)


# Note that Keras prints out the loss after training, not before, so the first loss appears lower, but otherwise this shows essentially the same training performance.

# ## Next steps
# 
# In this guide, you have seen how to use the core classes of tensors, variables, modules, and gradient tape to build and train a model, and further how those ideas map to Keras.
# 
# This is, however, an extremely simple problem. For a more practical introduction, see [Custom training walkthrough](../tutorials/customization/custom_training_walkthrough.ipynb).
# 
# For more on using built-in Keras training loops, see [this guide](https://www.tensorflow.org/guide/keras/train_and_evaluate).  For more on training loops and Keras, see [this guide](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch).  For writing custom distributed training loops, see [this guide](distributed_training.ipynb#using_tfdistributestrategy_with_basic_training_loops_loops).

# %%
