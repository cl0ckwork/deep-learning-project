# --- Source --- #
# https://github.com/waleedka/hiddenlayer

import time
import numpy as np
import torch
import torchvision.models
import hiddenlayer as hl  # Allows the creation of the model graphic
# import scipy.stats # # ImportError: No module named 'scipy._lib.decorator'


# VGG16 with BatchNorm
model = torchvision.models.vgg16()

# Build HiddenLayer graph
# Jupyter Notebook renders it automatically
hl.build_graph(model, torch.zeros([1, 3, 224, 224])) # correct torch.zeros size required

# save the graphic

# im.save(path="path_of_file/name_of_file" , format="jpg") # correct pathing

# -- scipy issue
# ImportError: No module named 'scipy._lib.decorator'

# -- uninstalled and reinstalled scipy and issue remains


#-- One Chart --#
# A History object to store metrics
history1 = hl.History()

# A Canvas object to draw the metrics
canvas1 = hl.Canvas()

# Simulate a training loop with two metrics: loss and accuracy
loss = 1
accuracy = 0
for step in range(800):
    # Fake loss and accuracy
    loss -= loss * np.random.uniform(-.09, 0.1)
    accuracy = max(0, accuracy + (1 - accuracy) * np.random.uniform(-.09, 0.1))

    # Log metrics and display them at certain intervals
    if step % 10 == 0:
        # Store metrics in the history object
        history1.log(step, loss=loss, accuracy=accuracy)

        # Plot the two metrics in one graph
        canvas1.draw_plot([history1["loss"], history1["accuracy"]])

        time.sleep(0.1)



#-- Two Charts --#
# New history and canvas objects
history2 = hl.History()
canvas2 = hl.Canvas()

# Simulate a training loop with two metrics: loss and accuracy
loss = 1
accuracy = 0
for step in range(800):
    # Fake loss and accuracy
    loss -= loss * np.random.uniform(-.09, 0.1)
    accuracy = max(0, accuracy + (1 - accuracy) * np.random.uniform(-.09, 0.1))

    # Log metrics and display them at certain intervals
    if step % 10 == 0:
        history2.log(step, loss=loss, accuracy=accuracy)

        # Draw two plots
        # Encluse them in a "with" context to ensure they render together
        with canvas2:
            canvas2.draw_plot([history1["loss"], history2["loss"]],
                              labels=["Loss 1", "Loss 2"])
            canvas2.draw_plot([history1["accuracy"], history2["accuracy"]],
                              labels=["Accuracy 1", "Accuracy 2"])
        time.sleep(0.1)




#-- Creating a graph of the network --#


import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import hiddenlayer as hl

# -- Example graph 1
with tf.Session() as sess:
    with tf.Graph().as_default() as tf_graph:
        # Setup input placeholder
        inputs = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
        # Build model
        predictions, _ = nets.vgg.vgg_16(inputs)
        # Build HiddenLayer graph
        hl_graph = hl.build_graph(tf_graph)

# Display graph
# Jupyter Notebook renders it automatically, mod for non-GUI remote system or import model local?
hl_graph


# -- example graph 2

# Resnet101
model = torchvision.models.resnet101()

# Rather than using the default transforms, build custom ones to group
# nodes of residual and bottleneck blocks.
transforms = [
    # Fold Conv, BN, RELU layers into one
    hl.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu"),
    # Fold Conv, BN layers together
    hl.transforms.Fold("Conv > BatchNorm", "ConvBn"),
    # Fold bottleneck blocks
    hl.transforms.Fold("""
        ((ConvBnRelu > ConvBnRelu > ConvBn) | ConvBn) > Add > Relu
        """, "BottleneckBlock", "Bottleneck Block"),
    # Fold residual blocks
    hl.transforms.Fold("""ConvBnRelu > ConvBnRelu > ConvBn > Add > Relu""",
                       "ResBlock", "Residual Block"),
    # Fold repeated blocks
    hl.transforms.FoldDuplicates(),
]

# Display graph using the transforms above
hl.build_graph(model, torch.zeros([1, 3, 224, 224]), transforms=transforms)


# -- example graph 3

with tf.Session() as sess:
    with tf.Graph().as_default() as tf_graph:
        # Setup input placeholder
        inputs = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
        # Build model
        predictions, _ = nets.inception.inception_v1(inputs)
        # Build layout
        hl_graph = hl.build_graph(tf_graph)

# Display
hl_graph




## -- Another graph
# Has an interesting way of creating additional model graphs if we want to somewhat manually create
# test on the initial model, NTL 1600 25 APR 19


# https://github.com/szagoruyko/pytorchviz
# pip install torchviz

import torch
from torch import nn
from torchviz import make_dot

# The method below is for building directed graphs of PyTorch operations, built during forward propagation and
# showing which operations will be called on backward. It omits subgraphs which do not require gradients.

model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))

x = torch.randn(1,8)

make_dot(model(x), params=dict(model.named_parameters()))


# -- Graphics package
# TNT is a library providing powerful dataloading, logging and visualization utilities for Python. It is closely
#     integrated with PyTorch and is designed to enable rapid iteration with any model or training regimen.
# -- https://github.com/pytorch/tnt

# pip install torchnet

# --- tensorboardX
# This package currently supports logging scalar, image, audio, histogram, text, embedding, and the route of back-propagation.

# pip install tensorboardX

# -- sklearn AUC
#
# sklearn.metrics.roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True

