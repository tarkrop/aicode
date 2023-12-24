import sys, os

sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from deep_conv import DeepConvNet
from dataset.mnist import load_mnist


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
network.load_params("deep_convnet_params.pkl")

sampled = 10000
x_test = x_test[:sampled]
t_test = t_test[:sampled]

print("caluculate accuracy... ")
print(network.accuracy(x_test, t_test))
