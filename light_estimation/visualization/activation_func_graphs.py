from matplotlib import pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def leaky_relu(x):
    return np.maximum(0.1 * x, x)

def elu(x):
    return np.where(x > 0, x, 1 * (np.exp(x) - 1))

x = np.linspace(-10, 10, 100)
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)
#y_leaky_relu = leaky_relu(x)
y_elu = elu(x)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(x, y_sigmoid, label='Sigmoid', color='blue')
plt.title("Sigmoid")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.subplot(2, 2, 2)
plt.plot(x, y_relu, label='ReLU', color='red')
plt.title("ReLU")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.subplot(2, 2, 3)
plt.plot(x, y_tanh, label='Tanh', color='green')
plt.title("Tanh")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
#plt.plot(x, y_leaky_relu, label='Leaky ReLU', color='purple', linestyle='--')
plt.subplot(2, 2, 4)
plt.plot(x, y_elu, label='ELU', color='orange')
plt.title("ELU")
#plt.axis([-10, 10, -1, 10])
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.tight_layout()
plt.show()