import numpy as np
import matplotlib.pyplot as plt

data = np.load("mnist.npz")

x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)



plt.imshow(x_train[0], cmap="gray")
plt.title(y_train[0])
plt.show()