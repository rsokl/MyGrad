import mygrad as mg
from mygrad.nnet.activations import relu
import matplotlib.pyplot as plt
x = mg.linspace(-2, 2, 100)
y = relu(x)
plt.title("relu(x)")
y.backward()
plt.plot(x, x.grad, label="df/dx")
plt.plot(x, y, label="f(x)")
plt.legend()
plt.grid()
plt.show()
