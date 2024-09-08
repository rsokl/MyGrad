import mygrad as mg
from mygrad.nnet.activations import leaky_relu
import matplotlib.pyplot as plt
x = mg.linspace(-2, 2, 100)
y = leaky_relu(x, slope=0.1)
plt.title("leaky_relu(x, slope=0.1)")
y.backward()
plt.plot(x, x.grad, label="df/dx")
plt.plot(x, y, label="f(x)")
plt.legend()
plt.grid()
plt.show()
