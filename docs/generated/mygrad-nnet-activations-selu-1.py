import mygrad as mg
from mygrad.nnet.activations import selu
import matplotlib.pyplot as plt
x = mg.linspace(-2, 2, 100)
y = selu(x)
plt.title("selu(x)")
y.backward()
plt.plot(x, x.grad, label="df/dx")
plt.plot(x, y, label="f(x)")
plt.legend()
plt.grid()
plt.show()
