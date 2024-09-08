import mygrad as mg
from mygrad.nnet.activations import elu
import matplotlib.pyplot as plt
x = mg.linspace(-2, 2, 100)
y = elu(x, alpha=0.1)
plt.title("elu(x, alpha=0.1)")
y.backward()
plt.plot(x, x.grad, label="df/dx")
plt.plot(x, y, label="f(x)")
plt.legend()
plt.grid()
plt.show()
