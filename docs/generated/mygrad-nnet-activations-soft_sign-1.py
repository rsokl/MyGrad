import mygrad as mg
from mygrad.nnet.activations import soft_sign
import matplotlib.pyplot as plt
x = mg.linspace(-10, 10, 100)
y = soft_sign(x)
plt.title("soft_sign(x)")
y.backward()
plt.plot(x, x.grad, label="df/dx")
plt.plot(x, y, label="f(x)")
plt.legend()
plt.grid()
plt.show()
