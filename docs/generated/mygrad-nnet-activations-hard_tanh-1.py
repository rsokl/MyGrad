import mygrad as mg
from mygrad.nnet.activations import hard_tanh
import matplotlib.pyplot as plt
x = mg.linspace(-6, 6, 100)
y = hard_tanh(x, lower_bound=-3, upper_bound=3)
plt.title("hard_tanh(x, lower_bound=-3, upper_bound=3)")
y.backward()
plt.plot(x, x.grad, label="df/dx")
plt.plot(x, y, label="f(x)")
plt.legend()
plt.grid()
plt.show()
