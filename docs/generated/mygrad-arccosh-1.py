import mygrad as mg
import matplotlib.pyplot as plt
x = mg.linspace(1.1, 10, 100)
y = mg.arccosh(x)
y.backward()
plt.plot(x, x.grad, label="df/dx")
plt.plot(x, y, label="f(x)")
plt.legend()
plt.grid()
plt.show()
