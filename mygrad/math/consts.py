from mygrad.tensor_base import Tensor

c = Tensor(2.99792458e9, constant=True) # speed of light in m/s
e = Tensor(2.718281828459045, constant=True)
euler_gamma = Tensor(0.5772156649015329, constant=True)
G = Tensor(6.67384e-11, constant=True) # gravitational constant in (Nm^2)/(kg^2)
h = Tensor(6.67384e-34, constant=True) # planck's constant in J*s
pi = Tensor(3.141592653589793, constant=True)
phi = Tensor(1.618033988749895, constant=True)
tau = pi * 2
