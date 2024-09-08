import mygrad as mg
from mygrad.nnet import conv_nd
import matplotlib.pyplot as plt
kernel = mg.ones(5)  # a square-wave signal
x = mg.zeros((1, 1, 16))  # a square-wave signal
x[..., 5:11] = 1
k = mg.ones((1, 1, 5))    # a constant-valued kernel
y = conv_nd(x, k, stride=1)   # performing a stride-1, 1D convolution
plt.title("conv(f, g); stride: 1")
y.backward()
plt.plot(x.data[0,0], label="f", ls="--", lw=3, drawstyle='steps-pre')
plt.plot(kernel, label="g", ls="--", lw=3, drawstyle='steps-pre')
plt.plot(y.data[0,0], label="f * g")
plt.plot(mg.arange(16.), x.grad[0, 0], label="d[sum(f * g)]/df")
kernel = mg.ones(5)  # a square-wave signal
plt.legend()
plt.grid()
plt.show()
