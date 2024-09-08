import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mygrad.nnet.layers import conv_nd
img = mpimg.imread('../_static/meerkat.png')
#
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
x = img.transpose(2,0,1)[:, None, :, :]
#
kernel = kernel.reshape(1, 1, *kernel.shape)
#
processed = conv_nd(x, kernel, stride=(1, 1)).data.squeeze().transpose(1, 2, 0)
#
fig, ax = plt.subplots()
ax.imshow(img)
#
fig, ax = plt.subplots()
ax.imshow(processed)
