import functools
import mygrad as mg
import timeit
import numpy as np
def multi_matmul_slow(arrays): return functools.reduce(mg.matmul, arrays)

x4 = np.random.randint(5, size=(2,20,40,60))
y4 = np.random.randint(5, size=(1,1,60,100))
z4 = np.random.randint(5, size=(20,100,10))
e14 = np.random.randint(5, size=(2,20,10,1000))
e24 = np.random.randint(5, size=(1000,20))
e34 = np.random.randint(5, size=(2,1,20,500))

arrays_4D = [x4,y4,z4,e14,e24,e34]




#print(timeit.timeit("test4s()", number=100, setup="from __main__ import test4s") , timeit.timeit("test4()", number=100, setup="from __main__ import test4"))

def test_speed():
	def _test4s(a=arrays_4D):
		multi_matmul_slow(a)

	def _test4(a=arrays_4D):
		mg.multi_matmul(a)

	assert timeit.timeit("_test4s()", number=100, setup="from __main__ import _test4s") < timeit.timeit("_test4()", number=100, setup="from __main__ import _test4")