from mygrad import clip
import numpy as np

from tests.wrappers.uber import fwdprop_test_factory


@fwdprop_test_factory(num_arrays=3, mygrad_func=clip, true_func=np.clip)
def test_clip():
    pass
