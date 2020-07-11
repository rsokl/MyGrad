Neural network operations (:mod:`mygrad.nnet`)
**********************************************

.. currentmodule:: mygrad.nnet.layers


Layer operations
----------------
.. autosummary::
   :toctree: generated/

   batchnorm
   conv_nd
   max_pool
   gru

.. currentmodule:: mygrad.nnet.losses

Losses
------
.. autosummary::
   :toctree: generated/

   focal_loss
   margin_ranking_loss
   multiclass_hinge
   negative_log_likelihood
   softmax_crossentropy
   softmax_focal_loss


.. currentmodule:: mygrad.nnet.activations

Activations
-----------
.. autosummary::
   :toctree: generated/


   elu
   glu
   hard_tanh
   leaky_relu
   logsoftmax
   selu
   sigmoid
   softmax
   soft_sign
   relu
   tanh


.. currentmodule:: mygrad.nnet.initializers

Initializers
------------
.. autosummary::
   :toctree: generated/


   glorot_normal
   glorot_uniform
   he_normal
   he_uniform
   normal
   uniform

.. currentmodule:: mygrad

Sliding Window View Utility
---------------------------
.. autosummary::
   :toctree: generated/

   sliding_window_view
