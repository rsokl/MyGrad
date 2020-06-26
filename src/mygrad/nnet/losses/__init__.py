from .margin_ranking_loss import margin_ranking_loss
from .multiclass_hinge import multiclass_hinge
from .negative_log_likelihood import negative_log_likelihood
from .softmax_crossentropy import softmax_crossentropy

__all__ = [
  "margin_ranking_loss",
  "multiclass_hinge",
  "negative_log_likelihood",
  "softmax_crossentropy",
]
