""" Stateless Dropout for Uncertainty Measure 
	which take seed as a tensor input """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

from tensorflow.python.util import deprecation
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import tf_logging as logging

from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

import tensorflow as tf


@deprecation.deprecated_args(None, "Please use `rate` instead of `keep_prob`. "
                             "Rate should be set to `rate = 1 - keep_prob`.",
                             "keep_prob")
def stateless_dropout(x, keep_prob=None, noise_shape=None, seed=None, name=None,
            rate=None):
  """Computes dropout.
  For each element of `x`, with probability `rate`, outputs `0`, and otherwise
  scales up the input by `1 / (1-rate)`. The scaling is such that the expected
  sum is unchanged.
  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
  and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
  kept independently and each row and column will be kept or not kept together.
  Args:
    x: A floating point tensor.
    keep_prob: (deprecated) A deprecated alias for `(1-rate)`.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A shape [2] integer Tensor of seeds to the random number generator.
    name: A name for this operation (optional).
    rate: A scalar `Tensor` with the same type as `x`. The probability that each
      element of `x` is discarded.
  Returns:
    A Tensor of the same shape of `x`.
  Raises:
    ValueError: If `rate` is not in `[0, 1)` or if `x` is not a floating
      point tensor.
  """
  try:
    keep = 1. - keep_prob if keep_prob is not None else None
  except TypeError:
    raise ValueError("keep_prob must be a floating point number or Tensor "
                     "(got %r)" % keep_prob)

  rate = deprecation.deprecated_argument_lookup(
      "rate", rate,
      "keep_prob", keep)

  if rate is None:
    raise ValueError("You must provide a rate to dropout.")

  return dropout_v2(x, rate, noise_shape=noise_shape, seed=seed, name=name)

def dropout_v2(x, rate, noise_shape=None, seed=None, name=None):
  """Computes dropout.
  With probability `rate`, drops elements of `x`. Input that are kept are
  scaled up by `1 / (1 - rate)`, otherwise outputs `0`.  The scaling is so that
  the expected sum is unchanged.
  **Note:** The behavior of dropout has changed between TensorFlow 1.x and 2.x.
  When converting 1.x code, please use named arguments to ensure behavior stays
  consistent.
  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
  and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
  kept independently and each row and column will be kept or not kept together.
  Args:
    x: A floating point tensor.
    rate: A scalar `Tensor` with the same type as x. The probability
      that each element is dropped. For example, setting rate=0.1 would drop
      10% of input elements.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A shape [2] integer Tensor of seeds to the random number generator.
    name: A name for this operation (optional).
  Returns:
    A Tensor of the same shape of `x`.
  Raises:
    ValueError: If `rate` is not in `(0, 1]` or if `x` is not a floating point
      tensor.
  """
  with ops.name_scope(name, "dropout", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    if not x.dtype.is_floating:
      raise ValueError("x has to be a floating point tensor since it's going to"
                       " be scaled. Got a %s tensor instead." % x.dtype)
    if isinstance(rate, numbers.Real):
      if not (rate >= 0 and rate < 1):
        raise ValueError("rate must be a scalar tensor or a float in the "
                         "range [0, 1), got %g" % rate)
      if rate > 0.5:
        logging.log_first_n(
            logging.WARN, "Large dropout rate: %g (>0.5). In TensorFlow "
            "2.x, dropout() uses dropout rate instead of keep_prob. "
            "Please ensure that this is intended.", 5, rate)

    # Early return if nothing needs to be dropped.
    if isinstance(rate, numbers.Real) and rate == 0:
      return x
    if context.executing_eagerly():
      if isinstance(rate, ops.EagerTensor):
        if rate.numpy() == 0:
          return x
    else:
      rate = ops.convert_to_tensor(
          rate, dtype=x.dtype, name="rate")
      rate.get_shape().assert_is_compatible_with(tensor_shape.scalar())

      # Do nothing if we know rate == 0
      if tensor_util.constant_value(rate) == 0:
        return x

    noise_shape = _get_noise_shape(x, noise_shape)
    # Sample a uniform distribution on [0.0, 1.0) and select values larger than
    # rate.
    #
    # NOTE: Random uniform actually can only generate 2^23 floats on [1.0, 2.0)
    # and subtract 1.0.
    random_tensor = stateless_random_ops.stateless_random_uniform(noise_shape, 
                      seed=seed, dtype=x.dtype) if seed is not None \
                    else random_ops.random_uniform(noise_shape, dtype=x.dtype)
    keep_prob = 1 - rate
    scale = 1 / keep_prob
    # NOTE: if (1.0 + rate) - 1 is equal to rate, then we want to consider that
    # float to be selected, hence we use a >= comparison.
    keep_mask = random_tensor >= rate
    ret = x * scale * math_ops.cast(keep_mask, x.dtype)
    if not context.executing_eagerly():
      ret.set_shape(x.get_shape())
    return ret


def _get_noise_shape(x, noise_shape):
  # If noise_shape is none return immediately.
  if noise_shape is None:
    return array_ops.shape(x)

  try:
    # Best effort to figure out the intended shape.
    # If not possible, let the op to handle it.
    # In eager mode exception will show up.
    noise_shape_ = tensor_shape.as_shape(noise_shape)
  except (TypeError, ValueError):
    return noise_shape

  if x.shape.dims is not None and len(x.shape.dims) == len(noise_shape_.dims):
    new_dims = []
    for i, dim in enumerate(x.shape.dims):
      if noise_shape_.dims[i].value is None and dim.value is not None:
        new_dims.append(dim.value)
      else:
        new_dims.append(noise_shape_.dims[i].value)
    return tensor_shape.TensorShape(new_dims)

  return noise_shape