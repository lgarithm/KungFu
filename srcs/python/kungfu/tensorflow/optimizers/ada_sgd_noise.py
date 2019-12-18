import tensorflow as tf
from kungfu._utils import map_maybe
from kungfu.tensorflow.compat import _tf_assign, _tf_hook
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
from kungfu.tensorflow.ops import (current_cluster_size,
                                   group_all_reduce, all_reduce,
                                   fuse, global_noise_scale)

from kungfu.tensorflow.optimizers.core import (_create_kungfu_keras_optimizer, _create_kungfu_optimizer,
                   _KungFuAlgorithm)


def AdaptiveSGDNoiseOptimizer(optimizer,
                         noise_threshold,
                         alpha=0.1,
                         name="AdaptiveSGDNoiseOptimizer",
                         use_locking=False,
                         with_keras=False):

    algo = _AdaptiveSGD(noise_threshold, alpha)
    if not with_keras:
        return _create_kungfu_optimizer(optimizer, algo, name, use_locking)
    else:
        return _create_kungfu_keras_optimizer(optimizer, algo)


class _AdaptiveSGD(_KungFuAlgorithm):
    def __init__(self, noise_threshold, alpha):
        self._num_workers = current_cluster_size()
        self._noise_threshold = noise_threshold
        self._changed = tf.get_variable("changed", initializer=1, trainable=False)
        self._alpha = alpha
        self._global_step = tf.train.get_or_create_global_step()
        self._batch_size = 8
        self._gradient_noise_moving_average = tf.get_variable("gradient_noise_moving_average", initializer=1.0, trainable=False)

    def change(self):
        print_op= tf.cond(
            tf.math.equal(self._changed, 1),
            lambda: tf.print("Change step", self._global_step),
            lambda: tf.no_op())
        with tf.control_dependencies([print_op]):
            return tf.assign(self._changed, 0)

    def filter_out_none(self, l):
        filtered_list = []
        for ele in l:
            if ele is not None:
                filtered_list.append(ele)
        
        return filtered_list


    def grad_noise(self, gradients, avg_gradients, beta=0.1):
        no_none_grads = self.filter_out_none(gradients)
        no_none_avg_grads = self.filter_out_none(avg_gradients)

        global_noise = global_noise_scale(self._batch_size,
                                        self._batch_size * self._num_workers,
                                        fuse(no_none_grads),
                                        fuse(no_none_avg_grads))
        abs_global_noise = tf.math.abs(global_noise)

        return tf.cond(
            tf.math.equal(self._global_step, 0),
            lambda: tf.assign(self._gradient_noise_moving_average, abs_global_noise),
            lambda: tf.assign(self._gradient_noise_moving_average, (1 - beta) * self._gradient_noise_moving_average + beta * abs_global_noise))

    def eval_change(self, noise_scale):
        return tf.cond(
            tf.math.greater(noise_scale, self._noise_threshold),
            lambda: self.change(),
            lambda: tf.identity(self._changed))

    def _ssgd(self, apply_grads_func, gradients, variables, **kwargs):
        sum_grads = group_all_reduce(gradients)
        avg_grads = map_maybe(lambda g: g / self._num_workers, sum_grads)

        noise_scale = self.grad_noise(gradients, avg_grads)
        change_op = self.eval_change(noise_scale)

        # We need to re-zip gradients and variables as grads_and_vars can be only unzipped once.
        grads_and_vars = zip(avg_grads, variables)

        with tf.control_dependencies([change_op]):
            return apply_grads_func(grads_and_vars, **kwargs)

    def _sma(self, apply_grads_func, gradients, variables, **kwargs):
        # It is important to apply model averaging every iteration [2]
        sum_vars = group_all_reduce(variables)
        avg_vars = [v / self._num_workers for v in sum_vars]

        # TODO: Apply momentum to the averaged model [2]
        assign_ops = [
            _tf_assign(v, (1 - self._alpha) * v + self._alpha * avg_v)
            for v, avg_v in zip(variables, avg_vars)
        ]

        # We need to re-zip gradients and variables as grads_and_vars can be only unzipped once.
        grads_and_vars = zip(gradients, variables)

        # We can overlap model averaging and local SGD [2].
        with tf.control_dependencies(assign_ops):
            return apply_grads_func(grads_and_vars, **kwargs)

    def apply_gradients(self, apply_grads_func, grads_and_vars, **kwargs):
        grads, vars = list(zip(*grads_and_vars))

        return tf.cond(
            tf.math.equal(self._changed, 1),
            lambda: self._ssgd(apply_grads_func, grads, vars, **kwargs),
            lambda: self._sma(apply_grads_func, grads, vars, **kwargs))
