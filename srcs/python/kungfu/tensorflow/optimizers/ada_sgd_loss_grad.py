import tensorflow as tf
from kungfu._utils import map_maybe
from kungfu.tensorflow.compat import _tf_assign, _tf_hook
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
from kungfu.tensorflow.ops import (counter, current_cluster_size,
                                   group_all_reduce, all_reduce)

from kungfu.tensorflow.optimizers.core import (_create_kungfu_keras_optimizer, _create_kungfu_optimizer,
                   _KungFuAlgorithm)


def AdaptiveSGDLossGradOptimizer(optimizer,
                         loss,
                         loss_threshold,
                         alpha=0.1,
                         name="AdaptiveSGDLossOptimizer",
                         use_locking=False,
                         with_keras=False):

    algo = _AdaptiveSGD(loss, loss_threshold, alpha)
    if not with_keras:
        return _create_kungfu_optimizer(optimizer, algo, name, use_locking)
    else:
        return _create_kungfu_keras_optimizer(optimizer, algo)


class _AdaptiveSGD(_KungFuAlgorithm):
    def __init__(self, loss, loss_grad_threshold, alpha):
        self._num_workers = current_cluster_size()
        self._loss = loss
        self._loss_grad_threshold = loss_grad_threshold
        self._changed = tf.get_variable("changed", initializer=1, trainable=False)
        self._loss_grad_ma = tf.get_variable("loss_grad_moving_average", initializer=0.0, trainable=False)
        self._last_loss = tf.get_variable("last_last", [], trainable=False)
        self._alpha = alpha
        self._global_step = tf.train.get_or_create_global_step()

        # log moving average of loss grad
        tf.summary.scalar("Loss_grad_moving_average", self._loss_grad_ma)

    def change(self):
        print_op= tf.cond(
            tf.math.equal(self._changed, 1),
            lambda: tf.print("Change step", self._global_step),
            lambda: tf.no_op())
        with tf.control_dependencies([print_op]):
            return tf.assign(self._changed, 0)

    def moving_average(self, beta=0.1):
        avg_loss = tf.math.divide(all_reduce(self._loss), self._num_workers)
        loss_grad = tf.math.subtract(avg_loss, self._last_loss)
        moving_avg_op = tf.cond(
            tf.math.equal(self._global_step, 0),
            lambda: tf.identity(self._loss_grad_ma),
            lambda: tf.cond(
                tf.math.greater(self._global_step, 1),
                lambda: tf.assign(self._loss_grad_ma, (1 - beta) * self._loss_grad_ma + beta * loss_grad),
                lambda: tf.assign(self._loss_grad_ma, loss_grad)))

        with tf.control_dependencies([moving_avg_op]):
            return tf.assign(self._last_loss, avg_loss)

    def eval_change(self):
        lgma_op = self.moving_average()
        with tf.control_dependencies([lgma_op]):
            return tf.cond(
                tf.math.less(self._loss_grad_ma, self._loss_grad_threshold),
                lambda: tf.identity(self._changed),
                lambda: self.change())

    def _ssgd(self, apply_grads_func, gradients, variables, **kwargs):
        sum_grads = group_all_reduce(gradients)
        avg_grads = map_maybe(lambda g: g / self._num_workers, sum_grads)

        # We need to re-zip gradients and variables as grads_and_vars can be only unzipped once.
        grads_and_vars = zip(avg_grads, variables)

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
        g, v = list(zip(*grads_and_vars))

        lgma_op = self.eval_change()

        with tf.control_dependencies([lgma_op]):
            # return tf.cond(
            #     tf.math.equal(self._changed, 1),
            #     lambda: self._sma(apply_grads_func, g, v, **kwargs),
            #     lambda: self._ssgd(apply_grads_func, g, v, **kwargs))
            return self._ssgd(apply_grads_func, g, v, **kwargs)

class AdaSGDHook(_tf_hook):
    def __init__(self):
        super(AdaSGDHook, self).__init__()
        self._changed_yet = False
    
    def begin(self):
        from kungfu.tensorflow.ops import broadcast
        self._ops = [tf.assign(v, broadcast(v)) for v in tf.global_variables()]
        self._changed = tf.get_default_graph().get_tensor_by_name("changed:0")

    def after_run(self, run_context, run_values):
        change = run_context.session.run(self._changed)
        if change == 0 and not self._changed_yet:
            run_context.session.run(self._ops)
            self._changed_yet = True
