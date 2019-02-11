"""QHM and QHAdam for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.training import distribution_strategy_context
from tensorflow.python.training import slot_creator
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

from tensorflow.python.training import optimizer
import tensorflow as tf


class QHMOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Quasi-Hyperbolic Momentum algorithm.
    See [Ma et al., 2019](https://arxiv.org/pdf/1810.06801.pdf)
    """

    def __init__(self, alpha=1.0, beta=0.999, nu=0.7, use_locking=False, name="QHM"):
        """Construct a new Eve optimizer.
        Args:
          alpha1: A Tensor or a floating point value.  
            The learning rate.
          beta1: A float value or a constant float tensor.
            The exponential decay rate for the 1st moment estimates.
          beta2: A float value or a constant float tensor.
            The exponential decay rate for the 2nd moment estimates.
          beta3: A float value or a constant float tensor.
            The exponential decay rate for computing relative change.
          epsilon: A float value or a constant float tensor.
            A small constant for numerical stability. 
          use_locking: If True use locks for update operations.
          name: Optional name for the operations created when applying gradients.
            Defaults to "Eve".
        @compatibility(eager)
        When eager execution is enabled, `alpha1`, `beta1`, `beta2`, `beta3`, `clip_value`, 
        and `epsilon` can each be a callable that takes no arguments and returns the
        actual value to use. This can be useful for changing these values across
        different invocations of optimizer functions.
        @end_compatibility
        """
        super(QHMOptimizer, self).__init__(use_locking, name)
        self.alpha = alpha
        self.beta = beta
        self.nu = nu

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for var in var_list:
            self._zeros_slot(var, "g", self._name)

    def _prepare(self):
        self.alpha = ops.convert_to_tensor(
            value=self._call_if_callable(self.alpha),
            name="alpha"
        )
        self.beta = ops.convert_to_tensor(
            value=self._call_if_callable(self.beta),
            name="beta"
        )
        self.nu = ops.convert_to_tensor(
            value=self._call_if_callable(self.nu),
            name="nu"
        )

    def _apply_dense(self, grad, var):

        g = self.get_slot(var, "g")

        alpha = math_ops.cast(self.alpha, var.dtype.base_dtype)
        beta = math_ops.cast(self.beta, var.dtype.base_dtype)
        nu = math_ops.cast(self.nu, var.dtype.base_dtype)

        g = g.assign(beta * g + (1 - beta) * grad)
        var = var.assign_sub(alpha * ((1 - nu) * grad + nu * g))

        return var


class QHAdamOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Quasi-Hyperbolic Adam algorithm.
    See [Ma et al., 2019](https://arxiv.org/pdf/1810.06801.pdf)
    """

    def __init__(self, alpha=1e-3, beta1=0.9, beta2=0.999, nu1=1.0, nu2=1.0,
                 epsilon=1e-8, use_locking=False, name="QHAdam"):
        """Construct a new Eve optimizer.
        Args:
          alpha1: A Tensor or a floating point value.  
            The learning rate.
          beta1: A float value or a constant float tensor.
            The exponential decay rate for the 1st moment estimates.
          beta2: A float value or a constant float tensor.
            The exponential decay rate for the 2nd moment estimates.
          beta3: A float value or a constant float tensor.
            The exponential decay rate for computing relative change.
          epsilon: A float value or a constant float tensor.
            A small constant for numerical stability. 
          use_locking: If True use locks for update operations.
          name: Optional name for the operations created when applying gradients.
            Defaults to "Eve".
        @compatibility(eager)
        When eager execution is enabled, `alpha1`, `beta1`, `beta2`, `beta3`, `clip_value`, 
        and `epsilon` can each be a callable that takes no arguments and returns the
        actual value to use. This can be useful for changing these values across
        different invocations of optimizer functions.
        @end_compatibility
        """
        super(QHAdamOptimizer, self).__init__(use_locking, name)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.nu1 = nu1
        self.nu2 = nu2
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable. Sort the var_list to make sure this device is consistent across
        # workers (these need to go on the same PS, otherwise some updates are
        # silently ignored).
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(
            initial_value=self.beta1,
            name="beta1_power",
            colocate_with=first_var
        )
        self._create_non_slot_variable(
            initial_value=self.beta2,
            name="beta2_power",
            colocate_with=first_var
        )
        # Create slots for the first and second moments.
        for var in var_list:
            self._zeros_slot(var, "g", self._name)
            self._zeros_slot(var, "s", self._name)

    def _prepare(self):
        self.alpha = ops.convert_to_tensor(
            value=self._call_if_callable(self.alpha),
            name="alpha"
        )
        self.beta1 = ops.convert_to_tensor(
            value=self._call_if_callable(self.beta1),
            name="beta1"
        )
        self.beta2 = ops.convert_to_tensor(
            value=self._call_if_callable(self.beta2),
            name="beta2"
        )
        self.nu1 = ops.convert_to_tensor(
            value=self._call_if_callable(self.nu1),
            name="nu1"
        )
        self.nu2 = ops.convert_to_tensor(
            value=self._call_if_callable(self.nu2),
            name="nu2"
        )
        self.epsilon = ops.convert_to_tensor(
            value=self._call_if_callable(self.epsilon),
            name="epsilon"
        )

    def _apply_dense(self, grad, var):

        with ops.init_scope():
            graph = None if context.executing_eagerly() else ops.get_default_graph()
            beta1_power = self._get_non_slot_variable("beta1_power", graph=graph)
            beta2_power = self._get_non_slot_variable("beta2_power", graph=graph)

        g = self.get_slot(var, "g")
        s = self.get_slot(var, "s")

        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        alpha = math_ops.cast(self.alpha, var.dtype.base_dtype)
        beta1 = math_ops.cast(self.beta1, var.dtype.base_dtype)
        beta2 = math_ops.cast(self.beta2, var.dtype.base_dtype)
        nu1 = math_ops.cast(self.nu1, var.dtype.base_dtype)
        nu2 = math_ops.cast(self.nu2, var.dtype.base_dtype)
        epsilon = math_ops.cast(self.epsilon, var.dtype.base_dtype)

        g = g.assign(beta1 * g + (1 - beta1) * grad)
        g_hat = g / (1 - beta1_power)

        s = s.assign(beta2 * s + (1 - beta2) * grad ** 2)
        s_hat = s / (1 - beta2_power)

        var = var.assign_sub(alpha * (
            ((1 - nu1) * grad + nu1 * g_hat) /
            ((((1 - nu2) * grad ** 2 + nu2 * s_hat) ** 0.5) + epsilon)
        ))

        return var

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            with ops.init_scope():
                graph = None if context.executing_eagerly() else ops.get_default_graph()
                beta1_power = self._get_non_slot_variable("beta1_power", graph=graph)
                beta2_power = self._get_non_slot_variable("beta2_power", graph=graph)
            with ops.colocate_with(beta1_power):
                update_beta1_power = beta1_power.assign(
                    value=beta1_power * self.beta1,
                    use_locking=self._use_locking
                )
                update_beta2_power = beta2_power.assign(
                    value=beta2_power * self.beta2,
                    use_locking=self._use_locking
                )
        return control_flow_ops.group(
            *(update_ops + [update_beta1_power, update_beta2_power]),
            name=name_scope
        )
