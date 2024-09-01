from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    left_point = vals[arg] - epsilon
    new_vals_left = list(vals)
    new_vals_left[arg] = left_point
    
    right_point = vals[arg] + epsilon
    new_vals_right = list(vals)
    new_vals_right[arg] = right_point
    
    return (f(*new_vals_right) - f(*new_vals_left)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = []
    order = []

    def visit(var: Variable):
        if var.unique_id in visited or var.is_constant():
            return
        visited.append(var.unique_id)
        for parent in var.parents:
            visit(parent)
        order.append(var)

    visit(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    ordered_vars = topological_sort(variable)
    var_to_grads = {variable.unique_id: deriv}
    for var_to_backprop in ordered_vars:
        d_output = var_to_grads[var_to_backprop.unique_id]
        if var_to_backprop.is_leaf():
            var_to_backprop.accumulate_derivative(d_output)
        else:
            vars_to_grads = var_to_backprop.chain_rule(d_output)
            for parent_var, grad in vars_to_grads:
                assert parent_var is not var_to_backprop, "what"
                if parent_var.unique_id in var_to_grads:
                    var_to_grads[parent_var.unique_id] += grad
                else:
                    var_to_grads[parent_var.unique_id] = grad
        
        


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
