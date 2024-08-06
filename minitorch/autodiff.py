from queue import Queue

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Dict

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
    # TODO: Implement for Task 1.1.
    vals_ = list(vals[:])
    vals_[arg] += epsilon
    left_f = f(*vals_)
    vals_[arg] -= 2 * epsilon
    right_f = f(*vals_)
    return (left_f - right_f) / (2 * epsilon)
    # raise NotImplementedError("Need to implement for Task 1.1")


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
    # TODO: Implement for Task 1.4.
    root = variable
    # get in-edges
    in_egdes = dict()
    filted_graph: Dict[int, List[Variable]] = dict()
    vis_queue: Queue[Variable] = Queue()
    visited = set()
    visited.add(root.unique_id)
    vis_queue.put(root)
    while not vis_queue.empty():
        cur_node = vis_queue.get()
        cid = cur_node.unique_id
        if cid not in filted_graph:
            filted_graph[cid] = []
        for parent_node in cur_node.parents:
            # not constant
            if not parent_node.is_constant():
                pid = parent_node.unique_id
                filted_graph[cid].append(parent_node)
                # not visited
                if pid not in visited:
                    visited.add(pid)
                    vis_queue.put(parent_node)
                    in_egdes[pid] = 1
                else:
                    in_egdes[pid] += 1
    # topological_sort
    result = list()
    result.append(root)
    vis_queue = Queue()
    vis_queue.put(root)
    while not vis_queue.empty():
        cur_node = vis_queue.get()
        cid = cur_node.unique_id
        for parent_node in filted_graph[cid]:
            pid = parent_node.unique_id
            in_egdes[pid] -= 1
            if in_egdes[pid] == 0 and not parent_node.is_leaf():
                result.append(parent_node)
                vis_queue.put(parent_node)
    return result
    # raise NotImplementedError("Need to implement for Task 1.4")


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    root = variable
    topo_sort = topological_sort(root)
    not_leaf_deriv_map = dict()
    not_leaf_deriv_map[root.unique_id] = deriv
    for cur_node in topo_sort:
        cid = cur_node.unique_id
        cur_deriv = not_leaf_deriv_map[cid]
        backlist = cur_node.chain_rule(cur_deriv)
        for pre_node, pre_deriv in backlist:
            if pre_node.is_leaf():
                pre_node.accumulate_derivative(pre_deriv)
            else:
                pid = pre_node.unique_id
                if pid not in not_leaf_deriv_map:
                    not_leaf_deriv_map[pid] = pre_deriv
                else:
                    not_leaf_deriv_map[pid] += pre_deriv
    # raise NotImplementedError("Need to implement for Task 1.4")


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
