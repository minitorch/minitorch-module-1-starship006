"""Collection of the core mathematical operators used throughout the code base."""


# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
import math
from typing import TypeVar, Iterable, Callable


def mul(a : float, b : float) -> float:
    return a * b
# - id
T = TypeVar('T')
def id(a : T) -> T:
    return a
# - add
def add(a : float, b : float) -> float:
    return a + b

# - neg
def neg(a : float) -> float:
    return -1.0 * a

# - lt
def lt(a : float, b : float) -> bool:
    return a < b
# - eq
def eq(a : float, b : float) -> bool:
    return a == b

# - max
def max(a : float, b : float) -> float:
    if a > b:
        return a
    else:
        return b

# - is_close
def is_close(a : float, b : float):
    return abs(a - b) < 1e-2

# - sigmoid
def sigmoid(x : float):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))

# - relu
def relu(x : float):
    return max(x, 0.0)

# - log
def log(x : float):
    return math.log(x)

# - exp
def exp(x : float):
    return math.exp(x)

# - log_back
def log_back(x : float, arg: float):
    return arg / x

# - inv
def inv(x : float):
    return 1 / x

# - inv_back
def inv_back(x : float, arg : float):
    return - (x ** -2) * arg

# - relu_back
def relu_back(x : float, arg: float):
    if x <= 0:
        return 0
    else:
        return arg

def sig_back(x : float, arg : float):
    sigx = sigmoid(x)
    return (sigx * (1 - sigx)) * arg


# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def map(elements: list[T], operation : Callable) -> list[T]:
    newiter = []
    for e in elements:
        newiter.append(operation(e))
    return newiter

def zipWith(elemOne : list, elemTwo : list, comb : Callable) -> list:
    return [comb(one, two) for one, two in zip(elemOne, elemTwo)]

def reduce(elements : list[T], func : Callable) -> T:
    elem = None
    for e in elements:
        if elem is None:
            elem = e
        else:
            elem = func(elem, e)
    assert elem is not None  
    return elem
    
def negList(numbers: list[float]) -> list[float]:
    return map(numbers, lambda x: -x)

# 2. addLists - Add corresponding elements from two lists using zipWith
def addLists(list1: list[int], list2: list[int]) -> list[int]:
    return zipWith(list1, list2, lambda x, y: x + y)

# 3. sum - Sum all elements in a list using reduce
def sum(numbers: list[int]) -> int:
    if len(numbers) == 0:
        return 0
    return reduce(numbers, lambda x, y: x + y)

# 4. prod - Calculate the product of all elements in a list using reduce
def prod(numbers: list[int]) -> int:
    if len(numbers) == 0:
        return 1
    return reduce(numbers, lambda x, y: x * y)


        