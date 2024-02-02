import numpy as np
import scipy
import pickle
import typing
import math
import types
import pickle 
from inspect import isfunction


from typing import Union, List, Tuple

def fun(x):
    return np.exp(-2*x) + x**2 - 1

def dfun(x):
    return -2 * np.exp(-2*x) + 2*x

def ddfun(x):
    return 4 * np.exp(-2*x) + 2

def bisection(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą bisekcji.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if all(isinstance(i, (int, float)) for i in [a, b]) and isinstance(epsilon, float) and isinstance(iteration, int) and callable(f):
       if f(a) * f(b) < 0:
            for i in range (iteration):
                x1 = (a + b) / 2
                f_x1 = f(x1)
                if f(a) * f_x1 < 0:
                    b = x1
                if f(b) * f_x1 < 0:
                    a = x1
                if abs(f_x1) < epsilon:
                    return x1, i
                if abs(b - a) < epsilon:
                    return x1, i
            return x1, iteration
    else:
        return None


def difference_quotient(f: typing.Callable[[float], float],x: Union[int,float], h: Union[int,float]):
    '''Funkcja obliczająca iloaz różnicowy zadanej funkcji
    Parametry:
    
    f - funkcja dla której jest poszukiwane rozwiązanie
    x - argument funkcji la której jest 
    h - krok różnicy wykorzystywanej do wyliczenia ilorazu różnicowego
    
    return:
    diff - wartość ilorazu różnicowego
    
    '''
    try:
        return (f(x+h) - f(x))/h
    except(ValueError, TypeError):
        return None


def newton(f: typing.Callable[[float], float], df: typing.Callable[[float], float], ddf: typing.Callable[[float], float], a: Union[int,float], b: Union[int,float], epsilon: float, iteration: int) -> Tuple[float, int]:
    ''' Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.
    Parametry: 
    f - funkcja dla której jest poszukiwane rozwiązanie
    df - pochodna funkcji dla której jest poszukiwane rozwiązanie
    ddf - druga pochodna funkcji dla której jest poszukiwane rozwiązanie
    a - początek przedziału
    b - koniec przedziału
    epsilon - tolerancja zera maszynowego (warunek stopu)
    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    try:
        if f(a)*f(b) < 0:
            if a < b and df(a) * df(b) > 0 and ddf(a) * ddf(b) > 0:
                for i in range(iteration):
                    a = a - f(a) / df(a)
                    if abs(f(a)) < epsilon:
                        return a, i-1
    except(ValueError, TypeError):
        return None