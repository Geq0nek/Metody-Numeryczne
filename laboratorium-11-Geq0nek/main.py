##
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss


from typing import Union, List, Tuple

def rectangular_rule(func, a, b, n):
    """
    Metoda prostokątów do przybliżonego rozwiązania całki oznaczonej.

    :param func: Funkcja, której całkę oznaczoną chcemy przybliżyć.
    :param a: Dolna granica całkowania.
    :param b: Górna granica całkowania.
    :param n: Liczba podprzedziałów (większa liczba n daje dokładniejsze przybliżenie).
    :return: Przybliżona wartość całki oznaczonej.
    """
    try:
        sum = 0
        h = (b - a)/n
        for i in range(n):
            sum += func(a + (i * h))

        return h * sum
    except(ValueError, TypeError):
        return None


def trapezoidal_rule(func, a, b, n):
    """
    Metoda trapezów do przybliżonego rozwiązania całki oznaczonej.

    :param func: Funkcja, której całkę oznaczoną chcemy przybliżyć.
    :param a: Dolna granica całkowania.
    :param b: Górna granica całkowania.
    :param n: Liczba podprzedziałów (większa liczba n daje dokładniejsze przybliżenie).
    :return: Przybliżona wartość całki oznaczonej.
    """
    try:
        sum = 0
        h = (b - a) / n
        for i in range(n):
            k = a + i * h
            sum += 2 * func(k)

        return sum * h / 2

    except(ValueError, TypeError):
        return None



def custom_integration(func, a, b, order):
    """
    Własna funkcja całkująca, wykorzystująca kwadraturę Gaussa-Legendre'a.

    :param func: Funkcja do zintegrowania.
    :param a: Dolna granica całkowania.
    :param b: Górna granica całkowania.
    :param order: Rząd kwadratury.
    :return: Przybliżona wartość całki.
    """
    # Przeskalowanie przedziału do (a, b)
    # Obliczenie wartości funkcji w przeskalowanych punktach
    # Obliczenie całki przy użyciu kwadratury Gaussa-Legendre'a
    try:
        x, w = leggauss(order)
        scaled_x = 0.5 * (b - a) * x + 0.5 * (b + a)
        scaled_values = func(scaled_x)
        integral_value = np.sum(w * scaled_values) * 0.5 * (b - a)
        return integral_value        
    except(ValueError, TypeError):
        return None


