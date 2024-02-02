##
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt

from typing import Union, List, Tuple

def linear_least_squares(x:np.ndarray,y:np.ndarray)-> np.ndarray:
    """Funkcja do obliczania współczynników liniowej aproksymacji metodą najmniejszych kwadratów,
    
    Parameters:
    x(np.ndarray): wartość x punktu danych
    y(np.ndarray): wartość y punktu danych

    Results:
    np.ndarray: wektor współczynników aproksymacji. 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and len(x) > 1 and len (y) > 1 and x.shape == y.shape:
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x_square = np.sum(x**2)

        a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_square - sum_x**2)
        b = (sum_y - a * sum_x) / n
        
        return np.array([a, b])
    else:
        return None



def chebyshev_nodes(n:int,interval:tuple)-> np.ndarray:
    """Funkcja tworząca wektor zawierający węzły czybyszewa w postaci wektora (n) dla zadanego przedziału i organizująca wynik posortowany od najmniejszego do największego węzła
    
    Parameters:
    n(int): ilość węzłów Czebyszewa. Wartość musi być większa od 0.
    interval (tuple): Przedział, na którym mają być wygenerowane węzły (początek, koniec).
     
    Results:
    np.ndarray: posortowany wektor węzłów Czybyszewa o rozmiarze (n). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(n, int) and n > 0 and isinstance(interval, tuple) and len(interval) == 2 and interval[0] < interval[1]:
        k = np.arange(1, n+1, dtype=int)
        x_k = 1/2 * (interval[0] + interval[1]) + 1/2 * (interval[1] - interval[0]) * np.cos(((2*k - 1)/(2*n))*np.pi)
        x_k = np.sort(x_k)
        return x_k
    else:
        return None