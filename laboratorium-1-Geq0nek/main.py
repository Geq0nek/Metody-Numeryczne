import numpy as np
import scipy
import math
from typing import Tuple, List


def cylinder_area(r: float,h: float) -> float:
    """Obliczenie pola powierzchni walca. 
    Szczegółowy opis w zadaniu 1.
    
    Parameters:
    r (float): promień podstawy walca 
    h (float): wysokosć walca
    
    Returns:
    float: pole powierzchni walca 
    """
    if r > 0 and h > 0:
       P = 2*(math.pi*r**2 + math.pi*r*h)
       return P
    else:
       return np.nan

def fib(n: int) -> List[int]:
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    Szczegółowy opis w zadaniu 3.
    
    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 
    
    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """
    array_fib = np.array([1, 1])

    if n <= 0 or type(n) != int:
        return None
    elif n == 1:
        return np.array([1])
    elif n == 2:
        return array_fib

    for i in range(2, n):
        next_element = array_fib[-1] + array_fib[-2]
        array_fib = np.append(array_fib, [next_element])

    return np.array(array_fib, dtype=float).reshape([1, n])


def matrix_calculations(a: float) -> Tuple[float, float, float]:
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej 
    na podstawie parametru a.  
    Szczegółowy opis w zadaniu 4.
    
    Parameters:
    a (float): wartość liczbowa 
    
    Returns:
    touple: krotka zawierająca wyniki obliczeń 
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """
    M = np.array([[a, 1, -a], [0, 1, 1], [-a, a, 1]])
    Mdet = np.linalg.det(M)
    Mt = np.transpose(M)

    if Mdet == 0:
        Minv = np.NaN 
    else:
        Minv = np.linalg.inv(M)

    return (Minv, Mt, Mdet)

def custom_matrix(m: int, n: int) -> List[List[int]]:
    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  
    
    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  
    
    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """
    if n <= 0 or m <= 0 or type(n) != int or type(m) != int:
        return None
    else:
       matrix = np.zeros((m, n))

       for i in range(m):
          for j in range(n):
            if i > j:
               matrix[i][j] = i
            else:
               matrix[i][j] = j
    
    return matrix
       

for m in range(3, 7):
    for n in range(3, 7):
        result = custom_matrix(m, n)
        if result is not None:
            print(f"Macierz {m}x{n}:")
            for row in result:
                print(row)
            print()
        else:
            print(f"Niepoprawne wymiary m={m}, n={n}, zwracam None.")
