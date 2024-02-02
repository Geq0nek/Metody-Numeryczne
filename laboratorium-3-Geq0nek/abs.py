import numpy as np
import scipy
import pickle

from typing import Union, List, Tuple


def absolut_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu bezwzględnego. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu bezwzględnego,
                                       NaN w przypadku błędnych danych wejściowych
    """

    if isinstance(v, (int, float, List, np.ndarray)) and isinstance(v_aprox, (int, float, List, np.ndarray)):
        
        # obie zmienne są liczbami 
        if isinstance(v, (int, float)) and isinstance(v_aprox, (int, float)):
            return abs(v - v_aprox)
        
        # obie zmienne są listami
        elif isinstance(v, list) and isinstance(v_aprox, list):
            if len(v) == len(v_aprox):
                ans = np.zeros(len(v_aprox), dtype=int)
                for i in range(len(v)):
                    ans[i] = abs(v[i] - v_aprox[i])
                return ans
            else:
                return np.NaN

        # pierwsza zmienna liczbą, druga listą
        elif isinstance(v, (int, float)) and isinstance(v_aprox, list):
            ans = np.zeros(len(v_aprox), dtype=int)
            for i in range(len(v_aprox)):
                ans[i] = abs(v - v_aprox[i])
            return ans

        # pierwsza zmienna listą, druga liczbą
        elif isinstance(v_aprox, (int, float)) and isinstance(v, list):
            ans = np.zeros(len(v), dtype=int)
            for i in range(len(v)):
                ans[i] = abs(v_aprox - v[i])
            return ans

        # obie zmienne wektorami
        elif isinstance(v, np.ndarray) and isinstance(v_aprox, np.ndarray):
            # zip to join two tuples together
            if all((m == n) or (m == 1) or (n == 1) for m, n in zip(v.shape[::-1], v_aprox.shape[::-1])):
                return abs(v - v_aprox)
            else:
                return np.NaN

        # jedna zmienna wekrotem druga liczbą
        elif isinstance(v, np.ndarray) and isinstance(v_aprox, (int, float)) or isinstance(v_aprox, np.ndarray) and isinstance(v, (int, float)):
            return abs(v - v_aprox)

        # jedna zmienna listą druga wektorem
        # przypadek v listą
        elif isinstance(v, list) and isinstance(v_aprox, np.ndarray):
            if len(v) == v_aprox.shape[0]:
                ans = np.zeros(len(v), dtype=int)
                for x in range(len(v)):
                    ans[x] = abs(v[x] - v_aprox[x])
                return ans
            else:
                return np.NaN

        # przypadek v wektorem
        elif isinstance(v_aprox, list) and isinstance(v, np.ndarray):
            if len(v_aprox) == v.shape[0]:
                ans = np.zeros(len(v_aprox), dtype=int)
                for x in range(len(v_aprox)):
                    ans[x] = abs(v[x] - v_aprox[x])
                return ans
            else:
                return np.NaN

    else:
        return np.NaN
        

def relative_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu względnego.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu względnego,
                                       NaN w przypadku błędnych danych wejściowych
    """


    x = absolut_error(v, v_aprox)
    
    if x is np.NaN or (isinstance(v, (int, float)) and v == 0) or isinstance(v, np.ndarray) and not v.any():
        return np.NaN

    elif isinstance(v, np.ndarray):
        return np.divide(x, v)

    elif isinstance(x, np.ndarray) and isinstance(v, list):
        ans = np.zeros(len(v))
        for i in range(len(v)):
            if v[i] == 0:
                return np.NaN
            ans[i] = x[i] / v[i]
        return ans

    else:
        return x / v

