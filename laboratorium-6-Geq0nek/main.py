import numpy as np
import scipy as sp
from scipy import linalg
from  datetime import datetime
import pickle

from typing import Union, List, Tuple

def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray):
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b

      Parameters:
      A: macierz A (m,n) zawierająca współczynniki równania
      x: wektor x (n,) zawierający rozwiązania równania
      b: wektor b (m,) zawierający współczynniki po prawej stronie równania

      Results:
      (float)- wartość normy residuom dla podanych parametrów
      """
    if isinstance(A, np.ndarray) and isinstance(x, np.ndarray) and isinstance(b, np.ndarray):
      r = np.linalg.norm(b - A@x)
      return r
    return None

def is_diagonaly_dominant(A: np.ndarray | sp.sparse._csc.csc_array):
    """Funkcja sprawdzająca czy podana macierz jest diagonalnie zdominowana
      Jeśli A jest innego typu, lub nie jest kwadratowa, funkcja powinna zwracać None
      
      Parameters:
      A: macierz A (m,m) podlegająca weryfikacji

      Results:
      (bool)- True jeśli macierz jest diagonalnie zdominowana, w przeciwnym wypadku False
      """
    if isinstance(A, (np.ndarray, sp.sparse._csc.csc_array)) and len(A.shape) == 2 and A.shape[0] == A.shape[1]:

      diag = abs(A).diagonal()
      row = np.sum(abs(A), axis=1) - diag

      if all(diag > row):
        return True
      return False
    return None
      
