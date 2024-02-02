import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from numpy.core._multiarray_umath import ndarray
from numpy.polynomial import polynomial as P
import pickle

# zad1
def polly_A(x: np.ndarray):
    """Funkcja wyznaczajaca współczynniki wielomianu przy znanym wektorze pierwiastków.
    Parameters:
    x: wektor pierwiastków
    Results:
    (np.ndarray): wektor współczynników
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(x, np.ndarray):
        return P.polyfromroots(x)
    else:
        return None

def roots_20(a: np.ndarray):
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray): wektor współczynników i miejsc zerowych w danej pętli
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(a, np.ndarray):
        if a.shape[0] != 0:
            a = np.array(a, dtype = float)
            r = np.random.random_sample(a.shape[0]) * 1e-10
            a += r
            v = P.polyroots(a)
            return a, v
    else:
        return None

# zad 2

def frob_a(wsp: np.ndarray):
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray, np.ndarray, np. ndarray,): macierz Frobenusa o rozmiarze nxn, gdzie n-1 stopień wielomianu,
    wektor własności własnych, wektor wartości z rozkładu schura, wektor miejsc zerowych otrzymanych za pomocą funkcji polyroots

                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(wsp, np.ndarray):
        frob = np.eye(wsp.shape[0] - 1)
        zero_vert = np.zeros((wsp.shape[0] - 1, 1))
        
        frob = np.concatenate((zero_vert, frob), axis=1)
        frob = np.concatenate((frob, np.reshape(-wsp, (1, wsp.shape[0]))), axis=0)

        return frob, np.linalg.eigvals(frob), scipy.linalg.schur(frob), P.polyroots(wsp)

    return None


# zad 4
def is_nonsingular(A: np.ndarray)->bool:
    """Funkcja sprawdzająca czy podana macierz jest niesingularna.
    
    Parameters:
    A (np.ndarray): macierz nxn do przetestowania 
    
    Results:
    (bool): jeżeli macierz A jest singularna funkcja zwraca False w przeciwnym przypadku zwraca True
    
    Jeżeli dane wejściowe są niepoprawne funkcja zwraca None
    """
    if isinstance(A, np.ndarray) and A.shape[0] == A.shape[1]:
        return False if np.linalg.det(A)==0 else True
    
    return None
    