import numpy as np
from typing import Union, Callable


def solve_euler(fun: Callable, t_span: np.array, y0: np.array):
    ''' 
    Funkcja umożliwiająca rozwiązanie układu równań różniczkowych z wykorzystaniem metody Eulera w przód.
    
    Parameters:
    fun: Prawa strona równania. Podana funkcja musi mieć postać fun(t, y). 
    Tutaj t jest skalarem i istnieją dwie opcje dla ndarray y: Może mieć kształt (n,); wtedy fun musi zwrócić array_like z kształtem (n,). 
    Alternatywnie może mieć kształt (n, k); wtedy fun musi zwrócić tablicę typu array_like z kształtem (n, k), tj. każda kolumna odpowiada jednej kolumnie w y. 
    t_span: wektor czasu dla którego ma zostać rozwiązane równanie
    y0: warunke początkowy równania o wymiarze (n,)
    Results:
    (np.array): macierz o wymiarze (n,m) zawierająca w wkolumnach kolejne rozwiązania fun w czasie t_span. W przypadku błędnych danych wejściowych powinna zwracać None

    '''
    try:
        h = t_span[1] - t_span[0]
        y = np.zeros((len(t_span), len(y0)))
        y[0] = y0

        if callable(fun):
            for i in range(0, len(t_span) - 1):
                y[i + 1] = y[i] + h * fun(t_span[i], y[i])
        elif isinstance(fun, np.ndarray) and all(callable(f) for f in fun):
            for i in range(0, len(t_span) - 1):
                dydt = np.array([f(t_span[i], y[i]) for f in fun])
                y[i + 1] = y[i] + h * dydt
        else:
            raise ValueError

        return y  
    except (ValueError, TypeError):
        return None

def arenstorf(t, x: np.array):
    '''     
    Parameters:
    t: czas
    x: wektor stanu 
    Results:
    (np.array): wektor pochodnych stanu
    '''
    try:
        if isinstance(t, str) or t == None:
            return None
        
        mi = 0.012277471
        mi_prim = 1 - mi
        D1 = ((x[0] + mi)**2 + x[2]**2)**(3 / 2)
        D2 = ((x[0] - mi_prim)**2 + x[2]**2)**(3 / 2)
        x1 = x[1]
        x2 = x[0] + 2 * x[3] - mi_prim * (x[0] + mi) / D1 - mi * (x[0] - mi_prim) / D2
        x3 = x[3]
        x4 = x[2] - 2 * x[1] - mi_prim * x[2] / D1 -  mi * x[2] / D2
        return np.array([x1, x2, x3, x4])
        
    except(ValueError, TypeError):
        return None
