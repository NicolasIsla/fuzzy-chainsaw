"""
Operaciones importantes
"""

""" def ecuacion_recta(x, m, xo, yo):
    return m*(x-xo)+yo

def u_A(A, x):
    if x < A[0] or x > A[3]: return 0
    elif x < A[1]: return ecuacion_recta(x, (1-0)/(A[1]-A[0]), A[0], 0)
    elif x > A[2]: return ecuacion_recta(x, (0-1)/(A[3]-A[2]), A[2], 1)
    else: return 1 """

def u_A(A, x):
    return np.interp(A, x, [0, 1, 1, 0])

def de_A(A, B):
    """
    Operador difuso de_A.
    """
    return (min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3]))

def estandarizar_valor(valor, rango_inicial, rango_final):
    """
    Fuzzifica valores de [-15,15] a [-1,1]
    """
    x_estandarizado = ((valor - rango_inicial[0]) / (rango_inicial[1] - rango_inicial[0])) * (rango_final[1] - rango_final[0]) + rango_final[0]
    return x_estandarizado

# Test
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    dominio = np.linspace(-15, 15, 2000)
    plt.plot(dominio, list(map(lambda x: estandarizar_valor(x, [-15, 15], [-1, 1]), dominio)))
    plt.show()

    dominio = np.linspace(-1, 1, 2000)
    plt.plot(dominio, list(map(lambda x: u_A(x, (-0.5, -0.25, 0.25, 0.5)), dominio)))
    plt.show()