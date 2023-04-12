"""
Operaciones importantes
"""

# Ecuación de la recta donde m es la pendiente y (xo, yo) es un punto por donde pasa la recta.
def ecuacion_recta(x, m, xo, yo):
    return m*(x-xo)+yo

# Función de pertenencia trapezoidal. Define la pertenencia del crisp value x en el conjunto difuso A.
def u_A(A, x):
    if x < A[0] or x > A[3]: return 0
    elif x < A[1]: return ecuacion_recta(x, (1-0)/(A[1]-A[0]), A[0], 0)
    elif x > A[2]: return ecuacion_recta(x, (0-1)/(A[3]-A[2]), A[2], 1)
    else: return 1

# Operador difuso de_A.
def de_A(A, B):
    return (min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3]))