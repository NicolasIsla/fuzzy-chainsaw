"""
Operaciones importantes. Idealmente se usan todas en orden.
"""

import numpy as np

"""
Funciones para difusión de crisp values.
"""

def saturador(x, rango):
    """
    Satura el valor x entre el rango deseado. Primero se desea saturar los valores
    de EP y TP ya que los sensores no pueden ir más allá de estos rangos.
    """
    return rango[1] if x > rango[1] else rango[0] if x < rango[0] else x

def estandarizar_valor(valor, rango_inicial = [-15, 15], rango_final = [-1, 1]):
    """
    Normaliza valores de un rango inicial a un rango final (ej. de [-15,15] a [-1,1}]).
    Con ello se pueden definir los grados de pertenencia correctamente.
    """
    return ((valor - rango_inicial[0]) / (rango_inicial[1] - rango_inicial[0])) * (rango_final[1] - rango_final[0]) + rango_final[0]

def u_A(A, x, pertenencia = [0, 1, 1, 0]):
    """
    Calcula el grado de pertenencia de un valor dado x en un conjunto difuso A. Usa una
    interpolación para calcular la curva. Se asume que x pertenece al rango admisible.
    Nota: funciona tanto como para valores puntuales (map lambda) como para dominios enteros.
    """
    return np.interp(x, A, pertenencia)

def de_A(A, B):
    """
    Operador difuso de_A, que entrega la envoltura convexa de ambos conjuntos difusos
    trapezoidales.
    """
    return (min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3]))

"""
Funciones para desdifusión de valores difusos.
"""

def regla_min(corte, pertenencia_original):
    """
    Satura los valores de pertenencia de un conjunto en el corte. Util para
    realizar la desdifusión.
    """
    filtro = pertenencia_original <= corte
    return pertenencia_original*(filtro)+(~filtro)*corte


def desdifusor(mapa_reglas, activacion, metodo_desdifusion='CG', N_muestras=200):
    """
    Realiza la desdifusión del tipo especificada en vase a los valores de activación
    calculados y el mapa de reglas.
    """
    if metodo_desdifusion == 'CG':
        """
        Regla de centro de gravedad, recive las reglas y el grado de activacion de cada una
        Retorna el valor perteneciente al centro de gravedad
        """
        dominio = np.linspace(-1, 1, N_muestras)
        # Salida de cada regla con la saturacion aplicada
        salida = np.zeros((len(mapa_reglas), N_muestras), dtype=np.float64)
        for i, DELTA_H in enumerate(mapa_reglas.values()):
            # se muestre el conjunto de salida
            conjunto_muestreado = u_A(DELTA_H, dominio)
            # se satura el valor con el mínimo de las entradas
            salida[i] = regla_min(activacion[i], conjunto_muestreado)
        # se toma la cobertura de las 17 salidas
        covertura = np.maximum.reduce(salida)
        # retorno del centro de gravedad: crisp value delta_h.
        if np.sum(covertura) != 0:
            return np.sum(dominio*covertura)/(np.sum(covertura))
        else:
            return 0

# Test
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dominio = np.linspace(-15, 15, 2000)
    plt.plot(dominio, list(map(lambda x: estandarizar_valor(x, [-15, 15], [-1, 1]), dominio)))
    plt.show()

    dominio = np.linspace(-1, 1, 2000)
    plt.plot(dominio, list(map(lambda x: u_A((-0.5, -0.25, 0.25, 0.5), x), dominio)))
    plt.show()

    dominio = np.linspace(-1, 1, 2000)
    plt.plot(dominio, u_A((-0.5, -0.25, 0.25, 0.5), dominio))
    plt.show()

    dominio = np.linspace(-40, 40, 2000)
    plt.plot(dominio, list(map(lambda x: saturador(x, [-15, 15]), dominio)))
    plt.show()

    