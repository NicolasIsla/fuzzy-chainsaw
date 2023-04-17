"""
Módulo para crear simulación temporal
"""
import sys    
print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)
import numpy as np
import matplotlib.pyplot as plt
from ..utils.FuzzyOperations import u_A

class Simulacion:
    """
    Crea simulación temporal arbitraria, permite muestrear funciones.
    """
    def __init__(self, duracion, frec_muestreo, f=lambda x: x, to=0):
        self.duracion = duracion
        self.frec_muestreo = frec_muestreo
        self.f = f
        self.to = to
        self.N = int(duracion * frec_muestreo)
        self.tiempo = np.linspace(to, duracion+to, self.N)
        self.P = np.zeros(self.N)  # Idealmente es presión
    
    # Ejecuta simulación básica sobre P.
    def run_sim(self):
        for i, t in enumerate(self.tiempo):
            self.P[i] = self.f(t+self.to)

    # Genera gráfico con resultados de simulación.
    def plot_sim(self, resultado=None):
        plt.figure(figsize=(7,5))
        if resultado is not None: plt.plot(self.tiempo, resultado)
        else: plt.plot(self.tiempo, self.P)
        plt.xlabel("Tiempo (segundos)")
        plt.ylabel("Presión (Pa)")
        plt.title("Gráfico")
        plt.show()

class Simulacion_CLD(Simulacion):
    """
    Crea simulación de un controlador difuso CLD.
    """
    def __init__(self, 
                 duracion, 
                 frec_muestreo, 
                 mapa_reglas={}, 
                 rango_EP=[-15, 15], 
                 rango_TP=[-15, 15], 
                 rango_deltaH=[-15, 15], 
                 K=0.6, 
                 P_obj=700, 
                 P_inicial=690,
                 metodo_desdifusion="CG",
                 verbose=True):
        
        super().__init__(duracion, frec_muestreo)
        self.mapa_reglas = mapa_reglas
        self.rango_EP = rango_EP
        self.rango_TP = rango_TP
        self.rango_deltaH = rango_deltaH
        self.P_obj = P_obj
        self.P_inicial = P_inicial
        self.metodo_desdifusion = metodo_desdifusion

        mensaje = f"""Se creó el simulador para el controlador difuso con los siguientes parámetros.
        duración = {duracion} [s]
        frec_muestreo = {frec_muestreo} [Hz]
        rango_EP = {rango_EP}
        rango_TP = {rango_TP}
        rango_deltaH = {rango_EP}

        La variable a controlar es P={K}*H usando lógica difusa.
        La presión objetivo es P_obj={P_obj}[Pa], y la presión inicial es P_inicial={P_obj}[Pa]
        """
        if verbose is True: print(mensaje)

    def definir_rangos(self, rango_EP, rango_TP, rango_deltaH):
        """
        Permite definir los rangos de operación del controlador difuso.
        """
        self.rango_EP = rango_EP
        self.rango_TP = rango_TP
        self.rango_deltaH = rango_deltaH
    
    def definir_presion_objetivo(self, P_obj):
        """
        Permite definir la presión objetivo.
        """
        self.P_obj = P_obj
    
    def definir_metodo_desdifusion(self, metodo_desdifusion):
        """
        Permite definir el método de desdifusión.
        """
        self.metodo_desdifusion = metodo_desdifusion

    # Override
    def run_sim(self):
        for i, t in enumerate(self.tiempo):
            # Se aplica la lógica difusa sobre los valores
            for i, (EP, TP) in enumerate(self.mapa_reglas.keys()):
                EP_fuzzy = u_A
    



if __name__ == "__main__":
    """
    Como prueba se grafica la función seno.
    """
    frec = 5  # Hz
    frec_muestreo = 100 # Hz
    f = lambda t: np.sin(2*np.pi*frec*t)
    sim = Simulacion_CLD(2, frec_muestreo, f)
    sim.run_sim()
    sim.plot_sim()
