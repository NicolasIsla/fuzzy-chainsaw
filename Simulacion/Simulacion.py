"""
Módulo para crear simulación temporal
"""
import numpy as np
import matplotlib.pyplot as plt

class Simulacion:
    def __init__(self, duracion, frec_muestreo, f, to=0):
        self.duracion = duracion
        self.frec_muestreo = frec_muestreo
        self.f = f
        self.to = to
        self.N = int(duracion * frec_muestreo)
        self.tiempo = np.linspace(to, duracion+to, self.N)
        self.resultado = np.zeros(self.N)
    
    # Ejecuta simulación.
    def run_f(self):
        for i, t in enumerate(self.tiempo):
            self.resultado[i] = self.f(t+self.to)
    
    # Genera gráfico con resultados de simulación.
    def plot_f(self):
        plt.figure(figsize=(7,5))
        plt.plot(self.tiempo, self.resultado)
        plt.show()

class Simulacion_CLD(Simulacion):
    def __init__(self, rango_EP, rango_TP, rango_deltaH, P_obj):
        super().__init__()
        self.rango_EP = rango_EP
        self.rango_TP = rango_TP
        self.rango_deltaH = rango_deltaH
        self.P_obj = P_obj

    def definir_rangos(self, rango_EP, rango_TP, rango_deltaH):
        self.rango_EP = rango_EP
        self.rango_TP = rango_TP
        self.rango_deltaH = rango_deltaH
    
    def definir_presion_objetivo(self, P_obj):
        self.P_obj = P_obj

if __name__ == "__main__":
    frec = 5  # Hz
    frec_muestreo = 10 # Hz
    f = lambda t: np.sin(2*np.pi*frec*t)
    sim = Simulacion(2, frec_muestreo, f)
    sim.run_f()
    sim.plot_f()