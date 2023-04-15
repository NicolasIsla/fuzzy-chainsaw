"""
Módulo para crear simulación temporal
"""
import numpy as np
import matplotlib.pyplot as plt

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
        self.resultado = np.zeros(self.N)
    
    # Ejecuta simulación básica.
    def run_f(self):
        for i, t in enumerate(self.tiempo):
            self.resultado[i] = self.f(t+self.to)

    # Genera gráfico con resultados de simulación.
    def plot_f(self):
        plt.figure(figsize=(7,5))
        plt.plot(self.tiempo, self.resultado)
        plt.xlabel("Tiempo (segundos)")
        plt.ylabel("f(t)")
        plt.title("Gráfico")
        plt.show()

class Simulacion_CLD(Simulacion):
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

        mensaje = f"""Se creó el simulador para el controlador con los siguientes parámetros.
        duración = {duracion} [s]
        frec_muestreo = {frec_muestreo} [Hz]
        rango_EP = {rango_EP}
        rango_TP = {rango_TP}
        rango_deltaH = {rango_EP}

        La variable a controlar es P={K}*H usando lógica difusa.
        La presión objetivo es P_obj={P_obj} [Pa], y la presión inicial es P_inicial={P_obj} [Pa]
        """
        if verbose is True: print(mensaje)

    def definir_rangos(self, rango_EP, rango_TP, rango_deltaH):
        """
        
        """
        self.rango_EP = rango_EP
        self.rango_TP = rango_TP
        self.rango_deltaH = rango_deltaH
    
    def definir_presion_objetivo(self, P_obj):
        self.P_obj = P_obj


if __name__ == "__main__":
    frec = 5  # Hz
    frec_muestreo = 100 # Hz
    f = lambda t: np.sin(2*np.pi*frec*t)
    sim = Simulacion_CLD(2, frec_muestreo, f)
    sim.run_f()
    sim.plot_f()
