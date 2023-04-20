"""
Módulo para crear simulación temporal
"""
import numpy as np
import matplotlib.pyplot as plt
from utils.fuzzy_utils import *

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
        self.presion = np.zeros(self.N, dtype=np.float64)  # Idealmente es presión
    
    # Ejecuta simulación básica sobre P.
    def run_sim(self):
        for i, t in enumerate(self.tiempo):
            self.presion[i] = self.f(t+self.to)

    # Genera gráfico con resultados de simulación.
    def plot_sim(self, resultado=None):
        plt.figure(figsize=(7,5))
        if resultado is not None: plt.plot(self.tiempo, resultado)
        else: plt.plot(self.tiempo, self.presion)
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
                 mapa_reglas = {}, 
                 rango_EP = [-15, 15], 
                 rango_TP = [-15, 15], 
                 rango_deltaH = [-15, 15], 
                 K = 0.6, 
                 P_obj = 700, 
                 P_inicial = 690,
                 metodo_desdifusion = "CG",
                 verbose = True):
        
        super().__init__(duracion, frec_muestreo)
        self.mapa_reglas = mapa_reglas
        self.rango_EP = rango_EP
        self.rango_TP = rango_TP
        self.rango_deltaH = rango_deltaH
        self.K = K
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
        La presión objetivo es P_obj={P_obj}[Pa], y la presión inicial es P_inicial={P_inicial}[Pa]
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
    
    def step_sim(self, ep, tp, k=0, verbose=False):
        """
        Funcionamiento de un instante de tiempo del controlador. Variables minúsculas
        son crisp values, mientras que variables mayúsculas son difusas.
        """
        # Se saturan entradas
        ep = saturador(ep, self.rango_EP)
        tp = saturador(tp, self.rango_TP)

        # Se normalizan entradas
        ep = estandarizar_valor(ep, self.rango_EP, [-1, 1])
        tp = estandarizar_valor(tp, self.rango_TP, [-1, 1])

        # Se calculan grados de pertenencia por cada regla y se aplica el mínimo,
        # calculando la activación por cada salida.
        activacion = np.zeros(len(self.mapa_reglas), dtype=np.float64)
        for i, (EP_SET, TP_SET) in enumerate(self.mapa_reglas.keys()):
            activacion[i] = min(u_A(EP_SET, ep), u_A(TP_SET, tp))

        # Se aplica desdifusión usando el método especificado.
        delta_h = desdifusor(self.mapa_reglas, activacion, self.metodo_desdifusion)

        # Se recupera la presión a partir de la ecuación entre el cambio de temperatura
        # versus el cambio de presión.
        delta_p = self.K * delta_h

        # Se guarda el valor en un vector de resultados.
        if k != 0: self.presion[k] = self.presion[k-1] + delta_p
        else: self.presion[k] = self.P_inicial + delta_p
        
        if verbose is True:
            print(f"Iteracion {k+1}:")
            print(f"(ep, tp) = {(ep, tp)}")
            print(f"activacion = {activacion}")
            print(f"delta_h = {delta_h}")
            print(f"delta_p = {delta_p}")
            print("="*20)

    # Override de método original
    def run_sim(self, ep_inicial=0, verbose=False):

        # Primera iteración:
        ep = self.P_inicial - self.P_obj
        tp = ep - ep_inicial
        self.step_sim(ep, tp, 0, verbose)

        # Siguientes iteraciones
        for i in range(1, len(self.tiempo)):
            ep = self.presion[i-1] - self.P_obj
            tp = ep - tp  # La información de la variable tp anterior está guardada en si misma.
            self.step_sim(ep, tp, i, verbose)

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
