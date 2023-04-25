"""
Módulo para crear simulación temporal
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.fuzzy_utils import *

class Simulacion:
    """
    Crea simulación temporal arbitraria, permite muestrear funciones.
    """
    def __init__(self, duracion, frec_muestreo, f=lambda x: x, to=0, metodo_desdifusion_label ="CG"):
        self.duracion = duracion
        self.frec_muestreo = frec_muestreo
        self.f = f
        self.to = to
        self.N = int(duracion * frec_muestreo)
        self.tiempo = np.linspace(to, duracion+to, self.N)
        self.presion = np.zeros(self.N, dtype=np.float64)  # Idealmente es presión
        self.deltah = np.zeros(self.N, dtype=np.float64)  # Vector de diferencias de calor adicional para parte opcional
        if metodo_desdifusion_label=="CG":
            self.metodo_desdifusion_label  = "Centro de Gravedad"
        elif metodo_desdifusion_label=="A":
            self.metodo_desdifusion_label  = "Alturas"
        elif metodo_desdifusion_label=="PS":
            self.metodo_desdifusion_label  = "Promedio de Supremos"
        else:
            self.metodo_desdifusion_label = "Indefinido"
    # Ejecuta simulación básica sobre P.
    def run_sim(self):
        for i, t in enumerate(self.tiempo):
            self.presion[i] = self.f(t+self.to)

    # Genera gráfico con resultados de simulación.
    def plot_sim(self, resultado=None, ylabel="Presion (Pa)", preciso = False):
        plt.figure(figsize=(7,5))
        
        if resultado is not None: plt.plot(self.tiempo, resultado, "-o")
        else: plt.plot(self.tiempo, self.presion, "-o")
        plt.xlabel("Tiempo")
        if preciso is True: plt.ylim(700,800)
        plt.hlines(750, 0, 50, colors="red", linestyles="dashed")
        plt.xticks(np.linspace(0,50,11))
        plt.ylabel(ylabel)
        plt.title(f"P(t) vs T, {self.metodo_desdifusion_label}")
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
                 C = 0.5,
                 P_obj = 700, 
                 P_inicial = 690,
                 metodo_desdifusion = "CG",
                 tipo = "lineal",
                 verbose = True):
        
        super().__init__(duracion, frec_muestreo,metodo_desdifusion_label=metodo_desdifusion)
        self.mapa_reglas = mapa_reglas
        self.rango_EP = rango_EP
        self.rango_TP = rango_TP
        self.rango_deltaH = rango_deltaH
        self.K = K
        self.C = C
        self.P_obj = P_obj
        self.P_inicial = P_inicial
        self.metodo_desdifusion = metodo_desdifusion
        self.tipo = tipo

        if tipo not in ("lineal", "exacto"):
            raise NotImplementedError("Método aún no implementado.")

        if verbose is True:
            print("Se creó el simulador para el controlador difuso con los siguientes parámetros.")
            print(f"duración = {duracion} [s]")
            print(f"frec_muestreo = {frec_muestreo} [Hz]")
            print(f"rango_EP = {rango_EP}")
            print(f"rango_TP = {rango_TP}")
            print(f"rango_deltaH = {rango_deltaH}")
            print(f"P_objetivo = {P_obj} [Pa]")
            print(f"P_inicial = {P_inicial} [Pa]")
            if tipo == "lineal": print(f"La variable a controlar es dP(t)={K}*dH(t-1) usando lógica difusa.")
            elif tipo == "exacto": print(f"La variable a controlar es dP(t)={K}*dH(t)+{C}*dP(t-1) usando lógica difusa.")

    def definir_rangos(self, rango_EP, rango_TP, rango_deltaH):
        """
        Permite definir los rangos de operación del controlador difuso.
        """
        self.rango_EP = rango_EP
        self.rango_TP = rango_TP
        self.rango_deltaH = rango_deltaH
    
    def definir_presion_inicial(self, P_inicial):
        """
        Permite definir la presión inicial.
        """
        self.P_inicial = P_inicial
    
    def definir_metodo_desdifusion(self, metodo_desdifusion):
        """
        Permite definir el método de desdifusión.
        """
        self.metodo_desdifusion = metodo_desdifusion
    
    def step_sim(self, ep, tp, dh_anterior, dp_anterior, k=0, verbose=False, participacion=False, nombre="experimento.csv"):
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
            minima_activacion = min(u_A(EP_SET, ep), u_A(TP_SET, tp))
            activacion[i] = minima_activacion

        # Se aplica desdifusión usando el método especificado.
        delta_h = desdifusor(self.mapa_reglas, activacion, self.metodo_desdifusion)

        # Se debe estirar el valor nuevo.
        delta_h = estandarizar_valor(delta_h, [-1, 1], self.rango_deltaH)

        # Se recupera la presión a partir de la ecuación entre el cambio de temperatura
        # versus el cambio de presión.
        if self.tipo == "lineal":
            delta_p = self.K * dh_anterior
        elif self.tipo == "exacto":
            delta_p = self.K * delta_h + self.C * dp_anterior

        # Se guarda el valor en un vector de resultados.
        if k != 0: 
            self.presion[k] = self.presion[k-1] + delta_p
            self.deltah[k] = delta_h
        else: 
            self.presion[k] = self.P_inicial + delta_p
            self.deltah[k] = delta_h

        if participacion is True:

            # Guardado de datos EP vs TP
            if k==0:
                columnas = ["EP", "TP"]
                plano = pd.DataFrame(columns = columnas)
                plano.loc[k]=[ep, tp] 
                
                plano.to_csv(nombre)
            else:
                plano = pd.read_csv(nombre, index_col=0)
                
                plano.loc[k]=[ep, tp] 
                plano.to_csv(nombre)
            
            # Guardado de datos de participación en cada regla
            if k == 0:
                columnas = ["R"+str(i+1) for i in range(len(self.mapa_reglas.keys()))]
                data_participacion = pd.DataFrame(columns = columnas)
                data_participacion.loc[k]=activacion
                
                data_participacion.to_csv(f"regla_participacion_{nombre}")
            else:
                data_participacion = pd.read_csv(f"regla_participacion_{nombre}", index_col=0)
                
                data_participacion.loc[k] = activacion
                data_participacion.to_csv(f"regla_participacion_{nombre}")
        
        if verbose is True:
            print(f"Iteracion {k+1}:")
            print(f"(ep, tp) = {(ep, tp)}")
            print(f"activacion = {activacion}")
            print(f"delta_h = {delta_h}")
            print(f"delta_p = {delta_p}")
            print("="*20)

        # Finalmente se entrega delta_h y delta_p. Se usarán en la siguiente iteración.
        return delta_h, delta_p

    # Override de método original
    def run_sim(self, ep_anterior=0, dh_anterior=0, dp_anterior=0, verbose=False, participacion=False, nombre="experimento.csv"):

        ep = self.P_inicial - self.P_obj
        tp = ep - ep_anterior

        # Variables del loop de simulación.
        delta_h = dh_anterior
        delta_p = dp_anterior

        if verbose is True:
            print(f"ep_anterior = {ep_anterior}")
            print(f"ep = self.P_inicial - self.P_obj = {ep}")
            print(f"tp = ep - ep_anterior = {tp}")
        
        # Primera iteración:
        delta_h, delta_p = self.step_sim(ep, tp, delta_h, delta_p, 0, 
                                         verbose=verbose, participacion=participacion, nombre=nombre)

        # Siguientes iteraciones
        for i in range(1, len(self.tiempo)):
            ep_anterior = ep
            ep = self.presion[i-1] - self.P_obj
            tp = ep - ep_anterior

            if verbose is True:
                print(f"ep_anterior = {ep_anterior}")
                print(f"ep = self.presion[i-1] - self.P_obj = {ep}")
                print(f"tp = ep - ep_anterior = {tp}")
                
            # Iteración i-ésima    
            delta_h, delta_p = self.step_sim(ep, tp, delta_h, delta_p, k=i, 
                                             verbose=verbose, participacion=participacion, nombre=nombre)
