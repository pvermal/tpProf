from matplotlib import pyplot as plt
import numpy as np
import numpy_financial as npf


def flujoFondos(p, q, cvu, cf, inv):
    """
    Calcula el flujo de fondos del proyecto. Util para calcular varios flujos de fondos
    y asi hacer un analisis de sensibilidad de diferentes variables.
    Recibe:
        * p:   (int)      - Precio por unidad.
        * q:   (np.array) - Cantidad de unidades vendidas por año. Poner 0 en el año 0.
        * cvu: (int)      - Costos variables por unidad.
        * cf:  (np.array) - Costos fijos por año. Poner 0 en el año 0.
        * inv: (int)      - Inversiones. Entero negativo.
    Devuelve:
        * ff:  (np.array) - Flujo de fondos. Incluye la inversion inicial en el año 0.
    """
    h = len(q)  # horizonte de proyecto
    i = p * q  # ingresos
    cv = cvu * q  # costos variables
    a = np.repeat(abs(inv) / h, h)  # amortizaciones
    uBruta = i - cv - cf - a  # utilidad bruta
    ig = uBruta * 0.35  # impuesto a las ganancias
    uNeta = uBruta - ig  # utilidad neta
    ff_ = uNeta + a  # flujo de fondos (sin inversion inicial)
    ff = np.insert(ff_, 0, inv)  # agrega las inversiones en el año 0

    return ff


## * Calculo de VAN y TIR para los valores calculados
p = 1000  # precio
q = np.array([26, 30, 30, 25, 24])  # cantidad
cvu = 258  # costos variables por unidad
sa = 0  # sueldo anual
cf = np.array([cvu, 0, 0, 0, 0]) + sa  # costos fijos
print(cf)
inv = -3000  # inversiones
td = 0.15  # tasa de descuento
ff = flujoFondos(p, q, cvu, cf, inv)
van = npf.npv(td, ff)
tir = npf.irr(ff)
plot = False

print("Flujo de fondos: ", ff)
print("VAN: ", van)
print("TIR: ", tir)

## * Analis de sensibilidad

# * Variacion de cantidad
# se repiten los calculos para valores que van entre el 70% y el 100% del q original
deltaQValues = np.arange(0.7, 1, 0.05)
tirDeltaQ = [
    npf.irr(flujoFondos(p, deltaQ * q, cvu, cf, inv)) for deltaQ in deltaQValues
]
vanDeltaQ = [
    npf.npv(td, flujoFondos(p, deltaQ * q, cvu, cf, inv)) for deltaQ in deltaQValues
]

if plot:
    plt.figure()
    plt.plot(deltaQValues * 100, np.array(tirDeltaQ) * 100)
    plt.title("Variacion de cantidad")
    plt.xlabel("Cantidad [% por periodo]")
    plt.ylabel("TIR [%]")
    plt.grid()

    plt.figure()
    plt.plot(deltaQValues * 100, vanDeltaQ)
    plt.title("Variacion de cantidad")
    plt.xlabel("Cantidad [% por periodo]")
    plt.ylabel("VAN [USD]")
    plt.grid()

# * Variacion del precio por unidad
# se repiten los calculos considerando un precio unitario desde USD 300 hasta USD 1000
deltaPValues = np.arange(300, 1050, 50)
tirDeltaP = [npf.irr(flujoFondos(deltaP, q, cvu, cf, inv)) for deltaP in deltaPValues]
vanDeltaP = [
    npf.npv(td, flujoFondos(deltaP, q, cvu, cf, inv)) for deltaP in deltaPValues
]

if plot:
    plt.figure()
    plt.plot(deltaPValues, np.array(tirDeltaP) * 100)
    plt.title("Variacion del precio por unidad")
    plt.xlabel("Precio [USD]")
    plt.ylabel("TIR [%]")
    plt.grid()

    plt.figure()
    plt.plot(deltaPValues, vanDeltaP)
    plt.title("Variacion del precio por unidad")
    plt.xlabel("Precio [USD]")
    plt.ylabel("VAN [USD]")
    plt.grid()

# * Variacion del precio por unidad (porecentual)
# se repiten los calculos considerando un precio unitario desde USD 300 hasta USD 1000
deltaPValuesPct = np.arange(0.7, 1, 0.05)
tirDeltaPPct = [
    npf.irr(flujoFondos(deltaP * p, q, cvu, cf, inv)) for deltaP in deltaPValuesPct
]
vanDeltaPPct = [
    npf.npv(td, flujoFondos(deltaP * p, q, cvu, cf, inv)) for deltaP in deltaPValuesPct
]

# * Variacion de los costos variables
# se repiten los calculos considerando costos variables por unidad hasta un 30% superiores
deltaCvValues = np.arange(1, 1.3, 0.05)
tirDeltaCv = [
    npf.irr(flujoFondos(p, q, deltaCv * cvu, cf, inv)) for deltaCv in deltaCvValues
]
vanDeltaCv = [
    npf.npv(td, flujoFondos(p, q, deltaCv * cvu, cf, inv)) for deltaCv in deltaCvValues
]

if plot:
    plt.figure()
    plt.plot(deltaCvValues * 100, np.array(tirDeltaCv) * 100)
    plt.title("Variacion de los costos variables")
    plt.xlabel("Costos variables [%]")
    plt.ylabel("TIR [%]")
    plt.grid()

    plt.figure()
    plt.plot(deltaCvValues * 100, vanDeltaCv)
    plt.title("Variacion de los costos variables")
    plt.xlabel("Costos variables [%]")
    plt.ylabel("VAN [USD]")
    plt.grid()

# * Variacion de la inversion inicial
# se repiten los calculos considerando una inversion inicial de hasta USD 6000
deltaInvValues = np.arange(3000, 6500, 500)
tirDeltaInv = [
    npf.irr(flujoFondos(p, q, cvu, cf, -deltaInv)) for deltaInv in deltaInvValues
]
vanDeltaInv = [
    npf.npv(td, flujoFondos(p, q, cvu, cf, -deltaInv)) for deltaInv in deltaInvValues
]
print("vanDeltaInv", vanDeltaInv)
print("tirDeltaInv", tirDeltaInv)

if plot:
    plt.figure()
    plt.plot(deltaInvValues, np.array(tirDeltaInv) * 100)
    plt.title("Variacion de la inversión inicial")
    plt.xlabel("Inversión inicial [USD]")
    plt.ylabel("TIR [%]")
    plt.grid()

    plt.figure()
    plt.plot(deltaInvValues, vanDeltaInv)
    plt.title("Variacion de la inversión inicial")
    plt.xlabel("Inversión inicial [USD]")
    plt.ylabel("VAN [USD]")
    plt.grid()

# * Variacion de la inversion inicial (porcentual)
# se repiten los calculos considerando una inversion inicial de hasta USD 6000
deltaInvValuesPct = np.arange(1, 1.3, 0.05)
tirDeltaInvPct = [
    npf.irr(flujoFondos(p, q, cvu, cf, deltaInv * inv))
    for deltaInv in deltaInvValuesPct
]
vanDeltaInvPct = [
    npf.npv(td, flujoFondos(p, q, cvu, cf, deltaInv * inv))
    for deltaInv in deltaInvValuesPct
]

# * Comparacion de variacion porcentual de VAN y TIR en funcion de
# * todas las variables sensibilizadas
pctAxis = deltaInvValuesPct = np.arange(0, 0.35, 0.05)

if plot:
    plt.figure()
    plt.plot(pctAxis * 100, np.flip(np.array(vanDeltaQ)), label="Q")
    plt.plot(pctAxis * 100, np.flip(np.array(vanDeltaPPct)), label="P")
    plt.plot(pctAxis * 100, np.array(vanDeltaCv), label="CV")
    plt.plot(pctAxis * 100, np.array(vanDeltaInvPct), label="$I_{0}$")
    plt.title("Comparación de sensibilidad")
    plt.xlabel("Variación [%]")
    plt.ylabel("VAN [USD]")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(pctAxis * 100, np.flip(np.array(tirDeltaQ)) * 100, label="Q")
    plt.plot(pctAxis * 100, np.flip(np.array(tirDeltaPPct)) * 100, label="P")
    plt.plot(pctAxis * 100, np.array(tirDeltaCv) * 100, label="CV")
    plt.plot(pctAxis * 100, np.array(tirDeltaInvPct) * 100, label="$I_{0}$")
    plt.title("Comparación de sensibilidad")
    plt.xlabel("Variación [%]")
    plt.ylabel("TIR [%]")
    plt.legend()
    plt.grid()

    plt.show()


"""
flujoFondosEstimado = [-3000, 12470.1, 14567, 14567, 12155.5, 11673.2]
tasaDescuento = 0.15

tir = npf.irr(flujoFondosEstimado)
van = npf.npv(tasaDescuento, flujoFondosEstimado)

print("Utilizando parametros estimados")
print("TIR:", tir)
print("VAN:", van)
print("--------")

tasasDescuento = np.arange(0.05, 0.5, 0.05)
sensibilidadVanVsTasaDescuento = [
    npf.npv(tasaDescuento, flujoFondosEstimado) for tasaDescuento in tasasDescuento
]

plt.figure()
plt.plot(tasasDescuento, sensibilidadVanVsTasaDescuento)
plt.xlabel("Tasa de descuento")
plt.ylabel("VAN")
plt.grid()
plt.show()

# * Analisis de sensibilidad -> RECALCULAR CUANDO TENGA EL FLUJO DE FONDOS APROBADO POR ENRIQUE

# * Variacion de cantidad - 20% menor a la estimada
flujoFondosVariacionCantidad = [-3000, 9498.6, 11113.2, 11113.2, 9184, 8701.7]
tirVariacionCantidad = npf.irr(flujoFondosVariacionCantidad)
vanVariacionCantidad = npf.npv(tasaDescuento, flujoFondosVariacionCantidad)

print("Variacion de cantidad")
print("TIR:", tirVariacionCantidad)
print("VAN:", vanVariacionCantidad)
print("--------")

# * Variacion del precio por unidad - USD 600 en vez de USD 1000
flujoFondosVariacionPrecio = [-3000, 5150.1, 6207, 6207, 5095.5, 4873.2]
tirVariacionPrecio = npf.irr(flujoFondosVariacionPrecio)
vanVariacionPrecio = npf.npv(tasaDescuento, flujoFondosVariacionPrecio)

print("Variacion de precio por unidad")
print("TIR:", tirVariacionPrecio)
print("VAN:", vanVariacionPrecio)
print("--------")

# * Variacion de los costos variables - 20% superior al estimado
flujoFondosVariacionCostosVariables = [
    -3000,
    11004.52,
    13000.8,
    13000.8,
    10757,
    10308.24,
]
tirVariacionCostosVariables = npf.irr(flujoFondosVariacionCostosVariables)
vanVariacionCostosVariables = npf.npv(
    tasaDescuento, flujoFondosVariacionCostosVariables
)

print("Variacion de costos variables")
print("TIR:", tirVariacionCostosVariables)
print("VAN:", vanVariacionCostosVariables)
print("--------")
"""
