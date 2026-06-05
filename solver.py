"""
solver.py
Motor MIP del modelo de Yu & Egbelu (2008) para Cross Docking.
Independiente de la interfaz — puede usarse desde Streamlit o desde terminal.
"""

from pulp import (
    LpProblem, LpMinimize, LpVariable, LpBinary, LpInteger,
    LpContinuous, lpSum, value, LpStatus, PULP_CBC_CMD
)


# ─────────────────────────────────────────────────────────────────────────────
# LECTURA DE ARCHIVOS TS5
# ─────────────────────────────────────────────────────────────────────────────

def leer_ts5_desde_texto(contenido: str) -> dict:
    """
    Parsea el contenido de un archivo TS5 (como string) y retorna
    los parámetros del modelo.

    Formato esperado:
        i  <num>           → número de camiones de entrada
        o  <num>           → número de camiones de salida
        n  <num>           → número de tipos de producto
        r  <i>  <n>  <qty> → unidades del producto n en camión entrada i
        s  <o>  <n>  <qty> → unidades del producto n requeridas por camión o
    """
    contenido = contenido.replace('\r', '\n').replace('\t', ' ')
    tokens = contenido.split()

    I = O = N = 0
    r, s = {}, {}

    idx = 0
    while idx < len(tokens):
        tipo = tokens[idx]
        if tipo == 'i':
            I = int(tokens[idx + 1]); idx += 2
        elif tipo == 'o':
            O = int(tokens[idx + 1]); idx += 2
        elif tipo == 'n':
            N = int(tokens[idx + 1]); idx += 2
        elif tipo == 'r':
            ti, tn, qty = int(tokens[idx+1]), int(tokens[idx+2]), int(tokens[idx+3])
            r[(ti, tn)] = qty; idx += 4
        elif tipo == 's':
            to_, tn, qty = int(tokens[idx+1]), int(tokens[idx+2]), int(tokens[idx+3])
            s[(to_, tn)] = qty; idx += 4
        else:
            idx += 1

    return {'I': I, 'O': O, 'N': N, 'r': r, 's': s}


def leer_ts5_desde_archivo(ruta: str) -> dict:
    contenido = open(ruta, encoding='utf-8').read()
    return leer_ts5_desde_texto(contenido)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDACIÓN DE BALANCE
# ─────────────────────────────────────────────────────────────────────────────

def validar_balance(datos: dict) -> tuple[bool, list]:
    """
    Verifica que Σ r(i,n) == Σ s(o,n) para cada producto n.
    Retorna (ok: bool, errores: list[str])
    """
    I, O, N = datos['I'], datos['O'], datos['N']
    r, s = datos['r'], datos['s']
    errores = []
    for n in range(1, N + 1):
        total_r = sum(r.get((i, n), 0) for i in range(1, I + 1))
        total_s = sum(s.get((o, n), 0) for o in range(1, O + 1))
        if total_r != total_s:
            errores.append(
                f"Producto n={n}: entrada={total_r} ≠ salida={total_s} "
                f"(diferencia={total_r - total_s})"
            )
    return len(errores) == 0, errores


# ─────────────────────────────────────────────────────────────────────────────
# FUSIÓN DE DATASETS (base TS5 + datos adicionales)
# ─────────────────────────────────────────────────────────────────────────────

def fusionar_datos(base: dict, adicional: dict) -> dict:
    """
    Combina dos conjuntos de datos.
    Los camiones adicionales se renumeran para evitar conflictos de índice.
    """
    I_base = base['I']
    O_base = base['O']
    N = max(base['N'], adicional['N'])

    r_nuevo = dict(base['r'])
    s_nuevo = dict(base['s'])

    # Renuinmerar camiones adicionales de entrada
    for (i, n), qty in adicional['r'].items():
        r_nuevo[(i + I_base, n)] = qty

    # Renumerar camiones adicionales de salida
    for (o, n), qty in adicional['s'].items():
        s_nuevo[(o + O_base, n)] = qty

    return {
        'I': I_base + adicional['I'],
        'O': O_base + adicional['O'],
        'N': N,
        'r': r_nuevo,
        's': s_nuevo
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODELO MIP — Yu & Egbelu (2008)
# ─────────────────────────────────────────────────────────────────────────────

def resolver_mip(datos: dict,
                 D: float = 10.0,
                 V: float = 5.0,
                 tau: float = 1.0,
                 tiempo_limite: int = 300,
                 gap: float = 0.01) -> dict:
    """
    Construye y resuelve el modelo MIP completo.

    Variables de decisión:
        T        : makespan (minimizar)
        c(i)     : tiempo entrada camión i a recepción
        F(i)     : tiempo salida  camión i de recepción
        d(o)     : tiempo entrada camión o a despacho
        L(o)     : tiempo salida  camión o de despacho
        x(i,o,n) : unidades producto n de camión i a camión o
        v(i,o)   : 1 si hay transferencia entre i y o
        p(i,i')  : 1 si i precede a i' en recepción
        q(o,o')  : 1 si o precede a o' en despacho

    Retorna dict con todos los resultados.
    """
    I  = datos['I']
    O  = datos['O']
    N  = datos['N']
    r  = datos['r']
    s  = datos['s']

    trucks_i = list(range(1, I + 1))
    trucks_o = list(range(1, O + 1))
    prods    = list(range(1, N + 1))

    total_unidades = sum(r.values()) + sum(s.values())
    M = total_unidades * tau + (I + O) * D + V + 9999

    prob = LpProblem("CrossDocking_MIP", LpMinimize)

    # ── Variables ─────────────────────────────────────────────────────────────
    T = LpVariable("T", lowBound=0)
    c = {i: LpVariable(f"c_{i}", lowBound=0) for i in trucks_i}
    F = {i: LpVariable(f"F_{i}", lowBound=0) for i in trucks_i}
    d = {o: LpVariable(f"d_{o}", lowBound=0) for o in trucks_o}
    L = {o: LpVariable(f"L_{o}", lowBound=0) for o in trucks_o}

    x = {(i, o, n): LpVariable(f"x_{i}_{o}_{n}", lowBound=0, cat=LpInteger)
         for i in trucks_i for o in trucks_o for n in prods}

    v = {(i, o): LpVariable(f"v_{i}_{o}", cat=LpBinary)
         for i in trucks_i for o in trucks_o}

    p = {(i, ip): LpVariable(f"p_{i}_{ip}", cat=LpBinary)
         for i in trucks_i for ip in trucks_i if i != ip}

    q = {(o, op): LpVariable(f"q_{o}_{op}", cat=LpBinary)
         for o in trucks_o for op in trucks_o if o != op}

    # ── Función objetivo ──────────────────────────────────────────────────────
    prob += T, "Minimizar_Makespan"

    # ── R1: T >= L(o) ─────────────────────────────────────────────────────────
    for o in trucks_o:
        prob += T >= L[o], f"R1_o{o}"

    # ── R2: balance salida camiones entrada ───────────────────────────────────
    for i in trucks_i:
        for n in prods:
            prob += (lpSum(x[(i, o, n)] for o in trucks_o)
                     == r.get((i, n), 0)), f"R2_i{i}_n{n}"

    # ── R3: balance llegada camiones salida ───────────────────────────────────
    for o in trucks_o:
        for n in prods:
            prob += (lpSum(x[(i, o, n)] for i in trucks_i)
                     == s.get((o, n), 0)), f"R3_o{o}_n{n}"

    # ── R4: vinculación x con v ───────────────────────────────────────────────
    for i in trucks_i:
        for o in trucks_o:
            for n in prods:
                prob += x[(i, o, n)] <= M * v[(i, o)], f"R4_i{i}_o{o}_n{n}"

    # ── R5: tiempo salida camión entrada ─────────────────────────────────────
    for i in trucks_i:
        total_r_i = sum(r.get((i, n), 0) for n in prods)
        prob += F[i] >= c[i] + tau * total_r_i, f"R5_i{i}"

    # ── R6-R7: secuencia muelle recepción ────────────────────────────────────
    for i in trucks_i:
        for ip in trucks_i:
            if i != ip:
                prob += c[i] >= F[ip] + D - M * (1 - p[(ip, i)]), f"R6_ip{ip}_i{i}"
                prob += c[ip] >= F[i] + D - M * p[(ip, i)],       f"R7_i{i}_ip{ip}"

    # ── R9: tiempo salida camión salida ───────────────────────────────────────
    for o in trucks_o:
        total_s_o = sum(s.get((o, n), 0) for n in prods)
        prob += L[o] >= d[o] + tau * total_s_o, f"R9_o{o}"

    # ── R10-R11: secuencia muelle despacho ───────────────────────────────────
    for o in trucks_o:
        for op in trucks_o:
            if o != op:
                prob += d[o] >= L[op] + D - M * (1 - q[(op, o)]), f"R10_op{op}_o{o}"
                prob += d[op] >= L[o] + D - M * q[(op, o)],       f"R11_o{o}_op{op}"

    # ── R13: conexión entre muelles ───────────────────────────────────────────
    for i in trucks_i:
        for o in trucks_o:
            transferencia_io = lpSum(x[(i, o, n)] for n in prods)
            prob += (L[o] >= c[i] + V + tau * transferencia_io
                     - M * (1 - v[(i, o)])), f"R13_i{i}_o{o}"

    # ── Resolver ──────────────────────────────────────────────────────────────
    solver = PULP_CBC_CMD(msg=0, timeLimit=tiempo_limite, gapRel=gap)
    estado = prob.solve(solver)
    estado_str = LpStatus[estado]

    if estado_str not in ['Optimal', 'Not Solved']:
        return {'estado': estado_str, 'makespan': None}

    # ── Extraer resultados ────────────────────────────────────────────────────
    makespan = round(value(T), 2)

    tiempos_entrada = {
        i: {'c': round(value(c[i]), 2), 'F': round(value(F[i]), 2)}
        for i in trucks_i
    }
    tiempos_salida = {
        o: {'d': round(value(d[o]), 2), 'L': round(value(L[o]), 2)}
        for o in trucks_o
    }

    secuencia_entrada = sorted(trucks_i, key=lambda i: tiempos_entrada[i]['c'])
    secuencia_salida  = sorted(trucks_o, key=lambda o: tiempos_salida[o]['d'])

    asignacion = {}
    for i in trucks_i:
        for o in trucks_o:
            for n in prods:
                val = value(x[(i, o, n)])
                if val and val > 0.5:
                    asignacion[(i, o, n)] = round(val)

    return {
        'estado'          : estado_str,
        'makespan'        : makespan,
        'D'               : D,
        'V'               : V,
        'tau'             : tau,
        'I'               : I,
        'O'               : O,
        'N'               : N,
        'tiempos_entrada' : tiempos_entrada,
        'tiempos_salida'  : tiempos_salida,
        'secuencia_entrada': secuencia_entrada,
        'secuencia_salida' : secuencia_salida,
        'asignacion'      : asignacion,
        'datos'           : datos,
    }
