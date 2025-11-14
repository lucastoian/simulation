import sys
from config import Config
from simulation import run_once

def main():
    # Redirigir toda la salida a salida.txt
    sys.stdout = open("salida.txt", "w", encoding="utf-8")
    # Configuración inicial
    cfg = Config(
        P=3,
        R=3,
        OP=1,
        sim_time=10_000,
        monitor_dt=1.0,
        seed=123  # fija la semilla para reproducibilidad
    )

    # Depuración inicial
    print("=== Inicio de simulación ===")
    print(f"Parámetros: P={cfg.P}, R={cfg.R}, OP={cfg.OP}, sim_time={cfg.sim_time}, monitor_dt={cfg.monitor_dt}, seed={cfg.seed}")
    print("Distribuciones: interarrival=25, prep=40, op=20, rec=40 (exponenciales)")
    print("------------------------------------------------------------")

    # Ejecutar simulación
    results = run_once(cfg)

    # Depuración final
    print("=== Fin de simulación ===")
    print("Resultados de la simulación (baseline exponencial):")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"- {k}: {v:.4f}")
        else:
            print(f"- {k}: {v}")

    # Ejemplos de cómo cambiar parámetros sin tocar lógica:
    # - distintas capacidades
    # cfg.P = 4; cfg.R = 2
    # - otras distribuciones: p.ej., determinista para operación
    # import random
    # cfg.time_fns['base']['op'] = lambda: 20.0
    # results2 = run_once(cfg)
    # print("\nResultados con op_time determinista=20:")
    # for k, v in results2.items():
    #     print(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")

if __name__ == '__main__':
    main()
