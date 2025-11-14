import simpy

class Monitor:
    """
    Proceso de monitorización por muestreo:
    - Toma snapshots de la longitud de la cola de preparación cada Δt.
    - También imprime el estado actual del quirófano para depuración.
    """
    def __init__(self, env: simpy.Environment, prep_res: simpy.Resource, metrics, dt: float):
        self.env = env
        self.prep_res = prep_res
        self.metrics = metrics
        self.dt = dt
        self.proc = env.process(self.run())
        print(f"[Monitor] Iniciado con intervalo de muestreo={dt}")

    def run(self):
        while True:
            # longitud de la cola de preparación
            qlen = len(self.prep_res.queue)
            self.metrics.record_prep_queue_sample(qlen)
            # Depuración: imprimir snapshot
            print(f"[Monitor] t={self.env.now:.1f}: cola preparación={qlen}, estado quirófano={self.metrics.theatre_state}")
            yield self.env.timeout(self.dt)
