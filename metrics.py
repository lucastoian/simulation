class Metrics:
    def __init__(self):
        # Conteos y tiempos
        self.n_done = 0
        self.throughput_sum = 0.0

        # Estado del quirófano para utilización y bloqueo
        self.theatre_state = 'idle'  # idle | busy | blocked
        self.last_state_change = 0.0
        self.theatre_busy_time = 0.0
        self.theatre_blocked_time = 0.0

        # Para cola de preparación (promedio por muestreo lo hará el monitor)
        self.prep_queue_samples_sum = 0.0
        self.prep_queue_samples_n = 0

        print("[Metrics] Inicializadas métricas.")

    def set_theatre_state(self, now: float, new_state: str):
        # acumular tiempo en el estado anterior
        dur = now - self.last_state_change
        if dur > 0:
            if self.theatre_state == 'busy':
                self.theatre_busy_time += dur
            elif self.theatre_state == 'blocked':
                self.theatre_blocked_time += dur
        print(f"[Metrics] t={now:.1f}: cambio estado quirófano {self.theatre_state} -> {new_state} (duración anterior={dur:.1f})")
        self.theatre_state = new_state
        self.last_state_change = now

    def record_patient_departure(self, t_exit: float, t_arrival: float):
        self.n_done += 1
        throughput = t_exit - t_arrival
        self.throughput_sum += throughput
        print(f"[Metrics] Paciente terminado en t={t_exit:.1f}, throughput={throughput:.1f}, total pacientes={self.n_done}")

    def record_prep_queue_sample(self, qlen: int):
        self.prep_queue_samples_sum += qlen
        self.prep_queue_samples_n += 1
        print(f"[Metrics] Muestra cola preparación: longitud={qlen}, muestras={self.prep_queue_samples_n}")

    def summarize(self, total_time: float):
        util = self.theatre_busy_time / total_time
        block = self.theatre_blocked_time / total_time
        avg_thr = (self.throughput_sum / self.n_done) if self.n_done else float('nan')
        avg_prep_queue = (self.prep_queue_samples_sum / self.prep_queue_samples_n) if self.prep_queue_samples_n else float('nan')
        print("[Metrics] Resumen final calculado.")
        return {
            'patients_done': self.n_done,
            'theatre_utilization': util,
            'theatre_block_rate': block,
            'avg_throughput_time': avg_thr,
            'avg_prep_queue_length': avg_prep_queue,
        }
