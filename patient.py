from dataclasses import dataclass

@dataclass
class Patient:
    pid: int
    ptype: str
    t_arrival: float
    prep_time: float
    op_time: float
    rec_time: float
    t_exit: float = None

    def __post_init__(self):
        # Depuración: mostrar datos del paciente al crearse
        print(f"[Patient] Creado paciente {self.pid} (tipo={self.ptype}) "
              f"llegada={self.t_arrival:.1f}, prep={self.prep_time:.1f}, "
              f"op={self.op_time:.1f}, rec={self.rec_time:.1f}")

    def debug_exit(self):
        if self.t_exit is not None:
            print(f"[Patient] Paciente {self.pid} ha salido del sistema "
                  f"en t={self.t_exit:.1f}, throughput={self.t_exit - self.t_arrival:.1f}")
