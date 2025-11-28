import math
import random
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import simpy

logger = logging.getLogger("hospital_sim")


@dataclass
class Config:
    P: int
    R: int
    OP: int = 1
    sim_time: float = 1000.0
    warmup: float = 200.0
    monitor_dt: float = 1.0
    seed: int = 123
    interarrival_mean: float = 25.0
    prep_mean: float = 40.0
    op_mean: float = 20.0
    rec_mean: float = 40.0
    scenario: str = "original"
    severe_prob: float = 0.3
    severe_op_mean: float = 35.0
    mild_op_mean: float = 15.0
    verbose: bool = False


class Metrics:
    def __init__(self, rec_capacity: int, verbose: bool = False):
        self.verbose = verbose
        self.observing = False
        self.obs_start_time = 0.0
        self.n_done = 0
        self.throughput_sum = 0.0
        self.theatre_state = "idle"
        self.last_state_change = 0.0
        self.theatre_busy_time = 0.0
        self.theatre_blocked_time = 0.0
        self.prep_queue_samples_sum = 0.0
        self.prep_queue_samples_n = 0
        self.prep_idle_samples_sum = 0.0
        self.prep_idle_samples_n = 0
        self.rec_capacity = rec_capacity
        self.rec_count = 0
        self.rec_last_change = 0.0
        self.rec_full_time = 0.0
        if self.verbose:
            logger.debug("Metrics initialized with recovery capacity %d", rec_capacity)

    def start_observation(self, now: float):
        if self.verbose:
            logger.debug("Starting observation window at time %.3f", now)
        self.observing = True
        self.obs_start_time = now
        self.n_done = 0
        self.throughput_sum = 0.0
        self.theatre_busy_time = 0.0
        self.theatre_blocked_time = 0.0
        self.prep_queue_samples_sum = 0.0
        self.prep_queue_samples_n = 0
        self.prep_idle_samples_sum = 0.0
        self.prep_idle_samples_n = 0
        self.rec_full_time = 0.0
        self.last_state_change = now
        self.rec_last_change = now

    def _flush_theatre(self, now: float):
        if not self.observing:
            self.last_state_change = now
            return
        dur = now - self.last_state_change
        if dur > 0:
            if self.theatre_state == "busy":
                self.theatre_busy_time += dur
            elif self.theatre_state == "blocked":
                self.theatre_blocked_time += dur
        self.last_state_change = now

    def _flush_rec(self, now: float):
        if not self.observing:
            self.rec_last_change = now
            return
        dur = now - self.rec_last_change
        if dur > 0 and self.rec_count == self.rec_capacity:
            self.rec_full_time += dur
        self.rec_last_change = now

    def set_theatre_state(self, now: float, new_state: str):
        if self.verbose:
            logger.debug("Time %.3f: theatre state %s -> %s", now, self.theatre_state, new_state)
        self._flush_theatre(now)
        self.theatre_state = new_state

    def rec_enter(self, now: float):
        self._flush_rec(now)
        self.rec_count += 1
        if self.verbose:
            logger.debug("Time %.3f: patient enters recovery, rec_count=%d", now, self.rec_count)

    def rec_leave(self, now: float):
        self._flush_rec(now)
        self.rec_count -= 1
        if self.verbose:
            logger.debug("Time %.3f: patient leaves recovery, rec_count=%d", now, self.rec_count)

    def record_prep_queue_sample(self, qlen: int):
        if not self.observing:
            return
        self.prep_queue_samples_sum += qlen
        self.prep_queue_samples_n += 1
        if self.verbose:
            logger.debug("Sampling prep queue length: %d", qlen)

    def record_prep_idle_sample(self, idle: int):
        if not self.observing:
            return
        self.prep_idle_samples_sum += idle
        self.prep_idle_samples_n += 1
        if self.verbose:
            logger.debug("Sampling prep idle capacity: %d", idle)

    def record_patient_departure(self, t_exit: float, t_arrival: float):
        if not self.observing:
            return
        self.n_done += 1
        self.throughput_sum += (t_exit - t_arrival)
        if self.verbose:
            logger.debug("Patient departed at time %.3f (arrival %.3f, sojourn %.3f)", t_exit, t_arrival, t_exit - t_arrival)

    def summarize(self, now: float) -> Dict[str, float]:
        self._flush_theatre(now)
        self._flush_rec(now)
        if not self.observing:
            total_time = 0.0
        else:
            total_time = now - self.obs_start_time
        if total_time <= 0:
            util = float("nan")
            block_rate = float("nan")
            prob_rec_full = float("nan")
        else:
            util = self.theatre_busy_time / total_time
            block_rate = self.theatre_blocked_time / total_time
            prob_rec_full = self.rec_full_time / total_time
        avg_thr = (self.throughput_sum / self.n_done) if self.n_done > 0 else float("nan")
        avg_qprep = (
            self.prep_queue_samples_sum / self.prep_queue_samples_n
            if self.prep_queue_samples_n > 0 else float("nan")
        )
        avg_prep_idle = (
            self.prep_idle_samples_sum / self.prep_idle_samples_n
            if self.prep_idle_samples_n > 0 else float("nan")
        )
        if self.verbose:
            logger.debug(
                "Summary: time=%.3f, util=%.4f, block_rate=%.4f, avg_qprep=%.4f, avg_prep_idle=%.4f, prob_rec_full=%.4f, patients_done=%d",
                total_time, util, block_rate, avg_qprep, avg_prep_idle, prob_rec_full, self.n_done
            )
        return {
            "obs_time": total_time,
            "patients_done": self.n_done,
            "theatre_utilization": util,
            "theatre_block_rate": block_rate,
            "avg_throughput_time": avg_thr,
            "avg_prep_queue_length": avg_qprep,
            "avg_prep_idle_capacity": avg_prep_idle,
            "prob_recovery_all_busy": prob_rec_full,
        }


class Monitor:
    def __init__(self, env: simpy.Environment,
                 prep_res: simpy.Resource,
                 metrics: Metrics,
                 dt: float,
                 verbose: bool = False):
        self.env = env
        self.prep_res = prep_res
        self.metrics = metrics
        self.dt = dt
        self.verbose = verbose
        self.proc = env.process(self.run())
        if self.verbose:
            logger.debug("Monitor created with sampling interval %.3f", dt)

    def run(self):
        while True:
            qlen = len(self.prep_res.queue)
            idle = self.prep_res.capacity - self.prep_res.count
            self.metrics.record_prep_queue_sample(qlen)
            self.metrics.record_prep_idle_sample(idle)
            if self.verbose:
                logger.debug(
                    "Time %.3f: monitoring snapshot, prep_queue=%d, prep_idle=%d",
                    self.env.now, qlen, idle
                )
            yield self.env.timeout(self.dt)


@dataclass
class Patient:
    pid: int
    ptype: str
    t_arrival: float
    prep_time: float
    op_time: float
    rec_time: float
    t_exit: float = None


def exp_sample(rng: random.Random, mean: float) -> float:
    return rng.expovariate(1.0 / mean)


def sample_op_time(cfg: Config, rng: random.Random) -> Tuple[float, str]:
    if cfg.scenario == "original":
        return exp_sample(rng, cfg.op_mean), "base"
    else:
        if rng.random() < cfg.severe_prob:
            return exp_sample(rng, cfg.severe_op_mean), "severe"
        else:
            return exp_sample(rng, cfg.mild_op_mean), "mild"


def patient_process(env: simpy.Environment,
                    patient: Patient,
                    cfg: Config,
                    prep_res: simpy.Resource,
                    theatre_res: simpy.Resource,
                    rec_res: simpy.Resource,
                    metrics: Metrics):
    if cfg.verbose:
        logger.debug(
            "Time %.3f: patient %d arrives (type=%s, prep=%.3f, op=%.3f, rec=%.3f)",
            patient.t_arrival, patient.pid, patient.ptype, patient.prep_time, patient.op_time, patient.rec_time
        )
    with prep_res.request() as req_prep:
        yield req_prep
        if cfg.verbose:
            logger.debug("Time %.3f: patient %d enters preparation", env.now, patient.pid)
        yield env.timeout(patient.prep_time)
        if cfg.verbose:
            logger.debug("Time %.3f: patient %d leaves preparation", env.now, patient.pid)
    with theatre_res.request() as req_theatre:
        yield req_theatre
        metrics.set_theatre_state(env.now, "busy")
        if cfg.verbose:
            logger.debug("Time %.3f: patient %d enters operating room", env.now, patient.pid)
        yield env.timeout(patient.op_time)
        if cfg.verbose:
            logger.debug("Time %.3f: patient %d completes surgery", env.now, patient.pid)
        if metrics.rec_count == metrics.rec_capacity:
            metrics.set_theatre_state(env.now, "blocked")
            if cfg.verbose:
                logger.debug("Time %.3f: patient %d waiting, all recovery units busy (OR blocked)", env.now, patient.pid)
        with rec_res.request() as req_rec:
            yield req_rec
            metrics.rec_enter(env.now)
            metrics.set_theatre_state(env.now, "idle")
            if cfg.verbose:
                logger.debug("Time %.3f: patient %d enters recovery", env.now, patient.pid)
            yield env.timeout(patient.rec_time)
            metrics.rec_leave(env.now)
            if cfg.verbose:
                logger.debug("Time %.3f: patient %d leaves recovery", env.now, patient.pid)
    patient.t_exit = env.now
    metrics.record_patient_departure(patient.t_exit, patient.t_arrival)
    if cfg.verbose:
        logger.debug("Time %.3f: patient %d exits system", patient.t_exit, patient.pid)


def source_process(env: simpy.Environment,
                   cfg: Config,
                   prep_res: simpy.Resource,
                   theatre_res: simpy.Resource,
                   rec_res: simpy.Resource,
                   metrics: Metrics,
                   rng: random.Random):
    pid = 0
    if cfg.verbose:
        logger.debug(
            "Starting source process with interarrival mean=%.3f, prep=%.3f, op=%.3f, rec=%.3f",
            cfg.interarrival_mean, cfg.prep_mean, cfg.op_mean, cfg.rec_mean
        )
    while True:
        ia = exp_sample(rng, cfg.interarrival_mean)
        yield env.timeout(ia)
        pid += 1
        op_time, ptype = sample_op_time(cfg, rng)
        prep_time = exp_sample(rng, cfg.prep_mean)
        rec_time = exp_sample(rng, cfg.rec_mean)
        p = Patient(
            pid=pid,
            ptype=ptype,
            t_arrival=env.now,
            prep_time=prep_time,
            op_time=op_time,
            rec_time=rec_time
        )
        env.process(patient_process(env, p, cfg, prep_res, theatre_res, rec_res, metrics))


def run_once(cfg: Config) -> Dict[str, float]:
    logger.debug(
        "Running one replication: P=%d, R=%d, OP=%d, sim_time=%.3f, warmup=%.3f, seed=%d, scenario=%s",
        cfg.P, cfg.R, cfg.OP, cfg.sim_time, cfg.warmup, cfg.seed, cfg.scenario
    )
    rng = random.Random(cfg.seed)
    env = simpy.Environment()
    prep_res = simpy.Resource(env, capacity=cfg.P)
    theatre_res = simpy.Resource(env, capacity=cfg.OP)
    rec_res = simpy.Resource(env, capacity=cfg.R)
    metrics = Metrics(rec_capacity=cfg.R, verbose=cfg.verbose)
    Monitor(env, prep_res, metrics, cfg.monitor_dt, verbose=cfg.verbose)
    env.process(source_process(env, cfg, prep_res, theatre_res, rec_res, metrics, rng))

    def do_warmup(env: simpy.Environment, metrics: Metrics, warmup: float):
        logger.debug("Starting warm-up period of length %.3f", warmup)
        yield env.timeout(warmup)
        logger.debug("Warm-up finished at time %.3f, starting observation", env.now)
        metrics.start_observation(env.now)

    env.process(do_warmup(env, metrics, cfg.warmup))
    env.run(until=cfg.warmup + cfg.sim_time)
    res = metrics.summarize(env.now)
    res.update({"P": cfg.P, "R": cfg.R, "scenario": cfg.scenario})
    logger.debug(
        "Replication finished: P=%d, R=%d, scenario=%s, block_rate=%.6f, avg_qprep=%.6f, avg_prep_idle=%.6f, prob_rec_full=%.6f",
        cfg.P, cfg.R, cfg.scenario,
        res["theatre_block_rate"], res["avg_prep_queue_length"],
        res["avg_prep_idle_capacity"], res["prob_recovery_all_busy"]
    )
    return res


def mean_ci_95(samples: List[float]):
    n = len(samples)
    if n == 0:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    mean = sum(samples) / n
    if n == 1:
        return mean, float("nan"), (mean, mean)
    var = sum((x - mean) ** 2 for x in samples) / (n - 1)
    s = math.sqrt(var)
    t_crit = 2.093
    half = t_crit * s / math.sqrt(n)
    return mean, half, (mean - half, mean + half)


def print_metric(label: str, samples: List[float]):
    m, h, (lo, hi) = mean_ci_95(samples)
    if math.isnan(m):
        print(f"{label}: mean=nan, 95%CI=(nan,nan)")
        logger.info("%s: mean=nan, CI undefined", label)
        return
    print(f"{label}: mean={m:.4f}, 95%CI=({lo:.4f},{hi:.4f})")
    if m > 0:
        rel = h / m
        print(f"    relative half-width = {rel:.2%}")
        logger.info(
            "%s: mean=%.6f, 95%%CI=(%.6f,%.6f), rel_half_width=%.4f",
            label, m, lo, hi, rel
        )
    else:
        print("    relative half-width = n/a (mean=0)")
        logger.info(
            "%s: mean=%.6f, 95%%CI=(%.6f,%.6f), mean is zero (relative width not defined)",
            label, m, lo, hi
        )


def run_independent_experiments():
    print("=== Independent experiments (different seeds) ===")
    logger.info("Starting independent experiments with different seeds")
    base_seed = 10_000
    n_rep = 20
    configs = {
        "3P4R": Config(P=3, R=4),
        "3P5R": Config(P=3, R=5),
        "4P5R": Config(P=4, R=5),
    }
    results = {
        name: {"block": [], "qprep": [], "recfull": [], "idle": []}
        for name in configs.keys()
    }
    for name, cfg in configs.items():
        print(f"\n--- Config {name} ---")
        logger.info("Running config %s with %d replications", name, n_rep)
        for r in range(n_rep):
            cfg.seed = base_seed + 1000 * list(configs.keys()).index(name) + r
            logger.debug("Replication %d for config %s with seed %d", r + 1, name, cfg.seed)
            res = run_once(cfg)
            results[name]["block"].append(res["theatre_block_rate"])
            results[name]["qprep"].append(res["avg_prep_queue_length"])
            results[name]["recfull"].append(res["prob_recovery_all_busy"])
            results[name]["idle"].append(res["avg_prep_idle_capacity"])
        print_metric("P(block OR)", results[name]["block"])
        print_metric("avg queue before prep", results[name]["qprep"])
        print_metric("avg idle capacity in prep", results[name]["idle"])
        print_metric("P(all recovery busy)", results[name]["recfull"])
    return results


def run_crn_experiments():
    print("\n\n=== Experiments with Common Random Numbers (CRN) ===")
    logger.info("Starting CRN experiments")
    base_seed = 20_000
    n_rep = 20
    cfg_3P4R = Config(P=3, R=4)
    cfg_3P5R = Config(P=3, R=5)
    cfg_4P5R = Config(P=4, R=5)
    diff_block_3P5R_4P5R = []
    diff_block_3P5R_3P4R = []
    diff_q_3P5R_4P5R = []
    diff_q_3P5R_3P4R = []
    for r in range(n_rep):
        seed = base_seed + r
        cfg_3P4R.seed = seed
        cfg_3P5R.seed = seed
        cfg_4P5R.seed = seed
        logger.debug("CRN replication %d with shared seed %d", r + 1, seed)
        res_3P4R = run_once(cfg_3P4R)
        res_3P5R = run_once(cfg_3P5R)
        res_4P5R = run_once(cfg_4P5R)
        diff_block_3P5R_4P5R.append(
            res_3P5R["theatre_block_rate"] - res_4P5R["theatre_block_rate"]
        )
        diff_block_3P5R_3P4R.append(
            res_3P5R["theatre_block_rate"] - res_3P4R["theatre_block_rate"]
        )
        diff_q_3P5R_4P5R.append(
            res_3P5R["avg_prep_queue_length"] - res_4P5R["avg_prep_queue_length"]
        )
        diff_q_3P5R_3P4R.append(
            res_3P5R["avg_prep_queue_length"] - res_3P4R["avg_prep_queue_length"]
        )

    def print_diff(label: str, diffs: List[float]):
        m, h, (lo, hi) = mean_ci_95(diffs)
        print(f"{label}: mean diff={m:.6f}, 95%CI=({lo:.6f},{hi:.6f})")
        logger.info("%s: mean_diff=%.6f, 95%%CI=(%.6f,%.6f)", label, m, lo, hi)

    print("\n--- Differences in P(block OR) ---")
    print_diff("3P5R - 4P5R (block)", diff_block_3P5R_4P5R)
    print_diff("3P5R - 3P4R (block)", diff_block_3P5R_3P4R)

    print("\n--- Differences in avg queue before prep ---")
    print_diff("3P5R - 4P5R (q_prep)", diff_q_3P5R_4P5R)
    print_diff("3P5R - 3P4R (q_prep)", diff_q_3P5R_3P4R)


def compare_block_vs_recfull_ci():
    print("\n\n=== Comparison of CI widths: blocking OR vs all recovery busy ===")
    logger.info("Comparing CI widths for blocking vs recovery-full probabilities")
    base_seed = 30_000
    n_rep = 20
    configs = {
        "3P4R": Config(P=3, R=4),
        "3P5R": Config(P=3, R=5),
        "4P5R": Config(P=4, R=5),
    }
    for name, cfg in configs.items():
        blocks = []
        recfulls = []
        logger.info("Config %s: computing CI for block and recovery-full", name)
        for r in range(n_rep):
            cfg.seed = base_seed + 1000 * list(configs.keys()).index(name) + r
            logger.debug("CI replication %d for config %s with seed %d", r + 1, name, cfg.seed)
            res = run_once(cfg)
            blocks.append(res["theatre_block_rate"])
            recfulls.append(res["prob_recovery_all_busy"])
        m_b, h_b, _ = mean_ci_95(blocks)
        m_r, h_r, _ = mean_ci_95(recfulls)
        rel_b = h_b / m_b if m_b > 0 else float("nan")
        rel_r = h_r / m_r if m_r > 0 else float("nan")
        print(f"\nConfig {name}:")
        if m_b > 0:
            print(f"  P(block OR): mean={m_b:.6f}, half={h_b:.6f}, relative half-width={rel_b:.2%}")
        else:
            print(f"  P(block OR): mean={m_b:.6f}, half={h_b:.6f}, relative half-width=n/a (mean=0)")
        if m_r > 0:
            print(f"  P(all recovery busy): mean={m_r:.6f}, half={h_r:.6f}, relative half-width={rel_r:.2%}")
        else:
            print(f"  P(all recovery busy): mean={m_r:.6f}, half={h_r:.6f}, relative half-width=n/a (mean=0)")
        logger.info(
            "Config %s: block mean=%.6f, half=%.6f, rel=%.4f; rec_full mean=%.6f, half=%.6f, rel=%.4f",
            name, m_b, h_b, rel_b if not math.isnan(rel_b) else -1.0,
            m_r, h_r, rel_r if not math.isnan(rel_r) else -1.0
        )


def twisted_scenario_experiment():
    print("\n\n=== Twisted scenario: same expected OR utilization (3P5R) ===")
    logger.info("Starting twisted scenario experiment for 3P5R")
    n_rep = 20
    base_seed = 40_000
    cfg_orig = Config(
        P=3, R=5,
        interarrival_mean=25.0,
        scenario="original"
    )
    cfg_twist = Config(
        P=3, R=5,
        interarrival_mean=26.25,
        scenario="twisted",
        severe_prob=0.3,
        severe_op_mean=35.0,
        mild_op_mean=15.0
    )
    diff_block = []
    diff_qprep = []
    diff_recfull = []
    for r in range(n_rep):
        seed = base_seed + r
        cfg_orig.seed = seed
        cfg_twist.seed = seed
        logger.debug("Twisted replication %d with shared seed %d", r + 1, seed)
        res_o = run_once(cfg_orig)
        res_t = run_once(cfg_twist)
        diff_block.append(res_t["theatre_block_rate"] - res_o["theatre_block_rate"])
        diff_qprep.append(res_t["avg_prep_queue_length"] - res_o["avg_prep_queue_length"])
        diff_recfull.append(res_t["prob_recovery_all_busy"] - res_o["prob_recovery_all_busy"])

    def print_diff(label: str, diffs: List[float]):
        m, h, (lo, hi) = mean_ci_95(diffs)
        print(f"{label}: mean diff={m:.6f}, 95%CI=({lo:.6f},{hi:.6f})")
        logger.info("%s: mean_diff=%.6f, 95%%CI=(%.6f,%.6f)", label, m, lo, hi)

    print("\nDifferences (twisted - original) for 3P5R:")
    print_diff("P(block OR)", diff_block)
    print_diff("avg queue before prep", diff_qprep)
    print_diff("P(all recovery busy)", diff_recfull)


def setup_logging():
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler("simulation.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


if __name__ == "__main__":
    setup_logging()
    logger.info("Starting hospital simulation experiments")
    run_independent_experiments()
    run_crn_experiments()
    compare_block_vs_recfull_ci()
    twisted_scenario_experiment()
    logger.info("All experiments completed")