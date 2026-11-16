"""Executors: exact Aer, Aer with readout noise, fake IBM backend, IBM hardware.

All return, for each pub, a 1-D int8 array of per-shot outcomes o in {+1, -1}
(o = +1 <-> bit '0').
"""
from __future__ import annotations

import numpy as np


def _bits_to_o(pub_result) -> np.ndarray:
    data = pub_result.data
    reg = getattr(data, "c", None)
    if reg is None:  # fall back to the first classical register
        reg = next(iter(data.values()))
    bits = (reg.array[:, 0] & 1).astype(np.int8)       # one clbit: bit 0 of byte 0
    return (1 - 2 * bits).astype(np.int8)


def run_independent_seeds(sampler, circuits, shots, base_seed: int, call_index: int):
    """Run each pub in its own call with an independent seed.

    qiskit-aer's SamplerV2 groups pubs by shot count and calls backend.run once per group,
    passing the SAME seed_simulator to every group. With multinomial (all-distinct) shot
    counts every pub then reuses an identical random stream, which correlates the sampling
    errors across pubs (observed: the standardised mean over 60 pubs had s.d. 6.4 instead of 1).
    We therefore derive one seed per pub from a SeedSequence and run pubs one at a time.
    """
    seeds = np.random.SeedSequence([int(base_seed), int(call_index)]).generate_state(len(circuits))
    out = []
    for qc, n, sd in zip(circuits, shots, seeds):
        sampler.options.run_options["seed_simulator"] = int(sd)
        out.append(_bits_to_o(sampler.run([(qc, None, int(n))]).result()[0]))
    return out


def readout_noise_model(p01: float, p10: float | None = None):
    """Readout error with P(read 1 | 0) = p01 and P(read 0 | 1) = p10 (symmetric by default)."""
    from qiskit_aer.noise import NoiseModel, ReadoutError
    p10 = p01 if p10 is None else p10
    nm = NoiseModel()
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]))
    return nm


class AerExecutor:
    """Aer SamplerV2, noiseless or with a noise model. No transpilation to a device."""

    def __init__(self, seed: int, noise_model=None):
        from qiskit_aer.primitives import SamplerV2
        opts = {"backend_options": {}}
        if noise_model is not None:
            opts["backend_options"]["noise_model"] = noise_model
        opts["run_options"] = {}
        self.sampler = SamplerV2(options=opts)          # seed set per pub, see run_independent_seeds
        self.seed = int(seed)
        self._calls = 0
        self.name = "aer_exact" if noise_model is None else "aer_readout_noise"
        self.meta = {"executor": self.name, "seeding": "independent SeedSequence seed per pub"}

    def prepare(self, circuits):
        return circuits

    def run(self, circuits, shots) -> list[np.ndarray]:
        self._calls += 1
        return run_independent_seeds(self.sampler, circuits, shots, self.seed, self._calls)


def best_readout_qubit(backend) -> int:
    """Physical qubit with the smallest reported readout error (ties: smallest sx error)."""
    props = backend.properties()
    best, key = 0, (np.inf, np.inf)
    for q in range(backend.num_qubits):
        try:
            ro = props.readout_error(q)
            sx = props.gate_error("sx", q)
        except Exception:
            continue
        if (ro, sx) < key:
            best, key = q, (ro, sx)
    return best


class BackendExecutor:
    """Fake IBM backend (local Aer with the device noise model) or real IBM hardware.

    Circuits are transpiled to the device with a fixed initial layout on one physical qubit.
    """

    def __init__(self, backend, qubit: int | None = None, seed: int = 0,
                 optimization_level: int = 1, is_hardware: bool = False, chunk: int = 500,
                 runtime_sampler: bool = False):
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        self.backend = backend
        self.qubit = best_readout_qubit(backend) if qubit is None else int(qubit)
        self.pm = generate_preset_pass_manager(
            optimization_level=optimization_level, backend=backend,
            initial_layout=[self.qubit], seed_transpiler=seed)
        self.seed = seed
        self.is_hardware = is_hardware
        self.chunk = int(chunk)
        # Fake backends: Aer SamplerV2 with the device noise model (fast). The runtime
        # SamplerV2 in local mode is the hardware code path but costs ~1 s per pub.
        self.runtime_sampler = bool(runtime_sampler or is_hardware)
        self._sampler = None
        self.name = backend.name
        props = backend.properties() if hasattr(backend, "properties") else None
        meta = {"executor": "ibm_hardware" if is_hardware else "fake_backend",
                "backend": backend.name, "qubit": self.qubit,
                "optimization_level": optimization_level, "shot_chunk": self.chunk,
                "sampler": "qiskit_ibm_runtime.SamplerV2" if self.runtime_sampler
                else "qiskit_aer.primitives.SamplerV2.from_backend"}
        try:
            meta["reported_readout_error"] = props.readout_error(self.qubit)
            meta["reported_prob_meas1_prep0"] = props.qubit_property(self.qubit, "prob_meas1_prep0")[0]
            meta["reported_prob_meas0_prep1"] = props.qubit_property(self.qubit, "prob_meas0_prep1")[0]
            meta["reported_sx_error"] = props.gate_error("sx", self.qubit)
            meta["properties_last_update"] = str(props.last_update_date)
        except Exception:
            pass
        self.meta = meta

    def prepare(self, circuits):
        return self.pm.run(list(circuits))

    def sampler(self):
        if self._sampler is None:
            if self.is_hardware:
                from qiskit_ibm_runtime import SamplerV2
                self._sampler = SamplerV2(mode=self.backend)
            elif self.runtime_sampler:
                from qiskit_ibm_runtime import SamplerV2
                self._sampler = SamplerV2(mode=self.backend,
                                          options={"simulator": {"seed_simulator": self.seed}})
            else:
                from qiskit_aer.primitives import SamplerV2
                self._sampler = SamplerV2.from_backend(self.backend)   # seeds set per pub
        return self._sampler

    def chunk_plan(self, shots) -> list[tuple[int, int]]:
        """IBM Runtime SamplerV2 requires the same shot count for every pub in a job. Each
        circuit needing n shots is therefore run as ceil(n / chunk) pubs of `chunk` shots, and
        only the first n outcomes are kept. Shots of one circuit are i.i.d., so truncation keeps
        the copy-level multinomial design exact; the discarded shots are reported."""
        return [(i, self.chunk) for i, n in enumerate(shots) for _ in range(-(-int(n) // self.chunk))]

    def executed_shots(self, shots) -> int:
        return sum(c for _, c in self.chunk_plan(shots))

    def run(self, transpiled, shots) -> list[np.ndarray]:
        plan = self.chunk_plan(shots)
        if self.runtime_sampler:
            job = self.sampler().run([(transpiled[i], None, c) for i, c in plan])
            self.last_job_id = job.job_id() if callable(getattr(job, "job_id", None)) else None
            outs = [_bits_to_o(r) for r in job.result()]
        else:
            # Fake backend: all chunk pubs have the same shot count, so Aer runs them in ONE
            # backend.run call and gives experiment i the seed (base + i): distinct streams.
            # (Per-pub calls would cost ~1 s each for the 156-qubit noise model.) The base seed
            # changes per call; independence is checked in tests/test_pipeline.py.
            self._calls = getattr(self, "_calls", 0) + 1
            base = int(np.random.SeedSequence([self.seed, self._calls]).generate_state(1)[0] % 2**31)
            smp = self.sampler()
            smp.options.run_options["seed_simulator"] = base
            outs = [_bits_to_o(r) for r in smp.run([(transpiled[i], None, c) for i, c in plan]).result()]
        parts: dict = {}
        for (i, _), o in zip(plan, outs):
            parts.setdefault(i, []).append(o)
        out = [np.concatenate(parts[i])[: int(n)] for i, n in enumerate(shots)]
        self.meta["shots_executed"] = int(sum(c for _, c in plan))
        self.meta["shots_discarded"] = int(sum(c for _, c in plan) - sum(int(n) for n in shots))
        self.meta["n_pubs_submitted"] = len(plan)
        return out
