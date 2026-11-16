"""Run search circuits on simulators and hardware through one interface.

A Backend transpiles circuits, runs a batch of them as one job, and turns the
results into 8 counts per circuit (one per line index). Calibration data
(gate errors, readout errors) is optional: a provider that publishes none can
still be used through calibration runs (see qttt.noise.calibrate).

Only IBM is implemented so far. Real IBM devices need an account, read in this
order: the QISKIT_IBM_TOKEN environment variable (and optionally
QISKIT_IBM_INSTANCE), the same names in a .env file in the current directory or
a parent (see .env.example), or an account saved once with
QiskitRuntimeService.save_account(). .env is git-ignored; never put a token in
code or in a committed file.
"""

import copy
import dataclasses
import hashlib
import math
import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.transpiler import PassManager, generate_preset_pass_manager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

from qttt.board import EMPTY, X
from qttt.encoding import bitstring_to_line_index, line_index_to_bitstring
from qttt.grover import board_flips, build_search_circuit
from qttt.stats import NUM_LINES

LINE_REGISTER = "line_cl"
ANCILLA_SETTINGS = ("auto", True, False)
ACCOUNT_PREFIX = "QISKIT_IBM_"  # the only .env keys read
ANCILLA_FLAGS = {"auto": "auto", "on": True, "off": False}  # command-line spelling of each setting
# Error suppression for real devices, off unless asked for: dynamical decoupling fills idle gaps (the circuit is
# ~1000 layers deep), gate twirling turns coherent two-qubit errors into random ones. On ibm_marrakesh and
# ibm_kingston it left less signal on the winning line than running without it (scripts/hardware_check.py).
# Measurement twirling stays off: the in-job readout matrix measures the readout bias, which twirling would
# change between calibration and boards.
ERROR_SUPPRESSION_OPTIONS = {"dynamical_decoupling": {"enable": True, "sequence_type": "XY4"},
                           "twirling": {"enable_gates": True}}
DEFAULT_TRANSPILE_SEEDS = 5  # for the command-line tools; enough to find a good layout on IBM devices


@dataclass(frozen=True)
class CircuitMetrics:
    """Cost of a transpiled circuit on one backend."""
    two_qubit_gates: int
    depth: int
    log10_esp: float | None  # None when the backend publishes no gate errors
    line_qubits: tuple[int, ...]  # physical qubit measured into line_cl[0], [1], [2]
    transpile_seed: int | None
    ancilla: bool = False  # the player search circuit's extra qubit (see Backend)
    log10_idle: float | None = None  # None without published T1/T2 (see Backend.idle_log10_survival)
    log10_readout: float | None = None  # None without published readout errors


@dataclass(frozen=True)
class JobOutput:
    """Counts per line index for each circuit in a job, plus the provider's job id."""
    counts: tuple[tuple[int, ...], ...]
    job_id: str | None


class Backend(ABC):
    """Common interface for every place a circuit can run.

    Search circuits are transpiled once per variant from an empty board (the
    template); each board then only adds X gates on the physical qubits where
    its board qubits start. Every board gets the same routed circuit, so gate
    counts, ESP and the measured line qubits are the same for all boards, and
    the compiler never sees the board.

    ancilla sets whether the player variant's search circuit gets one extra
    qubit that makes each oracle gate cheaper (see qttt.oracles.mcz). Whether
    that pays off depends on the device, so "auto" (the default) compiles both
    and keeps the better score (see score()); True or False forces one. The
    legacy full variant never uses it.
    """

    name: str
    provider: str
    simulated: bool

    def __init__(self, ancilla: bool | str = "auto"):
        if ancilla not in ANCILLA_SETTINGS:
            raise ValueError(f"ancilla must be one of {ANCILLA_SETTINGS}, got {ancilla!r}")
        self.ancilla = ancilla
        self._templates: dict[str, QuantumCircuit] = {}

    def uses_ancilla(self, variant: str) -> bool:
        """Whether this variant's search circuit has the ancilla qubit (with "auto", compiles both to decide)."""
        if variant != "player":
            return False
        if self.ancilla == "auto":
            return self.search_template(variant).metadata["ancilla"]
        return self.ancilla

    def logical_search_circuit(self, variant: str, board: Sequence[str], player: str | None = None) -> QuantumCircuit:
        """The search circuit for a board before transpiling, with or without the ancilla as this backend uses it."""
        return build_search_circuit(variant, board, player, ancilla=self.uses_ancilla(variant))

    @abstractmethod
    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Rewrite any circuit for this backend."""

    def search_template(self, variant: str) -> QuantumCircuit:
        """The transpiled search circuit for an empty board."""
        if variant not in self._templates:
            player = X if variant == "player" else None
            if variant != "player":
                choices = (False,)
            else:
                choices = (False, True) if self.ancilla == "auto" else (self.ancilla,)
            # Ties keep the first choice, the circuit without the ancilla.
            self._templates[variant] = max(
                (self.transpile(build_search_circuit(variant, [EMPTY] * 9, player, ancilla=a)) for a in choices),
                key=self.score)
        return self._templates[variant]

    def score(self, tqc: QuantumCircuit) -> tuple[float, int]:
        """How good a transpiled circuit is, higher is better, from published data only.

        First the log10 chance that nothing goes wrong: no gate fails (ESP), no
        qubit loses coherence while it waits, and the 3 line qubits read out
        correctly, counting each part the backend publishes; then fewer
        two-qubit gates.
        """
        parts = (self.gate_log10_esp(tqc), self.idle_log10_survival(tqc), self.readout_log10_success(tqc))
        return (sum(p for p in parts if p is not None), -two_qubit_count(tqc))

    def set_search_template(self, variant: str, tqc: QuantumCircuit) -> None:
        """Use a template transpiled earlier (for example one saved in a RunStore) instead of transpiling."""
        self._templates[variant] = tqc

    def search_circuit(self, variant: str, board: Sequence[str], player: str | None = None) -> QuantumCircuit:
        """The transpiled search circuit for a board: the template plus the board's X gates."""
        flips = board_flips(variant, board, player)
        template = self.search_template(variant)
        layout = (template.layout.initial_index_layout(filter_ancillas=True)
                  if template.layout is not None else list(range(template.num_qubits)))
        prep = QuantumCircuit(template.num_qubits)
        for logical in flips:
            prep.x(layout[logical])
        prep = self._native(prep)
        gates = [[inst.operation.name, [float(p) for p in inst.operation.params],
                  [prep.find_bit(q).index for q in inst.qubits]] for inst in prep.data]
        return with_board_prep(template, gates)

    def _native(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Single-qubit gates rewritten into the backend's own gate set, on the same qubits."""
        return circuit

    @abstractmethod
    def _sample(self, circuits: Sequence[QuantumCircuit], shots: int,
                on_submit: Callable[[str | None], None]) -> tuple[list[dict[str, int]], str | None]:
        """Run transpiled circuits as one job; return line_cl counts per circuit and the job id.

        on_submit(job_id) is called once the job is submitted, before waiting for its results.
        """

    def readout_circuits(self, line_qubits: Sequence[int]) -> list[QuantumCircuit]:
        """8 transpiled circuits that prepare line index 0-7 directly on the given physical qubits."""
        return [_prepare_line_index(i) for i in range(NUM_LINES)]

    def gate_log10_esp(self, tqc: QuantumCircuit) -> float | None:
        """log10 of the product of (1 - error) over every gate, or None without gate errors."""
        return None

    def idle_log10_survival(self, tqc: QuantumCircuit) -> float | None:
        """log10 of the chance that no qubit loses coherence while it waits, or None without timing data."""
        return None

    def readout_errors(self, qubits: Sequence[int]) -> list[tuple[float, float]] | None:
        """Per qubit (P(read 1 | prepared 0), P(read 0 | prepared 1)), or None if unpublished."""
        return None

    def readout_log10_success(self, tqc: QuantumCircuit) -> float | None:
        """log10 of the chance that the 3 line qubits all read out correctly (average of both errors)."""
        errors = self.readout_errors(line_qubits(tqc))
        if errors is None:
            return None
        return sum(math.log10(1 - (e01 + e10) / 2) for e01, e10 in errors)

    def calibration_date(self) -> str | None:
        """When the published calibration data was taken, if there is any."""
        return None

    def options(self) -> dict:
        """Run options to record with each result."""
        return {}

    def run(self, circuits: Sequence[QuantumCircuit], shots: int,
            on_submit: Callable[[str | None], None] | None = None) -> JobOutput:
        """Run transpiled circuits as one job and count each line index.

        on_submit(job_id) is called as soon as the job is submitted, so the id
        can be recorded before a possibly long wait in the queue.
        """
        raw, job_id = self._sample(circuits, shots, on_submit or (lambda job_id: None))
        return _job_output(raw, job_id)

    def fetch(self, job_id: str) -> JobOutput:
        """The counts of a job submitted earlier, by its id (waits if it is still running)."""
        raise NotImplementedError(f"{self.name} keeps no jobs to fetch")

    def board_metrics(self, tqc: QuantumCircuit, template: CircuitMetrics) -> CircuitMetrics:
        """metrics() of a board's circuit (search_circuit), given the metrics of the template it was built from.

        A board only adds single-qubit gates on board qubits in front of the
        template, so the two-qubit count, line qubits and idle survival (which
        counts only the exposed qubits after their first gate) are the
        template's; the gate ESP and depth are worked out again, which is cheap.
        This skips a scheduling pass of the whole circuit per board. A circuit
        without exposed-qubit metadata gets the full metrics().
        """
        if "exposed_qubits" not in (tqc.metadata or {}) or "board_prep" not in (tqc.metadata or {}):
            return self.metrics(tqc)
        return dataclasses.replace(template, depth=tqc.depth(), log10_esp=self.gate_log10_esp(tqc))

    def metrics(self, tqc: QuantumCircuit) -> CircuitMetrics:
        return CircuitMetrics(
            two_qubit_gates=two_qubit_count(tqc),
            depth=tqc.depth(),
            log10_esp=self.gate_log10_esp(tqc),
            line_qubits=line_qubits(tqc),
            transpile_seed=tqc.metadata.get("transpile_seed"),
            ancilla=tqc.metadata.get("ancilla", False),
            log10_idle=self.idle_log10_survival(tqc),
            log10_readout=self.readout_log10_success(tqc),
        )


class IdealBackend(Backend):
    """The perfect Aer simulator. A fixed seed makes the counts repeatable."""

    name = "ideal"
    provider = "aer"
    simulated = True

    def __init__(self, seed: int | None = None, ancilla: bool | str = "auto"):
        super().__init__(ancilla)
        self.seed = seed
        self._simulator = AerSimulator()

    def transpile(self, circuit):
        # As on IBM devices: the template must stay valid for any board, so the compiler may not assume |0>.
        tqc = transpile(circuit, self._simulator, seed_transpiler=self.seed, qubits_initially_zero=False)
        tqc.metadata = {**(tqc.metadata or {}), "transpile_seed": self.seed}
        return tqc

    def _sample(self, circuits, shots, on_submit):
        on_submit(None)
        result = AerSampler(seed=self.seed).run(list(circuits), shots=shots).result()
        return [_line_counts(r) for r in result], None

    def options(self):
        return {"seed": self.seed, "ancilla": self.ancilla}


class IBMBackend(Backend):
    """An IBM device, real or fake.

    Fake backends (snapshots of real devices) run on Aer with the device's
    noise model; real devices run through IBM's runtime Sampler. The
    transpiler is tried with `transpile_seeds` seeds and the circuit with the
    best score is kept.

    sampler_options go to IBM's runtime Sampler on real devices; None (default)
    means none, ERROR_SUPPRESSION_OPTIONS turns on error suppression. Simulated
    runs use no Sampler options, so they record {}.

    With schedule (default), simulated runs first pad every qubit's idle time
    with delays (ALAP scheduling), so the noise model's T1/T2 decay also acts
    while qubits wait, as it does on the device. Real devices time circuits
    themselves, so it has no effect there.
    """

    provider = "ibm"

    def __init__(self, backend, *, seed: int | None = None, transpile_seeds: int = 1,
                 sampler_options: dict | None = None, simulate: bool | None = None, ancilla: bool | str = "auto",
                 schedule: bool = True):
        from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

        if transpile_seeds < 1:
            raise ValueError(f"transpile_seeds must be at least 1, got {transpile_seeds}")
        super().__init__(ancilla)
        self.backend = backend
        self.name = backend.name
        self.simulated = isinstance(backend, FakeBackendV2) if simulate is None else simulate
        self.seed = seed
        self.transpile_seeds = transpile_seeds
        if self.simulated:
            self.sampler_options = {}
        else:
            self.sampler_options = copy.deepcopy(sampler_options or {})
        self.schedule = schedule

    def transpile(self, circuit):
        base = self.seed or 0
        best, best_score = None, None
        for s in range(base, base + self.transpile_seeds):
            pm = generate_preset_pass_manager(optimization_level=3, backend=self.backend,
                                              seed_transpiler=s, qubits_initially_zero=False)
            tqc = pm.run(circuit)
            score = self.score(tqc)
            if best_score is None or score > best_score:
                best, best_score = tqc, score
                best.metadata = {**(circuit.metadata or {}), "transpile_seed": s}
        return best

    def _native(self, circuit):
        basis = list(self.backend.target.operation_names)
        if all(inst.operation.name in basis for inst in circuit.data):
            return circuit
        return transpile(circuit, basis_gates=basis, optimization_level=0)

    def readout_circuits(self, line_qubits):
        return [transpile(_prepare_line_index(i), self.backend, initial_layout=list(line_qubits),
                          optimization_level=1, seed_transpiler=self.seed)
                for i in range(NUM_LINES)]

    def _sample(self, circuits, shots, on_submit):
        if self.simulated:
            on_submit(None)
            if self.schedule:
                target = self.backend.target
                scheduler = PassManager([ALAPScheduleAnalysis(target=target), PadDelay(target=target)])
                circuits = scheduler.run(list(circuits))
            sampler = AerSampler.from_backend(self.backend, seed=self.seed)
            result = sampler.run(list(circuits), shots=shots).result()
            return [_line_counts(r) for r in result], None
        # The client-side Sampler replaces the runtime SamplerV2 (deprecated in qiskit-ibm-runtime 0.50).
        from qiskit_ibm_runtime.executor_sampler import Sampler as RuntimeSampler

        job = RuntimeSampler(mode=self.backend, options=self.sampler_options or None).run(list(circuits), shots=shots)
        on_submit(job.job_id())
        return [_line_counts(r) for r in job.result()], job.job_id()

    def dry_run(self, circuits: Sequence[QuantumCircuit], shots: int):
        """Have IBM validate a job and estimate its usage without running it on the device; returns the job.

        Its results are random mock data, so they are never counted or saved.
        Real devices only.
        """
        if self.simulated:
            raise ValueError("a dry run is for real devices; a simulated backend runs nothing remotely")
        from qiskit_ibm_runtime.executor_sampler import Sampler as RuntimeSampler

        sampler = RuntimeSampler(mode=self.backend, options=self.sampler_options or None)
        return sampler.run(list(circuits), shots=shots, dry_run=True)

    def fetch(self, job_id):
        if self.simulated:
            return super().fetch(job_id)
        job = self.backend.service.job(job_id)
        return _job_output([_line_counts(r) for r in job.result()], job_id)

    def gate_log10_esp(self, tqc):
        target = self.backend.target
        total = 0.0
        found = False
        for inst in tqc.data:
            name = inst.operation.name
            if name in ("barrier", "measure") or name not in target:
                continue
            props = target[name].get(tuple(tqc.find_bit(q).index for q in inst.qubits))
            if props is not None and props.error is not None:
                found = True
                # Summed as logs: ESP itself can be smaller than a float can hold.
                total += math.log10(1 - props.error) if props.error < 1 else -math.inf
        return total if found else None

    def idle_log10_survival(self, tqc):
        """Each idle stretch on an exposed qubit, after its first gate, survives with exp(-t / T2).

        Exposed qubits are the ones the search circuit holds in superposition
        the whole time (the line register and the ancilla or phase qubit, from
        the circuit's "exposed_qubits" metadata), at the physical qubits where
        they start and where the line is measured; board qubits only hold 0 or
        1 as controls, so losing phase does not hurt them. A circuit without
        that metadata counts every qubit. The circuit is scheduled as late as
        possible (as the device and the simulator run it), and T2 is capped at
        2 T1. scripts/score_check.py checks this ranking against measured signal.
        """
        target = self.backend.target
        props = target.qubit_properties
        if not props or target.dt is None:
            return None
        exposed = None
        if "exposed_qubits" in (tqc.metadata or {}) and tqc.layout is not None:
            layout = tqc.layout.initial_index_layout(filter_ancillas=True)
            exposed = {layout[i] for i in tqc.metadata["exposed_qubits"]} | set(line_qubits(tqc))
        scheduled = PassManager([ALAPScheduleAnalysis(target=target), PadDelay(target=target)]).run(tqc)
        started = set()
        total = 0.0
        found = False
        for inst in scheduled.data:
            q = scheduled.find_bit(inst.qubits[0]).index if inst.qubits else None
            if inst.operation.name != "delay":
                started.update(scheduled.find_bit(b).index for b in inst.qubits)
                continue
            p = props[q] if q is not None and q < len(props) else None
            if q not in started or (exposed is not None and q not in exposed) or p is None or p.t1 is None \
                    or p.t2 is None:
                continue
            seconds = inst.operation.duration * (target.dt if inst.operation.unit == "dt" else 1)
            total -= seconds / min(p.t2, 2 * p.t1) / math.log(10)
            found = True
        return total if found else None

    def readout_errors(self, qubits):
        props = self.backend.properties() if hasattr(self.backend, "properties") else None
        errors = []
        for q in qubits:
            info = props.qubit_property(q) if props is not None else {}
            if "prob_meas1_prep0" in info and "prob_meas0_prep1" in info:
                errors.append((info["prob_meas1_prep0"][0], info["prob_meas0_prep1"][0]))
                continue
            measure = self.backend.target["measure"].get((q,)) if "measure" in self.backend.target else None
            if measure is None or measure.error is None:
                return None
            errors.append((measure.error, measure.error))  # only the average is published
        return errors

    def calibration_date(self):
        props = self.backend.properties() if hasattr(self.backend, "properties") else None
        return props.last_update_date.isoformat() if props is not None else None

    def options(self):
        return {"seed": self.seed, "transpile_seeds": self.transpile_seeds,
                "sampler_options": self.sampler_options, "ancilla": self.ancilla, "schedule": self.schedule}


def get_backend(name: str, *, seed: int | None = None, transpile_seeds: int = 1, sampler_options: dict | None = None,
                instance: str | None = None, ancilla: bool | str = "auto") -> Backend:
    """Backend by name: "ideal", a fake IBM backend such as "fake_berlin", or a real IBM device."""
    if name == "ideal":
        return IdealBackend(seed, ancilla)
    if name.startswith("fake_"):
        from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

        with warnings.catch_warnings():
            # Loading the catalog builds every fake backend; fake_nighthawk warns that its errors are placeholders.
            warnings.filterwarnings("ignore", message="Properties of fake_nighthawk")
            device = FakeProviderForBackendV2().backend(name)
    else:
        device = _runtime_service(instance).backend(name)
    return IBMBackend(device, seed=seed, transpile_seeds=transpile_seeds, sampler_options=sampler_options,
                      ancilla=ancilla)


def _runtime_service(instance=None):
    """IBM account from QISKIT_IBM_TOKEN (environment or .env) if set, otherwise the saved account."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    # The real environment wins over .env.
    settings = {**read_dotenv(), **{k: v for k, v in os.environ.items() if k.startswith(ACCOUNT_PREFIX)}}
    instance = instance or settings.get("QISKIT_IBM_INSTANCE") or None
    token = settings.get("QISKIT_IBM_TOKEN")
    if token:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
    return QiskitRuntimeService(instance=instance)


def read_dotenv(start: str | os.PathLike | None = None) -> dict[str, str]:
    """The QISKIT_IBM_* settings in the nearest .env (start directory, then its parents).

    Only keys starting with ACCOUNT_PREFIX are read, and nothing is written to
    os.environ, so other secrets in the file never reach this process's
    environment. Blank lines and # comments are skipped; values may be quoted.
    Nothing read here is printed or logged.
    """
    directory = Path(start or Path.cwd()).resolve()
    for folder in (directory, *directory.parents):
        path = folder / ".env"
        if path.is_file():
            break
    else:
        return {}
    settings = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        if not key.startswith(ACCOUNT_PREFIX):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        settings[key] = value
    return settings


def with_board_prep(template: QuantumCircuit, gates: Sequence) -> QuantumCircuit:
    """The template with a board's gates in front, each [name, params, physical qubits].

    The gates are kept in the circuit's metadata ("board_prep"), so a store can
    save them instead of the whole circuit and rebuild it exactly from the
    template (see RunStore.load_circuit).
    """
    standard = get_standard_gate_name_mapping()
    tqc = template.copy()
    for name, params, qubits in reversed(gates):
        tqc.data.insert(0, CircuitInstruction(type(standard[name])(*params), tuple(tqc.qubits[q] for q in qubits)))
    tqc.metadata = {**(template.metadata or {}), "board_prep": [list(g) for g in gates]}
    return tqc


def template_fingerprint(tqc: QuantumCircuit) -> dict:
    """What a calibration's noise depends on beyond the line qubits: the compiled gates and where they run.

    ancilla, the two-qubit gate count, the initial layout and a hash of every
    gate with its physical qubits (and parameters, rounded) say which circuit it
    is; two templates that agree on them run the same gates on the same
    physical qubits, so a profile measured on one holds for the other. The
    transpile seed is kept to say how it was found, but not compared: two
    seeds can find the same circuit.
    """
    layout = tqc.layout.initial_index_layout(filter_ancillas=True) if tqc.layout is not None else None
    gates = hashlib.sha256()
    for inst in tqc.data:
        qubits = ",".join(str(tqc.find_bit(q).index) for q in inst.qubits)
        params = ",".join(f"{float(p):.6g}" if isinstance(p, (int, float)) else str(p) for p in inst.operation.params)
        gates.update(f"{inst.operation.name}({params})[{qubits}];".encode())
    return {
        "ancilla": bool((tqc.metadata or {}).get("ancilla", False)),
        "two_qubit_gates": two_qubit_count(tqc),
        "layout": None if layout is None else [int(q) for q in layout],
        "gates_sha256": gates.hexdigest(),
        "transpile_seed": (tqc.metadata or {}).get("transpile_seed"),
    }


_UNCOMPARED = ("transpile_seed",)


def template_mismatch(saved: dict | None, current: dict) -> list[str]:
    """How a saved template_fingerprint differs from the current one; [] if it matches or nothing was saved."""
    if not saved:
        return []
    return [f"{key} {saved[key]} != {current[key]}" for key in current
            if key not in _UNCOMPARED and key in saved and saved[key] != current[key]]


def two_qubit_count(tqc: QuantumCircuit) -> int:
    return sum(1 for inst in tqc.data if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")


def line_qubits(tqc: QuantumCircuit) -> tuple[int, ...]:
    """Physical qubit measured into line_cl[0], [1], [2]."""
    found = {}
    for inst in tqc.data:
        if inst.operation.name != "measure":
            continue
        for register, index in tqc.find_bit(inst.clbits[0]).registers:
            if register.name == LINE_REGISTER:
                found[index] = tqc.find_bit(inst.qubits[0]).index
    return tuple(found[b] for b in sorted(found))


def _prepare_line_index(index: int) -> QuantumCircuit:
    """Prepare one line index on a bare 3-qubit line register and measure it."""
    line_reg = QuantumRegister(3, name="line")
    line_cl = ClassicalRegister(3, name=LINE_REGISTER)
    qc = QuantumCircuit(line_reg, line_cl, name=f"readout_{index}")
    for q, bit in enumerate(reversed(line_index_to_bitstring(index))):
        if bit == "1":
            qc.x(line_reg[q])
    qc.barrier()
    qc.measure(line_reg, line_cl)
    return qc


def _job_output(raw: Sequence[dict[str, int]], job_id: str | None) -> JobOutput:
    """line_cl counts per circuit, turned into counts per line index."""
    counts = []
    for circuit_counts in raw:
        per_line = [0] * NUM_LINES
        for bits, n in circuit_counts.items():
            per_line[bitstring_to_line_index(bits)] = n
        counts.append(tuple(per_line))
    return JobOutput(tuple(counts), job_id)


def _line_counts(pub_result) -> dict[str, int]:
    return getattr(pub_result.data, LINE_REGISTER).get_counts()
