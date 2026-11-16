"""Collect labeled data: run N random legal boards on a backend and save every result.

    python scripts/collect.py --backend fake_kingston --n 30 --seed 1
    python scripts/collect.py --backend ibm_kingston --n 30 --seed 1 --tag kingston-first
    python scripts/collect.py --backend ideal --mix natural --n 200

Every result is appended to results/runs.jsonl with its true label (no win,
single win, double win) and the tag, together with the calibration measured in
the same job. Real hardware asks for confirmation first; --dry-run asks IBM to
check the job and estimate its usage without running it.
"""

import argparse
import sys

from qiskit.providers.exceptions import QiskitBackendNotFoundError

from qttt.backends import ANCILLA_FLAGS, DEFAULT_TRANSPILE_SEEDS, ERROR_SUPPRESSION_OPTIONS, get_backend
from qttt.detector import DEFAULT_BUDGET, DEFAULT_SHOTS
from qttt.experiments import (
    DEFAULT_MAX_SHOTS,
    MISS_TARGET,
    MIXES,
    choose_shots,
    collect,
    default_tag,
    sample_boards,
    save_pooled_calibration,
    saved_profile,
    summarize,
)
from qttt.grover import VARIANTS
from qttt.noise import CALIBRATION_CIRCUITS, calibration_circuits, known_boards
from qttt.results import LABELS, RunStore


def _confirm(read) -> bool:
    if read("This uses your plan's device time. Continue? [y/N] ").strip().lower() in ("y", "yes"):
        return True
    print("Not sent.")
    return False


def _positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _shots(text):
    return text if text == "auto" else _positive(text)


def main(argv=None, read=input):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", default="ideal", help='"ideal", a fake IBM device, or a real IBM device')
    parser.add_argument("--n", type=_positive, default=30, help="number of boards (default 30)")
    parser.add_argument("--seed", type=int, default=None, help="seed for choosing boards and for simulators")
    parser.add_argument("--mix", choices=MIXES, default="balanced",
                        help="balanced split of no win / single / double (default) or their natural frequencies")
    parser.add_argument("--variant", choices=VARIANTS, default="player")
    parser.add_argument("--transpile-seeds", type=_positive, default=DEFAULT_TRANSPILE_SEEDS,
                        help=f"compiler seeds tried on IBM devices, best kept (default {DEFAULT_TRANSPILE_SEEDS})")
    parser.add_argument("--ancilla", choices=ANCILLA_FLAGS, default="auto",
                        help="extra qubit for a cheaper oracle: auto keeps the better compiled circuit (default)")
    parser.add_argument("--error-suppression", action="store_true",
                        help="real devices only: run with dynamical decoupling and gate twirling (default: off)")
    parser.add_argument("--shots", type=_shots, default=DEFAULT_SHOTS,
                        help=f'shots per board, or "auto": as many as a short calibration job says are needed for'
                             f' at most 1%% misses, up to --max-shots (default {DEFAULT_SHOTS}); the batch still'
                             f' calibrates in its own jobs at those shots')
    parser.add_argument("--max-shots", type=_positive, default=DEFAULT_MAX_SHOTS,
                        help=f"the most shots per board --shots auto may choose (default {DEFAULT_MAX_SHOTS})")
    parser.add_argument("--calibration", nargs="+", metavar="TAG", default=None,
                        help="use calibrations saved earlier with these tags (pooled) instead of calibrating in"
                             " the batch job: no calibration circuits are sent")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    parser.add_argument("--boards-per-job", type=_positive, default=None,
                        help="split into jobs of this many boards, each with its own calibration unless --calibration"
                             " is given (default: one job)")
    parser.add_argument("--tag", default=None, help="name for this batch in the store (default: a timestamp)")
    parser.add_argument("--store", default="results", help="store directory (default results/)")
    parser.add_argument("--yes", action="store_true", help="do not ask before sending to real hardware")
    parser.add_argument("--dry-run", action="store_true",
                        help="real IBM devices only: have IBM validate the first job and estimate its usage, without"
                             " running it on the device; nothing is saved")
    args = parser.parse_args(argv)

    boards = sample_boards(args.n, seed=args.seed, mix=args.mix)
    backend = get_backend(args.backend, seed=args.seed, transpile_seeds=args.transpile_seeds,
                          ancilla=ANCILLA_FLAGS[args.ancilla],
                          sampler_options=ERROR_SUPPRESSION_OPTIONS if args.error_suppression else None)
    store = RunStore(args.store)
    tag = args.tag or default_tag()
    ask = not backend.simulated and not args.yes
    shots = args.shots
    profile = "auto"
    if args.calibration:
        profile, records = saved_profile(store, args.calibration, backend, args.variant)
        print(f"Using {len(records)} saved calibration(s) ({', '.join(args.calibration)}): ε = {profile.epsilon:.3f},"
              f" line qubits {list(profile.line_qubits)}; the batch sends no calibration circuits.")
        if shots == "auto":
            needed = profile.shots_needed(MISS_TARGET, args.budget, args.max_shots)
            shots = needed or args.max_shots
            print(f"Shots from that calibration: {shots} per board"
                  + (f" for at most {MISS_TARGET:.0%} misses." if needed else
                     f" (the cap; the worst win is missed {profile.miss_rates(shots, args.budget).worst:.0%}"
                     f" of the time)."))
    elif shots == "auto":
        if args.dry_run:
            parser.error("--dry-run needs a fixed --shots: choosing shots runs a calibration job")
        print(f"Choosing shots: one calibration job on {backend.name},"
              f" {CALIBRATION_CIRCUITS} circuits x {DEFAULT_SHOTS} shots.")
        if ask and not _confirm(read):
            return 1
        # This profile only picks the shots; each batch job calibrates again at those shots (profile stays "auto").
        shots, chosen = choose_shots(backend, args.variant, args.budget, args.max_shots, store, tag)
        capped = chosen.shots_needed(MISS_TARGET, args.budget, args.max_shots, in_job=True) is None
        worst = chosen.with_calibration_shots(shots).miss_rates(shots, args.budget).worst
        print(f"Measured ε = {chosen.epsilon:.3f}, bit factors {', '.join(f'{f:.3f}' for f in chosen.bit_factors)}:"
              f" {shots} shots per board" + (f" (the cap; at {shots} shots the worst win is missed"
                                            f" {worst:.0%} of the time)"
                                            if capped else f" for at most {MISS_TARGET:.0%} misses") + ".")

    jobs = -(-len(boards) // (args.boards_per_job or len(boards)))
    calibration = 0 if args.backend == "ideal" or args.calibration else CALIBRATION_CIRCUITS * jobs
    circuits = len(boards) + calibration
    split = ", ".join(f"{label}: {sum(b.label == label for b in boards)}" for label in LABELS)
    print(f"{len(boards)} boards ({split})"
          f" + {calibration} calibration circuits on {backend.name}, {jobs} job(s), {shots} shots each"
          f" = {circuits * shots:,} shots.")
    if args.dry_run:
        return _dry_run(backend, boards[:args.boards_per_job or len(boards)], args, shots,
                        calibrate=calibration > 0)
    if ask and not _confirm(read):
        return 1

    if args.calibration:
        profile = save_pooled_calibration(store, profile, records, backend, args.variant, tag)
    results = collect(boards, backend, variant=args.variant, shots=shots, budget=args.budget, profile=profile,
                      store=store, tag=tag, boards_per_job=args.boards_per_job)
    profile = results[0].profile
    print(f"Saved {len(results)} results to {store.runs_file} (tag {tag}).")
    if profile is not None and profile.source != "ideal":
        print(f"Noise profile {profile.source}: ε = {profile.epsilon:.3f}")
    print(summarize(boards, results).text())
    return 0


def _dry_run(backend, boards, args, shots, calibrate):
    """Send the first job to IBM as a dry run: validated and its usage estimated, but not run, and nothing saved."""
    if backend.simulated:
        print("--dry-run is for real IBM devices; simulators run nothing remotely.")
        return 1
    known = known_boards(args.variant) if calibrate else ()
    extra = calibration_circuits(backend, args.variant, known)[0] if calibrate else []
    circuits = extra + [backend.search_circuit(args.variant, b.board, b.player if args.variant == "player" else None)
                        for b in boards]
    try:
        job = backend.dry_run(circuits, shots)
    except QiskitBackendNotFoundError as error:
        # IBM dry-runs on a "mock_" copy of the device, which not every plan can see (the Open plan cannot).
        print(f"IBM could not dry-run on {backend.name}: {error}. Dry runs use IBM's mock copy of the device,"
              " which this account does not have. Nothing was sent.")
        return 1
    print(f"Dry run {job.job_id()} of {len(circuits)} circuits x {shots} shots sent; it runs nothing on the device.")
    print(f"IBM's usage estimate: {job.usage_estimation}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
