"""Collect labeled data: run N random legal boards on a backend and save every result.

    python scripts/collect.py --backend fake_kingston --n 30 --seed 1
    python scripts/collect.py --backend ibm_kingston --n 30 --seed 1 --tag kingston-first
    python scripts/collect.py --backend ideal --mix natural --n 200

Every result is appended to results/runs.jsonl with its true label (no win,
single win, double win) and the tag, together with the calibration measured in
the same job. Real hardware asks for confirmation first.
"""

import argparse
import sys

from qttt.backends import ANCILLA_FLAGS, DEFAULT_TRANSPILE_SEEDS, get_backend
from qttt.detector import DEFAULT_BUDGET, DEFAULT_SHOTS
from qttt.experiments import (
    DEFAULT_MAX_SHOTS,
    MISS_TARGET,
    MIXES,
    choose_shots,
    collect,
    default_tag,
    sample_boards,
    summarize,
)
from qttt.grover import VARIANTS
from qttt.noise import CALIBRATION_CIRCUITS
from qttt.results import LABELS, RunStore


def _confirm(read) -> bool:
    if read("This uses your plan's device time. Continue? [y/N] ").strip().lower() in ("y", "yes"):
        return True
    print("Not sent.")
    return False


def _shots(text):
    return text if text == "auto" else int(text)


def main(argv=None, read=input):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", default="ideal", help='"ideal", a fake IBM device, or a real IBM device')
    parser.add_argument("--n", type=int, default=30, help="number of boards (default 30)")
    parser.add_argument("--seed", type=int, default=None, help="seed for choosing boards and for simulators")
    parser.add_argument("--mix", choices=MIXES, default="balanced",
                        help="balanced split of no win / single / double (default) or their natural frequencies")
    parser.add_argument("--variant", choices=VARIANTS, default="player")
    parser.add_argument("--transpile-seeds", type=int, default=DEFAULT_TRANSPILE_SEEDS,
                        help=f"compiler seeds tried on IBM devices, best kept (default {DEFAULT_TRANSPILE_SEEDS})")
    parser.add_argument("--ancilla", choices=ANCILLA_FLAGS, default="auto",
                        help="extra qubit for a cheaper oracle: auto keeps the better compiled circuit (default)")
    parser.add_argument("--no-error-suppression", action="store_true",
                        help="real devices only: run without the default dynamical decoupling and gate twirling")
    parser.add_argument("--shots", type=_shots, default=DEFAULT_SHOTS,
                        help=f'shots per board, or "auto": as many as a short calibration job says are needed for'
                             f' at most 1%% misses, up to --max-shots (default {DEFAULT_SHOTS})')
    parser.add_argument("--max-shots", type=int, default=DEFAULT_MAX_SHOTS,
                        help=f"the most shots per board --shots auto may choose (default {DEFAULT_MAX_SHOTS})")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    parser.add_argument("--boards-per-job", type=int, default=None,
                        help="split into jobs of this many boards, each with its own calibration (default: one job)")
    parser.add_argument("--tag", default=None, help="name for this batch in the store (default: a timestamp)")
    parser.add_argument("--store", default="results", help="store directory (default results/)")
    parser.add_argument("--yes", action="store_true", help="do not ask before sending to real hardware")
    args = parser.parse_args(argv)

    boards = sample_boards(args.n, seed=args.seed, mix=args.mix)
    backend = get_backend(args.backend, seed=args.seed, transpile_seeds=args.transpile_seeds,
                          ancilla=ANCILLA_FLAGS[args.ancilla],
                          sampler_options={} if args.no_error_suppression else None)
    store = RunStore(args.store)
    tag = args.tag or default_tag()
    ask = not backend.simulated and not args.yes
    shots = args.shots
    if shots == "auto":
        print(f"Choosing shots: one calibration job on {backend.name},"
              f" {CALIBRATION_CIRCUITS} circuits x {DEFAULT_SHOTS} shots.")
        if ask and not _confirm(read):
            return 1
        shots, profile = choose_shots(backend, args.variant, args.budget, args.max_shots, store, tag)
        capped = profile.shots_needed(MISS_TARGET, args.budget, args.max_shots) is None
        print(f"Measured ε = {profile.epsilon:.3f}, bit factors {', '.join(f'{f:.3f}' for f in profile.bit_factors)}:"
              f" {shots} shots per board" + (f" (the cap; at {shots} shots the worst win is missed"
                                            f" {profile.miss_rates(shots, args.budget).worst:.0%} of the time)"
                                            if capped else f" for at most {MISS_TARGET:.0%} misses") + ".")

    jobs = -(-len(boards) // (args.boards_per_job or len(boards)))
    calibration = 0 if args.backend == "ideal" else CALIBRATION_CIRCUITS * jobs
    circuits = len(boards) + calibration
    split = ", ".join(f"{label}: {sum(b.label == label for b in boards)}" for label in LABELS)
    print(f"{len(boards)} boards ({split})"
          f" + {calibration} calibration circuits on {backend.name}, {jobs} job(s), {shots} shots each"
          f" = {circuits * shots:,} shots.")
    if ask and not _confirm(read):
        return 1

    results = collect(boards, backend, variant=args.variant, shots=shots, budget=args.budget,
                      store=store, tag=tag, boards_per_job=args.boards_per_job)
    profile = results[0].profile
    print(f"Saved {len(results)} results to {store.runs_file} (tag {store.load('detection')[-1]['tag']}).")
    if profile is not None and profile.source != "ideal":
        print(f"Noise profile {profile.source}: ε = {profile.epsilon:.3f}")
    print(summarize(boards, results).text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
