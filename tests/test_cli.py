"""The qttt command, driven with scripted input."""

import pytest

from qttt import cli
from qttt.backends import ERROR_SUPPRESSION_OPTIONS, IdealBackend


class _NoSignalBackend(IdealBackend):
    """Ideal backend whose counts carry no Grover signal, as if noise had erased it."""

    def run(self, circuits, shots, on_submit=None):
        out = super().run(circuits, shots, on_submit)
        return type(out)(tuple(tuple(shots // 8 for _ in range(8)) for _ in out.counts), out.job_id)


def _run(argv, moves):
    """Run the CLI with scripted input; returns (exit code, all output)."""
    inputs = iter(moves)
    output = []

    def read(prompt):
        output.append(prompt)
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError from None

    code = cli.main(argv, read=read, write=output.append)
    return code, "\n".join(output)


def test_two_player_game_to_a_win():
    code, out = _run(["--seed", "4903", "--shots", "256"], ["0", "3", "1", "4", "2"])
    assert code == 0
    assert "Quantum check, turn 5 (X just moved)" in out
    assert "X wins (top row)." in out


def test_bad_moves_are_asked_again():
    code, out = _run(["--seed", "4903", "--shots", "256"], ["9", "x", "0", "0", "3", "1", "4", "2"])
    assert code == 0
    assert out.count("Pick an empty cell") == 3
    assert "X wins" in out


@pytest.mark.parametrize("moves", [["q"], []], ids=["q", "end_of_input"])
def test_quit(moves):
    code, out = _run([], moves)
    assert code == 0
    assert out.endswith("Quit.")


def test_computer_opponent():
    code, out = _run(["--seed", "4903", "--shots", "256", "--computer", "O"], ["0", "1", "2", "3", "5", "6", "7", "8"])
    assert code == 0
    assert "Computer (O) plays 4." in out  # rules strategy takes the center
    assert "wins" in out or "Draw." in out


def test_missed_win_is_reported(monkeypatch):
    monkeypatch.setattr(cli, "get_backend", lambda name, seed=None, **kwargs: _NoSignalBackend(seed=seed))
    code, out = _run(["--shots", "256"], ["0", "3", "1", "4", "2"])
    assert code == 0
    assert "MISSED: X has [0]" in out
    assert "X wins (top row)." in out
    assert "missed a win on 1 turn(s)" in out


def test_verify_after_two_games(tmp_path):
    code, out = _run(["--seed", "4903", "--shots", "256", "--games", "2", "--verify", "ideal",
                      "--save", str(tmp_path)],
                     ["0", "3", "1", "4", "2", "0", "3", "1", "4", "2"])
    assert code == 0
    assert "Game 2 of 2" in out
    assert "Verifying 2 board(s) on ideal in one job" in out
    assert "2 board(s), 2 of 2 win(s) found" in out
    from qttt.results import RunStore

    assert len(RunStore(tmp_path).load("detection")) == 4  # 2 played + 2 verified


def test_real_device_plays_on_its_snapshot_and_asks_before_sending(monkeypatch):
    names = []

    def fake_get_backend(name, seed=None, **kwargs):
        names.append(name)
        return IdealBackend(seed=seed)

    monkeypatch.setattr(cli, "get_backend", fake_get_backend)
    monkeypatch.setattr(cli, "fake_twin", lambda name: "fake_kingston")
    code, out = _run(["--backend", "ibm_kingston", "--shots", "64"], ["0", "3", "1", "4", "2", "n"])
    assert code == 1
    assert "playing on fake_kingston, then verifying every board on ibm_kingston" in out
    assert "Not sent." in out
    assert names == ["fake_kingston", "ibm_kingston"]  # connected up front (no job sent), so a bad account fails early


def test_missing_account_fails_before_the_games(monkeypatch):
    def fake_get_backend(name, seed=None, **kwargs):
        if name == "ibm_kingston":
            raise RuntimeError("no IBM account")
        return IdealBackend(seed=seed)

    monkeypatch.setattr(cli, "get_backend", fake_get_backend)
    monkeypatch.setattr(cli, "fake_twin", lambda name: "fake_kingston")
    with pytest.raises(RuntimeError, match="no IBM account"):
        _run(["--backend", "ibm_kingston"], [])  # raises before asking for a single move


def test_confirmation_names_the_save_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "get_backend", lambda name, seed=None, **kwargs: IdealBackend(seed=seed))
    monkeypatch.setattr(cli, "fake_twin", lambda name: "fake_kingston")
    _, out = _run(["--backend", "ibm_kingston", "--shots", "64", "--save", str(tmp_path / "mine")],
                  ["0", "3", "1", "4", "2", "n"])
    assert f"saved to {tmp_path / 'mine'}/" in out


def test_nothing_to_verify():
    code, out = _run(["--verify", "ideal"], ["0", "1", "q"])
    assert code == 0
    assert "no game reached turn 5" in out


@pytest.mark.parametrize("argv", [["--shots", "0"], ["--budget", "9"], ["--games", "0"]])
def test_bad_options(argv):
    code, out = _run(argv, [])
    assert code == 2


def test_fake_twin():
    assert cli.fake_twin("ibm_kingston") == "fake_kingston"
    assert cli.fake_twin("ibm_not_a_device") is None


def test_render_board_numbers_empty_cells():
    assert cli.render_board(["X", " ", "O", " ", " ", " ", " ", " ", " "]).startswith(" X | 1 | O")


@pytest.mark.parametrize("flag, setting", [("auto", "auto"), ("on", True), ("off", False)])
def test_compile_flags_reach_the_backend(monkeypatch, flag, setting):
    seen = []

    def fake_get_backend(name, seed=None, **kwargs):
        seen.append(kwargs)
        return IdealBackend(seed=seed)

    monkeypatch.setattr(cli, "get_backend", fake_get_backend)
    code, _ = _run(["--transpile-seeds", "3", "--ancilla", flag], ["q"])
    assert code == 0
    assert seen == [{"transpile_seeds": 3, "ancilla": setting, "sampler_options": None}]


def test_default_transpile_seeds(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "get_backend", lambda name, seed=None, **kwargs: seen.append(kwargs) or IdealBackend())
    _run([], ["q"])
    assert seen == [{"transpile_seeds": 5, "ancilla": "auto", "sampler_options": None}]

    seen.clear()
    _run(["--error-suppression"], ["q"])
    assert seen[0]["sampler_options"] == ERROR_SUPPRESSION_OPTIONS


def test_bad_transpile_seeds_are_refused():
    code, out = _run(["--transpile-seeds", "0"], [])
    assert code == 2 and "--transpile-seeds" in out


def test_digits_that_are_not_cell_numbers_are_asked_again():
    code, out = _run(["--seed", "4903", "--shots", "256"], ["²", "٣", "0", "3", "1", "4", "2"])
    assert code == 0
    assert out.count("Pick an empty cell") == 2
