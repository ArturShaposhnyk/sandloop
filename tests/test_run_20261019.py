"""Regressions from the first live run debug/runs/20261019-162703 (L186: chests, mystery buckets,
a wall). 9 good taps, then the bot froze at 4/5 logging "looks face-up but the path rule says
closed" for every cell.

Cause: once the bot stopped tapping, the boosters below the grid pulsed into the grid zone in
every other run of frames; the grid region then reached the boosters, the lattice got a 7th row
with a smaller pitch and every cell was read across two rows ("pink/darkgreen" towers, boosters
as buckets). The correct 6-row frames chose r4c0 darkgreen, the 7-row ones nothing, so the
2-frame confirmation never completed.

Frames: dataset shots 057 (start) / 058 (the frozen state, 6 rows), and in
tests/data/run-20261019-162703/: 00009 (chest r4c1 closed) -> 00018 (it opened and scattered).
"""
import dataclasses
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from hints import LatticeLock  # noqa: E402
from state import LevelState  # noqa: E402
from strategy import choose  # noqa: E402
from tests.test_bot import Controls, Failsafe, Tapper, make_bot, run  # noqa: E402
from tests.test_state import analyzer  # noqa: E402,F401  (fixture)
from runlog import load_events  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "dataset" / "screens"
DATA = Path(__file__).resolve().parent / "data" / "run-20261019-162703"
S057 = SHOTS / "20261019-162708-850-057-live.png"
S058 = SHOTS / "20261019-162746-400-058-live.png"


def img(path):
    out = cv2.imread(str(path))
    assert out is not None, path
    return out


def bridged(frame):
    """The pulsing booster under column 1 joins the grid region (what the live run saw)."""
    out = frame.copy()
    cv2.rectangle(out, (80, 1095), (140, out.shape[0]), (235, 215, 200), -1)
    return out


def read(an, frame, lattices=None):
    from vision import read_gameplay

    return read_gameplay(frame, an.palette, an.cfg.vision, an.reader, log=lambda s: None,
                         icons=an.icons, hints=an.hints, lattices=lattices)


def test_booster_joining_the_grid_shifts_every_row_without_the_lattice_lock(analyzer):  # noqa: F811
    clean = read(analyzer, img(S058))
    bad = read(analyzer, bridged(img(S058)))
    assert clean.lattice.rows == 6 and bad.lattice.rows == 7
    assert clean.labels() != bad.labels()[:6]
    assert any("/" in lab for lab in " ".join(bad.labels()).split())   # phantom towers
    assert bad.disagreements                                           # the frozen log


def test_lattice_lock_keeps_the_levels_grid(analyzer):  # noqa: F811
    lock = LatticeLock(lock_frames=1)
    clean = read(analyzer, img(S058), lock)
    again = read(analyzer, bridged(img(S058)), lock)
    assert again.lattice.rows == 6 and again.labels() == clean.labels()
    assert not again.disagreements


def test_lattice_lock_is_never_replaced_mid_level():
    """Live run 20261020-125351: a reading held 3 frames used to replace the lock, and the
    lattice flip-flopped 7x7 <-> 4x4. Now only unlock() (a non-gameplay screen) or another
    frame size frees it."""
    from vision import Lattice, VisionError
    import numpy as np

    logs = []
    lock = LatticeLock(lock_frames=3)
    a = Lattice(10, 100, 50, 45, np.ones((6, 7), bool))
    b = Lattice(30, 140, 40, 36, np.ones((8, 9), bool))   # another level's grid
    taller = Lattice(10, 100, 50, 45, np.ones((7, 7), bool))
    misfit = Lattice(10, 100, 50, 45, np.ones((6, 7), bool), fit=0.3)
    with pytest.raises(VisionError, match="walls off"):
        lock.check(misfit, logs.append)                       # never locked on a misread
    assert lock.check(a, logs.append) is a and lock.locked    # first fitting reading: locked
    assert lock.check(taller, logs.append) is a               # extra row at the bottom
    for _ in range(10):
        assert lock.check(b, logs.append) is a                # any number of frames: kept
    assert lock.fixed
    assert sum("read with the level's" in s for s in logs) == 1   # logged once per streak
    lock.unlock()                                             # non-gameplay screen
    assert lock.check(b, logs.append) is b


def test_lattice_lock_takes_a_finer_reading_only_at_the_start():
    """User, 2026-10-20: the initial lock must prefer the finer grid. A coarse first reading
    (2x pitch over the same outline, or extra rows from the boosters) is replaced by the finer
    one within lock_frames frames; after that nothing replaces the lock."""
    from vision import Lattice
    import numpy as np

    fine = Lattice(101.0, 748.0, 56.6, 50.1, np.ones((7, 7), bool))
    coarse = Lattice(101.0, 748.0, 99.0, 87.6, np.ones((4, 4), bool))
    taller = Lattice(101.0, 748.0, 56.6, 50.1, np.ones((8, 7), bool))
    for first in (coarse, taller):
        lock = LatticeLock(lock_frames=3)
        assert lock.check(first, lambda s: None) is first
        assert lock.check(first, lambda s: None) is first
        assert lock.check(fine, lambda s: None) is fine       # finer, inside the window
        for _ in range(3):
            assert lock.check(first, lambda s: None) is fine
        assert lock.fixed
    lock = LatticeLock(lock_frames=1)
    lock.check(coarse, lambda s: None)
    lock.check(coarse, lambda s: None)                        # window over: coarse is fixed
    assert lock.check(fine, lambda s: None) is coarse


def test_lattice_lock_frees_on_another_frame_size():
    from vision import Lattice
    import numpy as np

    lock = LatticeLock(lock_frames=1)
    a = Lattice(10, 100, 50, 45, np.ones((6, 7), bool))
    b = Lattice(60, 300, 75, 67, np.ones((6, 7), bool))
    assert lock.check(a, lambda s: None, (1298, 598)) is a
    assert lock.check(b, lambda s: None, (1298, 598)) is a
    assert lock.check(b, lambda s: None, (1947, 897)) is b    # window scaled: new geometry


def test_l186_cells(analyzer):  # noqa: F811
    gv = read(analyzer, img(S057))
    g = gv.grid
    assert [g[4][1].label(), g[4][5].label(), g[5][1].label(), g[5][5].label()] == \
        ["chest4", "chest4", "mystery", "mystery"]
    assert g[3][3] is None                                   # the wall: no cell, blocks paths
    st = LevelState.from_view(gv)
    assert sorted(l.kind for l in st.locks if l.kind in ("chest", "mystery")) == \
        ["chest", "chest", "mystery", "mystery"]
    # r3c1 magenta (the run's tap #2) opens the chest under it: priced, not forbidden
    d = choose(st, analyzer.cfg.strategy)
    cand = next(c for c in d.candidates if c.pos == (3, 1))
    assert cand.chests == [((4, 1), 4)] and cand.scatter is not None


def test_frozen_state_taps_darkgreen_on_the_idle_belt(analyzer):  # noqa: F811
    """058 with the belt as at t=64 s (every colour stalled): the 6-row read chose r4c0."""
    st = LevelState.from_view(read(analyzer, img(S058)))
    st.belt_stalled = {"cream": 59.9, "orange": 18.8, "pink": 11.3, "red": 15.1}
    d = choose(st, analyzer.cfg.strategy)
    assert d.move is not None and (d.move.pos, d.move.color) == ((4, 0), "darkgreen")


def test_bot_taps_through_the_booster_flicker(tmp_path):
    """The live loop on alternating clean / bridged frames: with the lattice lock both read the
    same grid, the move confirms and is tapped (it never was in the run)."""
    from analyzer import Analyzer
    from config import load_config

    cfg = load_config()
    cfg.bot = dataclasses.replace(cfg.bot, settle_min_s=0.0, settle_max_s=0.0, speed_x2=False,
                                  retap_guard_s=0.0, loop_interval_s=0.05)
    cfg.strategy = dataclasses.replace(cfg.strategy, stall_s=0.0)   # the belt sat still for 20 s+
    an = Analyzer(cfg)
    clean = img(S058)
    frames = [clean, bridged(clean)] * 3
    tapper = Tapper()
    bot, log = make_bot(cfg, an, tmp_path, frames, tapper=tapper, max_taps=1)
    run(bot, log)
    taps = [e for e in load_events(log.dir) if e["type"] == "tap"]
    assert len(taps) == 1 and (tuple(taps[0]["pos"]), taps[0]["color"]) == ((4, 0), "darkgreen")


def test_analyzer_rereads_after_a_chest_opens(analyzer):  # noqa: F811
    before, after = img(DATA / "00009-tap.png"), img(DATA / "00018-tap.png")
    logs = []
    analyzer.reset_level()
    try:
        r0 = analyzer.analyze(before, log=logs.append, now=0.0)
        assert r0.view.grid[4][1].label() == "chest4"
        r1 = analyzer.analyze(after, log=logs.append, now=1.0)
        assert any("chest r4c1 (4) opened" in s for s in logs)
        assert r1.decision.move is None and "chest opened" in r1.decision.reason
        settled = 1.0 + analyzer.cfg.strategy.chest_settle_s
        r2 = analyzer.analyze(after, log=logs.append, now=settled)
        assert "chest opened" not in r2.decision.reason
    finally:
        analyzer.reset_level(track=False)


class _Stuck:
    """Analyzer stand-in: one real read of 058 with a cell forced to disagree and no move."""

    def __init__(self, analyzer):
        from strategy import Decision

        self.res = analyzer.analyze(img(S058), log=lambda s: None)
        self.res.view.disagreements = [(3, 0)]
        self.res.decision = Decision(None, "wait: no bucket passes the rules",
                                     self.res.decision.candidates)

    def analyze(self, frame, log=print, now=None, adjust=None):
        return self.res


def test_bot_reports_a_cell_that_keeps_disagreeing(analyzer, tmp_path):  # noqa: F811
    from config import load_config

    cfg = load_config()
    cfg.bot = dataclasses.replace(cfg.bot, loop_interval_s=0.05, disagree_report_s=0.5)
    frames = [img(S058)] * 30
    bot, log = make_bot(cfg, _Stuck(analyzer), tmp_path, frames, tapper=Tapper(),
                        controls=Controls(), failsafe=Failsafe())
    run(bot, log)
    stuck = [e for e in load_events(log.dir) if e["type"] == "stuck"]
    assert stuck and stuck[0]["cells"] == ["r3c0 pink-up"]
    assert (log.dir / stuck[0]["png"]).exists()
    assert "STUCK" in (log.dir / "bot.log").read_text()
    assert len(stuck) < 5                                   # repeated every 0.5 s, not per frame
