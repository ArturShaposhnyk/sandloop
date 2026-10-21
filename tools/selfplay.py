"""Self-play on the recorded start boards (redesign phase C gate; phase D tunes with it).

A **world** is the fitted simulator (sim.py) started on a recorded level start frame
(debug/replay/cache, tools/replay.py's frame reads), with what the board hides drawn once per
seed (worlds.py; colours an attempt saw under locks - levels.jsonl "revealed" - are the truth
there) and its own belt speed per colour (x lognormal(0, ``--speed-sigma``)), so the deciders'
model is never exactly right. Both deciders see what vision would: the grid (opened cells show
their colour), the k counter, the belt fills (each reading hidden with ``--hide``), sand /
contact / lower-band shares of the picture. A decision taps (the first tap costs
``planner.tap_s`` belt seconds, burst follow-ups ``planner.burst_tap_s``) or waits one second.

* **old** = strategy.choose (rules + lookahead) + levelplan.PlanRunner (plan, re-plans) +
  tempo.burst, the belt stall clock from state.BeltTracker - as bot.py live.
* **search** = decider.SearchDecider (belief + search).

Won = every bucket filled or at most 0.5 % of the sand left (the simulated capacities are
rounded, the last bucket may miss a few cells); lost = belt full and nothing collected for
``--lost-s``, or no move and nothing collected for ``--lost-s``; else ``--max-s``.

    python -m tools.selfplay                      # L187 L189 L190, 8 seeds, both deciders
    python -m tools.selfplay L189 --seeds 20 --decider search --jobs 8
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# level -> (run, start frame = the level's first tap frame, levels.jsonl attempt for "revealed")
SCENARIOS: Dict[str, Tuple[str, int, int]] = {
    "L187": ("20261021-000105", 9, 1),      # the won L187 run
    "L189": ("20261021-112241", 234, 2),    # the won L189 attempt
    "L190": ("20261021-143729", 5, 1),
}


# how the world differs from the deciders' model: "model" = the fitted simulator itself
# (pessimistic: gain recall 51-64 %, tools/replay.py), "medium" / "generous" = buckets collect
# below the fitted 5 % contact share and from deeper in the picture (the live win rates of the
# old decider, ~1 in 5 on L187 and L189, lie between)
PRESETS: Dict[str, Dict[str, float]] = {
    "model": {},
    "medium": {"fill_min_exposed": 0.02, "contact": 1.5},
    "generous": {"fill_min_exposed": 0.0, "contact": 2.0},
    # phase D fit (tools/simfit.py): reproduces the 10 real outcomes (config decider.model_*)
    "fitted": {"fill_rate": 600.0, "fill_max": 40.0, "fill_min_exposed": 0.0,
               "reach_delay_s": 3.5, "finish_tol": 0.03, "contact": 3.0},
}


@dataclass
class Episode:
    level: str
    seed: int
    decider: str
    world: str = "custom"
    result: str = "timeout"
    t: float = 0.0
    taps: int = 0
    collected: float = 0.0
    idle_s: float = 0.0          # waited while no belt bucket gained sand and a tap was legal
    decide_s: float = 0.0        # wall time spent deciding
    log: List[str] = field(default_factory=list)


# --- the world -----------------------------------------------------------------------------------


def load_start(cfg, level: str):
    from tools.replay import Reader, Start

    run, frame, _ = SCENARIOS[level]
    rd = Path(cfg.paths.debug) / "runs" / run
    path = next((p for p in (rd / "frames").glob(f"{frame:05d}-*.png")
                 if "annotated" not in p.name), rd / "frames" / f"{frame:05d}-tap.png")
    reader = Reader(cfg, Path(cfg.paths.debug) / "replay" / "cache", cell_px=cfg.planner.cell_px)
    state, pic = reader.read(Start(run, frame, 0.0, path, True))
    if state is None or pic is None:
        raise SystemExit(f"{level}: start frame {path} not readable")
    return state, pic


def truth(cfg, level: str) -> Dict[Tuple[int, int], Tuple[str, Optional[str]]]:
    """Colours seen under locks in the recorded attempt (levels.jsonl "revealed")."""
    run, _, attempt = SCENARIOS[level]
    path = Path(cfg.paths.debug) / "levels.jsonl"
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("run") == run and r.get("level") == attempt:
            out = {}
            for key, label in (r.get("revealed") or {}).items():
                row, col = (int(x) for x in key.split(","))
                top, _, body = label.partition("/")
                out[(row, col)] = (top, body or None)
            return out
    return {}


def make_world(cfg, state, pic, seed: int, sigma: float, memory,
               world: Optional[Dict[str, float]] = None):
    """``world``: SimConfig fields (fill_rate, fill_min_exposed, fill_max) and ``contact``
    (x the picture's contact layer) of the world, if it should differ from the model."""
    import dataclasses

    from levelplan import sim_config
    from worlds import worlds

    world = dict(world or {})
    contact = world.pop("contact", 1.0)
    sc = dataclasses.replace(sim_config(cfg.planner, state.conveyor_slots, True), **world)
    pic = pic.copy()
    pic.contact = max(1, round(pic.contact * contact))
    pic._layer = None
    world = worlds(state, pic, sc, 1, 100_000 + seed, memory=memory)[0]
    rng = random.Random(seed)
    world.mult = {c: math.exp(rng.gauss(0.0, sigma)) for c in pic.names}
    world.grid.disagreements = []
    world.grid.tappable = world.grid.playable()
    return world


class Observer:
    """What vision would read of the world: a LevelState per frame."""

    def __init__(self, cfg, start, world, hide: float, seed: int) -> None:
        self.cfg, self.hide = cfg, hide
        self.rng = random.Random(seed * 31 + 7)
        self.area = world.pic.total() / max(1e-9, sum(start.sand.values()))
        self.height = max((len(c) for c in world.pic.cols), default=1)

    def __call__(self, world):
        from state import ConveyorBucket

        st = world.grid.fork()
        st.conveyor_k = len(world.belt)
        st.conveyor_slots = world.sc.slots
        st.conveyor = []
        for s in world.belt:
            cap = max(1, world.capacity.get(s.colour, 1))
            fill = int(100 * (1 - s.left / cap))
            if self.rng.random() < self.hide:
                continue              # off screen / label not read
            st.conveyor.append(ConveyorBucket(s.colour, fill, stages=s.stages[s.stage + 1:]))
        pic = world.pic
        names = [n for n in pic.names if n != "locked"]
        st.sand = {c: pic.count(c) / self.area for c in names if pic.count(c) > 0}
        st.exposed = {c: pic.share(c) for c in names if pic.share(c) > 0.01}
        band = max(1, round(self.cfg.vision.lower_band * self.height))
        low = {c: 0 for c in names}
        for col in pic.cols:
            for i in col[:band]:
                if pic.names[i] in low:
                    low[pic.names[i]] += 1
        size = band * max(1, len(pic.cols))
        st.lower = {c: n / size for c, n in low.items() if n / size > 0.01}
        st.above = above(pic)
        st.disagreements, st.belt_stalled = [], {}
        st.tappable = st.playable()
        st.shells = []
        return st


def above(pic) -> Dict[str, Dict[str, float]]:
    """sand.contact_layers' "above" from the stacks: contact colour -> colour right above it
    -> share of the contact layer it becomes once collected."""
    k = pic.contact
    total = sum(min(k, len(c)) for c in pic.cols)
    out: Dict[str, Dict[str, float]] = {}
    for col in pic.cols:
        nxt = col[k:2 * k]
        if not nxt:
            continue
        for i in set(col[:k]):
            cc = col[:k].count(i)
            row = out.setdefault(pic.names[i], {})
            for j in set(nxt):
                row[pic.names[j]] = row.get(pic.names[j], 0.0) + cc * nxt.count(j) / len(nxt) / total
    return {c: {d: v for d, v in r.items() if v > 0.002} for c, r in out.items() if r}


# --- deciders ------------------------------------------------------------------------------------


class OldDriver:
    def __init__(self, cfg, level: str, memory=None) -> None:
        from levelplan import PlanRunner
        from state import BeltTracker

        class Runner(PlanRunner):
            def picture(self, img, state):
                return self.world_pic.copy()

        self.cfg = cfg
        self.tracker = BeltTracker(cfg.strategy.stall_min_growth, cfg.strategy.stall_forget_s)
        self.runner = Runner(cfg, None) if cfg.planner.enabled else None
        if self.runner is not None:
            self.runner.signature = f"selfplay-{level}"
            self.runner.x2 = True
            self.runner.preset_hidden = dict(memory or {})
        self.capacity = None

    def decide(self, st, world, now: float, log) -> List[Tuple[int, int]]:
        from analyzer import colour_capacity
        from strategy import choose
        from tempo import burst

        st.belt_stalled = self.tracker.update(st.conveyor, now, st.conveyor_k)
        st.belt_tracked = self.tracker.tracked
        if self.capacity is None:
            self.capacity = colour_capacity(st)
        st.colour_capacity = dict(self.capacity)
        d = choose(st, self.cfg.strategy)
        if self.runner is not None:
            self.runner.world_pic = world.pic
            d = self.runner.decide(None, None, st, d, now, log)
        if d.move is None:
            return []
        if d.move.level_plan:
            out = [d.move.pos]
            for f in d.move.follow:
                if f.pos not in st.tappable:
                    break
                out.append(f.pos)
            return out
        return [c.pos for c, _ in burst(st, d.move, set(st.tappable), self.cfg.strategy,
                                        self.cfg.bot, {})]

    def tapped(self, pos, colour, stages, now) -> None:
        if self.runner is not None:
            self.runner.tapped(pos)


class SearchDriver:
    def __init__(self, cfg, level: str, memory=None) -> None:
        from decider import SearchDecider

        self.d = SearchDecider(cfg)
        self.d.signature, self.d.x2 = f"selfplay-{level}", True
        self.d.memory = dict(memory or {})

    def decide(self, st, world, now: float, log) -> List[Tuple[int, int]]:
        self.d.observe(st, now)
        dec = self.d.decide(st, lambda: world.pic.copy(), now, log)
        if dec.move is None:
            return []
        return [dec.move.pos] + [f.pos for f in dec.move.follow if f.pos in st.tappable]

    def tapped(self, pos, colour, stages, now) -> None:
        self.d.tapped(pos, colour, stages, now)


# --- one episode ---------------------------------------------------------------------------------


def play(cfg, level: str, seed: int, decider: str, start=None, hide: float = 0.2,
         sigma: float = 0.25, max_s: float = 900.0, lost_s: float = 20.0,
         verbose: bool = False, world_set: Optional[Dict[str, float]] = None,
         memory: bool = False) -> Episode:
    """``memory``: the deciders know the colours under the locks (a Try Again with level
    memory, as L189's won attempt 2)."""
    from cells import Kind

    state, pic = start or load_start(cfg, level)
    known = truth(cfg, level)
    world = make_world(cfg, state, pic, seed, sigma, known, world_set)
    see = Observer(cfg, state, world, hide, seed)
    mem = known if memory else None
    drv = OldDriver(cfg, level, mem) if decider == "old" else SearchDriver(cfg, level, mem)
    ep = Episode(level, seed, decider)
    total = max(1, world.pic.total())
    pc = cfg.planner
    now, last_gain = 0.0, 0.0
    logs: List[str] = []
    log = logs.append if verbose else (lambda s: None)
    while now < max_s:
        st = see(world)
        t0 = time.monotonic()
        taps = drv.decide(st, world, now, log)
        ep.decide_s += time.monotonic() - t0
        done = 0
        for i, pos in enumerate(taps):
            cell = world.grid.cell(pos)
            if (world.free <= 0 or pos not in world.grid.playable() or cell is None
                    or cell.kind is not Kind.BUCKET or not cell.color):
                break
            stages = (cell.body,) if cell.body else ()
            world.place(pos)
            drv.tapped(pos, cell.color, stages, now)
            ep.taps += 1
            done += 1
            ep.log.append(f"{now:5.0f}s tap r{pos[0]}c{pos[1]} {cell.color}"
                          + (f"/{cell.body}" if cell.body else "") + f" belt {world.belt_fills()}")
            for _ in range(round((pc.tap_s if i == 0 else pc.burst_tap_s) / world.sc.lap_s)):
                if world.lap() > 0:
                    last_gain = now
                now += world.sc.lap_s
        if not done:
            if now - last_gain >= 3.0 and world.free > 0 and st.tappable:
                ep.idle_s += world.sc.lap_s
            if world.lap() > 0:
                last_gain = now
            now += world.sc.lap_s
        if world.won or world.pic.total() <= 0.005 * total:   # capacity rounding: a few cells
            ep.result = "win"
            break
        if now - last_gain >= lost_s and (world.free <= 0 or not world.grid.playable()):
            ep.result = "lost"
            break
    ep.t, ep.collected = now, world.collected / total
    ep.log += logs
    return ep


def _job(args):
    from config import load_config

    level, seed, decider, opts = args[:4]
    cfg = load_config()
    for k, v in opts.get("set", {}).items():
        sect, name = k.split(".", 1)
        setattr(getattr(cfg, sect), name, type(getattr(getattr(cfg, sect), name))(v))
    world = args[4] if len(args) > 4 else None
    ep = play(cfg, level, seed, decider, hide=opts["hide"], sigma=opts["sigma"],
              max_s=opts["max_s"], lost_s=opts["lost_s"],
              world_set=PRESETS[world] if world else opts.get("world"),
              memory=opts.get("memory", False))
    ep.world = world or "custom"
    return ep


def summarize(eps: Sequence[Episode]) -> List[str]:
    lines = []
    for world, level, dec in sorted({(e.world, e.level, e.decider) for e in eps}):
        if True:
            mine = [e for e in eps if (e.world, e.level, e.decider) == (world, level, dec)]
            wins = sum(e.result == "win" for e in mine)
            n = len(mine)
            lines.append(
                f"{world:8s} {level} {dec:6s}: won {wins}/{n}, collected {sum(e.collected for e in mine) / n:.0%}"
                f", taps {sum(e.taps for e in mine) / n:.0f}, {sum(e.t for e in mine) / n:.0f}s"
                f" belt time, idle {sum(e.idle_s for e in mine) / n:.0f}s, decide "
                f"{sum(e.decide_s for e in mine) / n:.0f}s wall  ["
                + " ".join(f"{e.seed}:{e.result[0]}" for e in sorted(mine, key=lambda e: e.seed))
                + "]")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("levels", nargs="*", default=sorted(SCENARIOS))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--decider", choices=("old", "search", "both"), default="both")
    ap.add_argument("--hide", type=float, default=0.2, help="share of belt readings hidden")
    ap.add_argument("--speed-sigma", type=float, default=0.25)
    ap.add_argument("--max-s", type=float, default=900.0)
    ap.add_argument("--lost-s", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--memory", action="store_true",
                    help="the deciders know the colours under the locks (level memory)")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE",
                    help="config override for the deciders")
    ap.add_argument("--world", action="append", default=[], metavar="FIELD=VALUE",
                    help="the world's fill_rate / fill_min_exposed / fill_max / contact (x)")
    ap.add_argument("--preset", action="append", default=[], choices=sorted(PRESETS),
                    help="world presets to play (instead of --world), e.g. all three")
    ap.add_argument("--out", help="JSON (default debug/selfplay/<stamp>.json)")
    args = ap.parse_args(argv)
    deciders = ["old", "search"] if args.decider == "both" else [args.decider]
    opts = {"hide": args.hide, "sigma": args.speed_sigma, "max_s": args.max_s,
            "lost_s": args.lost_s, "set": dict(s.split("=", 1) for s in args.set),
            "world": {k: float(v) for k, v in (s.split("=", 1) for s in args.world)},
            "memory": args.memory}
    jobs = [(lv, s, d, opts, w) for w in (args.preset or [None]) for lv in args.levels
            for s in range(args.seed0, args.seed0 + args.seeds) for d in deciders]
    t0 = time.monotonic()
    if args.jobs > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.jobs) as pool:
            eps = pool.map(_job, jobs, chunksize=1)
    else:
        eps = [_job(j) for j in jobs]
    for line in summarize(eps):
        print(line)
    from config import load_config

    out = Path(args.out) if args.out else (
        Path(load_config().paths.debug) / "selfplay" / f"{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "episodes": [e.__dict__ for e in eps]},
                              indent=1), encoding="utf-8")
    print(f"-> {out} ({time.monotonic() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
