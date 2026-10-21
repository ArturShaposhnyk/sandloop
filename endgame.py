"""Belt-stuck / endgame rules for strategy.py: last-slot gamble, sacrifice, endgame / key / ice
reserves, lost position (pure logic on state.LevelState; user rules 2026-10-20 / -21,
docs/GAME_RULES.md "Last-slot gamble", "Endgame" and "Ice")."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from config import StrategyConfig
from state import LevelState

if TYPE_CHECKING:
    from strategy import Candidate


def endgame_reserve(state: LevelState, colour: str, sc: StrategyConfig) -> List[str]:
    """User (2026-10-20, run 20261020-145310): when only ``endgame_colours`` colours are left on
    the board, keep a free slot for each other colour whose sand is still in the picture (at
    least ``endgame_sand_min`` of a bucket), that has buckets on the board and none on the belt,
    instead of adding a bucket of a colour that is almost done (on the belt already, or less
    than a bucket of sand left). Only colours more exposed than ``colour`` are kept (else two
    almost-done colours would each wait for the other: run 145310 #26, purple 27 % vs darkred
    39 % of the contact layer -> purple waits, darkred goes). Returns those colours (empty = no
    reserve for ``colour``)."""
    board = state.grid_buckets()
    if not board or len(board) > sc.endgame_colours:
        return []
    cap = state.capacity_for(colour)
    almost_done = state.on_belt(colour) or (cap is not None
                                            and state.sand.get(colour, 0.0) < cap)
    if not almost_done:
        return []
    keep = []
    for d in sorted(board):
        cap_d = state.capacity_for(d)
        if (d != colour and not state.on_belt(d) and cap_d is not None
                and state.sand.get(d, 0.0) >= sc.endgame_sand_min * cap_d
                and state.exposed.get(d, 0.0) > state.exposed.get(colour, 0.0)):
            keep.append(d)
    return keep


def belt_stuck(state: LevelState, sc: StrategyConfig) -> Optional[float]:
    """Seconds every unfinished belt bucket has not gained sand, if that is at least
    ``gamble_stuck_s`` (else None). Buckets the reader cannot see right now keep their colour's
    stall clock (BeltTracker forgets a colour only after ``stall_forget_s``)."""
    colours = {b.color for b in state.conveyor if b.color and (b.fill or 0) < 100}
    stalls = [state.belt_stalled.get(c) for c in colours | set(state.belt_stalled)]
    if not stalls or any(s is None or s < sc.gamble_stuck_s for s in stalls):
        return None
    return min(stalls)


def uncovers_for_belt(state: LevelState, colour: str) -> float:
    """Contact share of the belt's colours that collecting ``colour`` uncovers (the sand right
    above it, sand.contact_layers)."""
    belt = {b.color for b in state.conveyor if b.color and (b.fill or 0) < 100}
    belt |= {s for b in state.conveyor for s in b.stages}
    return sum(share for d, share in state.above.get(colour, {}).items()
               if d in belt and d != colour)


def can_fill(state: LevelState, colour: str, sc: StrategyConfig) -> bool:
    """User (run 20261020-224631): a bucket of ``colour`` can fill when its colour's sand left
    covers one more bucket plus what same-colour belt buckets still need
    (``gamble_fill_margin`` x capacity; colours without a capacity count as able)."""
    cap = state.capacity_for(colour)
    if not cap:
        return True
    need = (state.conveyor_need(colour) + 1) * cap * sc.gamble_fill_margin
    return state.sand.get(colour, 0.0) >= need


def gamble_fill(state: LevelState, pos, colour: str, sc: StrategyConfig) -> Optional[str]:
    """Why the bucket at ``pos`` may take the last slot: it can fill itself (a tower: its top
    colour), or the tap opens the path to a bucket that can - else None."""
    if can_fill(state, colour, sc):
        return "fills itself"
    after = state.after_tap(pos)
    for p in state.opened_by(pos)[0]:
        c = after.cell(p).color
        if c and can_fill(after, c, sc):
            return f"opens r{p[0]}c{p[1]} {c}"
    return None


def last_slot_gamble(state: LevelState, cands: List["Candidate"], sc: StrategyConfig
                     ) -> Optional["Candidate"]:
    """User rule (2026-10-20): the belt is stuck (no belt bucket gained sand for
    ``gamble_stuck_s``), one slot is left and every remaining move breaks the last-slot exposure
    minimum (or the sand rule). Waiting changes nothing, the level is lost otherwise: take the
    last slot. User (run 20261020-224631, purple at 1.95 % sand < a 3.67 % bucket took the slot
    from yellow, which would have filled): never a bucket that cannot fill (``can_fill``) unless
    the tap opens the path to one that can; buckets that fill themselves rank first, then the
    sand uncovered for the belt's colours, then their own exposed sand. A colour stalled on the
    belt never (it uncovers nothing)."""
    from strategy import _score, evaluate

    stuck = belt_stuck(state, sc)
    if state.free_slots != 1 or stuck is None:
        return None
    pool = []
    # a lookahead "jam" on a stuck belt with one slot is this very situation (run 224631 #27:
    # yellow, a key colour, passed the rules but the lookahead called the belt jammed)
    for c in (c for c in cands if c.low_exposure or c.reserved or c.sand_short or c.jam):
        g = evaluate(state, c.pos, sc, gamble=True)
        if g.rejected is not None or g.color in state.belt_stalled:
            continue
        g.fills = gamble_fill(state, c.pos, g.color, sc)
        if g.fills is None:
            continue
        g.gamble, g.uncovers = True, uncovers_for_belt(state, g.color)
        pool.append(g)
    if not pool:
        return None
    best = max(pool, key=lambda c: (c.fills == "fills itself", round(c.uncovers, 6),
                                    round(c.exposed, 6), -c.pos[0], -c.pos[1]))
    _score(state, best, sc)
    return best


def sacrifice(state: LevelState, cand: "Candidate", now, check, sc: StrategyConfig
              ) -> Optional["Candidate"]:
    """User (2026-10-20): a bucket that cannot fill now (sand short) may take a slot if the
    lookahead shows it opens the way to what unblocks the belt - a continuation of 2+ taps after
    which a belt colour that gets no sand now gets some - and a slot stays free after it. More
    of its colour may appear once the covering layer is collected, so it may still fill later."""
    from lookahead import dead_colours, unblocking
    from strategy import _score, evaluate

    if (state.free_slots or 0) < 2:
        return None
    dead = dead_colours(state, sc)
    if not dead:
        return None
    o = unblocking(state, cand.pos, check, sc, dead)
    if o is None:
        return None
    c = evaluate(state, cand.pos, sc, gamble=True)
    if c.rejected:
        return None
    _score(state, c, sc)
    c.sacrifice, c.outlook = True, o
    c.look_bonus = sc.weight_lookahead * (o.value - now.value)
    c.score += c.look_bonus
    return c


def _keys(state: LevelState, sc: StrategyConfig):
    """(key colours, buried belt colours) - see ``key_reserve``."""
    belt = {b.color for b in state.conveyor if b.color and (b.fill or 0) < 100}
    belt |= {s for b in state.conveyor for s in b.stages}
    buried = {d for d in belt if state.exposed.get(d, 0.0) < sc.look_min_exposed}
    if not buried:
        return [], buried
    keys = [k for k in state.grid_buckets() if not state.on_belt(k)
            and state.exposed.get(k, 0.0) >= sc.look_min_exposed
            and (state.exposed.get(k, 0.0) >= sc.key_min_exposed
                 or any(d in buried and d != k and share >= sc.look_min_gain
                        for d, share in state.above.get(k, {}).items()))]
    return keys, buried


def key_colours(state: LevelState, sc: StrategyConfig) -> List[str]:
    """Board colours the key reserve keeps slots for. Run 20261020-224631 #27: the reserve kept
    the last slot for yellow (22.9 % of the contact layer), then the last-slot minimum (25 %)
    rejected yellow itself - so a key colour's own bucket takes the last slot at
    ``key_min_exposed`` (``strategy.min_exposure``)."""
    return _keys(state, sc)[0]


def key_reserve(state: LevelState, pos, sc: StrategyConfig) -> Optional[str]:
    """User (2026-10-20, run 20261020-171133): a "key" colour is one whose contact sand covers
    (``look_min_gain`` of the contact layer at least, ``state.above``) the sand of belt buckets
    that collect nothing now (their colour < ``look_min_exposed`` of it), or that fills
    ``key_min_exposed`` of the contact layer while such buckets wait (L187: darkred 54 % / cream 40 % of the
    contact layer over yellow / orange / blue / darkgreen at 96-99 %). Reaching its nearest
    bucket takes 1 + the buckets in front of it on the path (darkred r3c2 behind yellow r2c2 =
    2 slots). A tap at ``pos`` that is neither a key colour's bucket nor on the path to the
    nearest one must leave that many slots free - only while they are free now (a key out of
    reach anyway reserves nothing: L186 run 162703, 1 free, magenta behind pink). Returns why
    it is rejected, or None."""
    from cells import Kind
    from planner import shortest_path

    free = state.free_slots
    if not free:
        return None
    keys, buried = _keys(state, sc)
    cell = state.cell(pos)
    if not keys or (cell is not None and cell.color in keys):
        return None                     # a key colour's bucket itself is always allowed
    costs = {(c.row, c.col): (1.0 if c.kind is Kind.BUCKET else 0.0)
             for c in state.cells() if c.kind in (Kind.BUCKET, Kind.EMPTY)}
    best = None                         # (slots needed, key colour, target, path)
    for c in state.cells(Kind.BUCKET):
        if c.color not in keys:
            continue
        found = shortest_path(state, (c.row, c.col), costs)
        if found is None:
            continue
        steps = found[1]
        if pos == (c.row, c.col) or pos in steps:
            return None                 # this tap is on the way to a key colour
        need = len(steps) + 1
        if need <= free and (best is None or need < best[0]):   # reachable with today's slots
            best = (need, c.color, (c.row, c.col), steps)
    if best is None or free - 1 >= best[0]:
        return None
    need, colour, (r, c), steps = best
    covers = sorted(d for d in state.above.get(colour, {}) if d in buried and d != colour)
    path = ", ".join(f"r{p[0]}c{p[1]} {state.cell(p).color}" for p in steps)
    return (f"keys: keep {need} slot(s) to reach {colour} r{r}c{c} (its sand covers "
            f"{', '.join(covers)} on the belt" + (f"; in front: {path}" if path else "") + ")")


def ice_reserve(state: LevelState, pos, sc: StrategyConfig) -> Optional[str]:
    """User (L189 run 20261021-092045): ice with N = N more taps before the bucket under it can
    be tapped, and every tap takes a belt slot. The bot tapped red / darkred buckets that
    collected nothing (3-6 % of the contact layer) while the ice counted down; at ice 1 the belt
    was 4/5 with dead buckets: the tap that breaks the ice fills the belt and its cyan bucket has
    no slot. Keep max(N, b) + 1 slots (b = buckets in front of the ice on its cheapest path:
    those taps count the ice down too) - only while that many slots exist. A tap that would
    leave fewer is allowed only for a bucket that fills and leaves the belt again (exposed >=
    ``ice_live_exposed``, enough sand, not stalled). Returns why it is rejected, or None."""
    from cells import Kind
    from planner import shortest_path

    free = state.free_slots
    if not free:
        return None
    ices = [l for l in state.locks if l.kind == "ice" and l.counter]
    if not ices:
        return None
    cell = state.cell(pos)
    colour = cell.color if cell is not None else None
    exposed = state.exposed.get(colour or "", 0.0)
    stalled = state.belt_stalled.get(colour or "", 0.0) >= sc.stall_s
    if (colour is not None and exposed >= sc.ice_live_exposed and not stalled
            and can_fill(state, colour, sc)):
        return None                     # it fills and leaves: the slot comes back
    costs = {(c.row, c.col): (1.0 if c.kind is Kind.BUCKET else 0.0)
             for c in state.cells() if c.kind in (Kind.BUCKET, Kind.EMPTY)}
    for ice in sorted(ices, key=lambda l: l.counter):
        found = min((f for f in (shortest_path(state, p, costs) for p in ice.cells) if f),
                    key=lambda f: f[0], default=None)
        if found is None:
            continue                    # out of reach: its bucket is far away anyway
        need = max(ice.counter, len(found[1])) + 1
        if need > state.conveyor_slots or free >= need:
            continue
        r, c = ice.cells[0]
        return (f"ice r{r}c{c} ({ice.counter}): keep {need} slot(s) - {need - 1} tap(s) to break "
                f"it + 1 for the bucket under it ({free} free; {colour} {exposed:.0%} exposed "
                f"would stay on the belt)")
    return None


def lost_finish(state: LevelState, cands: List["Candidate"], sc: StrategyConfig
                ) -> Optional["Candidate"]:
    """User (L189 run 20261021-092045: 200 s on a lost board, belt 4/5 of dead buckets, ice 1):
    the belt has been stuck for ``lost_stuck_s``, no move passes the rules and the last-slot
    gamble finds none - waiting changes nothing. Tap the best remaining bucket (the gamble's
    ranking: one that can fill first, then sand uncovered for the belt, then its own exposure)
    to end the level quickly; the loss goes to levels.jsonl (the level memory bans the
    sequence) and the flow goes to Try Again."""
    from strategy import _score, evaluate

    stuck = belt_stuck(state, sc)
    if not state.free_slots or stuck is None or stuck < sc.lost_stuck_s:
        return None
    pool = []
    for c in cands:
        g = evaluate(state, c.pos, sc, gamble=True)
        if g.shell_wait:
            continue
        g.rejected = None
        g.fills = gamble_fill(state, c.pos, g.color, sc)
        g.lost, g.uncovers = True, uncovers_for_belt(state, g.color)
        pool.append(g)
    if not pool:
        return None
    best = max(pool, key=lambda c: (c.fills is not None, round(c.uncovers, 6),
                                    round(c.exposed, 6), -c.pos[0], -c.pos[1]))
    _score(state, best, sc)
    return best
