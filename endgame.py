"""Belt-stuck / endgame rules for strategy.py: last-slot gamble, sacrifice, endgame reserve (pure logic on state.LevelState; user rules
2026-10-20, docs/GAME_RULES.md "Last-slot gamble" and "Endgame")."""
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


def last_slot_gamble(state: LevelState, cands: List["Candidate"], sc: StrategyConfig
                     ) -> Optional["Candidate"]:
    """User rule (2026-10-20): the belt is stuck (no belt bucket gained sand for
    ``gamble_stuck_s``), one slot is left and every remaining move breaks the last-slot exposure
    minimum. Waiting changes nothing, the level is lost otherwise: take the last slot with the
    move that uncovers the most sand for the belt's colours, ties by its own exposed sand. Only
    moves rejected for that minimum count; one short of sand is allowed too (same reasoning:
    the level is lost otherwise), a colour stalled on the belt never (it uncovers nothing)."""
    from strategy import _score, evaluate

    stuck = belt_stuck(state, sc)
    if state.free_slots != 1 or stuck is None:
        return None
    pool = []
    for c in (c for c in cands if c.low_exposure or c.reserved):
        g = evaluate(state, c.pos, sc, gamble=True)
        if g.rejected is None and g.color not in state.belt_stalled:
            g.gamble, g.uncovers = True, uncovers_for_belt(state, g.color)
            pool.append(g)
    if not pool:
        return None
    best = max(pool, key=lambda c: (round(c.uncovers, 6), round(c.exposed, 6),
                                    -c.pos[0], -c.pos[1]))
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
