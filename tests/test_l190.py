"""L190 (SUPER HARD) tunnels / dispensers from the user's screenshots (dataset shots 073-084,
tools.collect --label l190): start board and frames after both row-1 tunnels dropped buckets.
Shots 073 / 076-079 / 081 / 082 / 084 are annotated (every cell checked in test_vision); here
every shot, also the unannotated 074 / 075 / 080 / 083, must read each tunnel's number and
opening, and a real board must drop into the cell in front of the opening."""
import pytest

cv2 = pytest.importorskip("cv2")

from cells import Kind  # noqa: E402
from state import LevelState  # noqa: E402
from tests.test_state import analyzer  # noqa: E402,F401  (fixture)
from tools.build_palette import read_shot  # noqa: E402
from vision import read_gameplay  # noqa: E402

DATASET = __import__("pathlib").Path(__file__).resolve().parents[1] / "dataset"
# (r1c1 opening right, r1c6 opening left); r5c3 (opening up) shows 5 in every shot
ROW1 = {"073": (5, 5), "074": (5, 5), "075": (5, 5), "076": (5, 4), "077": (5, 3),
        "078": (5, 2), "079": (5, 1), "080": (5, 5), "081": (4, 5), "082": (3, 5),
        "083": (2, 5), "084": (2, 5)}


def view(an, seq):  # noqa: F811
    return read_gameplay(read_shot(DATASET, seq), an.palette, an.cfg.vision, an.reader,
                         log=lambda s: None, icons=an.icons, hints=an.hints)


@pytest.mark.parametrize("seq", sorted(ROW1))
def test_tunnels_read_with_number_and_opening(analyzer, seq):  # noqa: F811
    g = view(analyzer, seq).grid
    left, right = ROW1[seq]
    assert g[1][1].label() == f"tunnel{left}:right"
    assert g[1][6].label() == f"tunnel{right}:left"
    assert g[5][3].label() == "tunnel5:up"


def test_taking_the_bucket_in_front_drops_the_next(analyzer):  # noqa: F811
    st = LevelState.from_view(view(analyzer, "076"))       # r1c5 blue, dropped by r1c6 (4)
    tun = next(l for l in st.locks if l.cells == ((1, 6),))
    assert (tun.kind, tun.counter, tun.front()) == ("tunnel", 4, (1, 5))
    assert (1, 5) in st.tappable
    after = st.after_tap((1, 5))
    assert after.cell((1, 5)).kind is Kind.UNKNOWN                    # colour not seen yet
    assert next(l for l in after.locks if l.cells == ((1, 6),)).counter == 3
    assert (1, 5) not in after.tappable
