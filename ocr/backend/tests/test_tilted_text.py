"""Tilted text: snap-to-ink follows the slant, and region OCR straightens a
tilted quad instead of cutting out its outer box."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from PIL import Image as PILImage

from app.services.geometry import quad_matches_bbox
from app.services.lines import ink_angle, ink_mask, snap_box_to_ink, snap_box_to_ink_tilted
from app.services.quad_crop import crop_quad


def page_with_line(deg: float, W: int = 1400, H: int = 600) -> np.ndarray:
    """A white page with one printed line, rotated by ``deg`` around its centre
    (OpenCV sign: positive = counter-clockwise on screen = rising to the right)."""
    g = np.full((H, W), 255, np.uint8)
    text, font, thick = "THE QUICK BROWN FOX JUMPS 1234", cv2.FONT_HERSHEY_SIMPLEX, 4
    scale = 1.6
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    if tw > 0.75 * W:  # a narrower page: smaller print, still one line in the middle
        scale *= 0.75 * W / tw
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(g, text, (int((W - tw) / 2), int((H + th) / 2)), font, scale, 0, thick, cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1.0)
    return cv2.warpAffine(g, M, (W, H), borderValue=255)


def _ink_inside_quad(gray, quad) -> float:
    """Share of the page's ink pixels that lie inside the quad."""
    ys, xs = np.nonzero(ink_mask(gray))
    poly = np.array(quad, np.float32)
    inside = sum(cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
                 for x, y in zip(xs[::7], ys[::7]))
    return inside / len(xs[::7])


@pytest.mark.parametrize("deg", [4.0, -6.0, 10.0])
def test_ink_angle_measures_the_tilt(deg):
    g = page_with_line(deg)
    ys, xs = np.nonzero(ink_mask(g))
    # image y points down, so the measured angle has the opposite sign
    assert ink_angle(xs, ys) == pytest.approx(-deg, abs=0.3)


@pytest.mark.parametrize("deg", [4.0, -6.0])
def test_loose_box_around_tilted_line_snaps_to_a_tight_tilted_quad(deg):
    g = page_with_line(deg)
    loose = [120, 120, 1300, 500]
    box, quad = snap_box_to_ink_tilted(g, loose)
    assert quad is not None and len(quad) == 4
    # the text is inside the quad ...
    assert _ink_inside_quad(g, quad) > 0.98
    # ... and the quad is much tighter than the straight snapped box
    sb = snap_box_to_ink(g, loose)
    quad_area = cv2.contourArea(np.array(quad, np.float32))
    assert quad_area < 0.6 * (sb[2] - sb[0]) * (sb[3] - sb[1])
    # the saved quad is kept on save (its outer bounds are the bbox)
    assert quad_matches_bbox(quad, box)
    # TL, TR, BR, BL order: top-left is left of top-right and above bottom-left
    tl, tr, br, bl = quad
    assert tl[0] < tr[0] and bl[0] < br[0] and tl[1] < bl[1] and tr[1] < br[1]


def test_straight_text_snaps_exactly_as_before():
    g = page_with_line(0.0)
    loose = [120, 120, 1300, 500]
    box, quad = snap_box_to_ink_tilted(g, loose)
    assert quad is None
    assert box == snap_box_to_ink(g, loose)


def test_tilted_quad_stays_inside_its_parent_region():
    g = page_with_line(6.0)
    parent = [300, 150, 1100, 450]
    _, quad = snap_box_to_ink_tilted(g, [120, 120, 1300, 500], clip_to=parent)
    assert quad is not None
    assert all(parent[0] <= x <= parent[2] and parent[1] <= y <= parent[3] for x, y in quad)


def test_crop_quad_straightens_a_tilted_line():
    deg = 6.0
    g = page_with_line(deg)
    _, quad = snap_box_to_ink_tilted(g, [120, 120, 1300, 500])
    img = PILImage.fromarray(g)
    out = np.asarray(crop_quad(img, quad).convert("L"))
    # one straight strip: as long as the text, about one line high
    tl, tr, _, bl = quad
    assert out.shape[1] == pytest.approx(math.dist(tl, tr), abs=2)
    assert out.shape[0] == pytest.approx(math.dist(tl, bl), abs=2)
    assert out.shape[0] < 90
    # and the text in it is level
    ys, xs = np.nonzero(ink_mask(out))
    assert abs(ink_angle(xs, ys)) < 0.5
    # the old outer-box crop was a tall slanted patch
    xs_, ys_ = [p[0] for p in quad], [p[1] for p in quad]
    assert max(ys_) - min(ys_) > 1.5 * out.shape[0]


def test_crop_quad_of_a_rectangle_is_the_plain_crop():
    g = page_with_line(0.0)
    img = PILImage.fromarray(g)
    q = [[200, 270], [900, 270], [900, 340], [200, 340]]
    assert np.array_equal(np.asarray(crop_quad(img, q)), np.asarray(img.crop((200, 270, 900, 340))))


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402

from conftest import H as PH, W as PW, make_page  # noqa: E402


@pytest.fixture()
def client():
    from app.main import app
    return TestClient(app)


def test_snap_box_endpoint_returns_a_tilted_quad(pagestore, client):
    pagestore(make_page(), page_with_line(5.0, W=PW, H=PH))
    r = client.post("/documents/DOC-0001/snap-box", json={"bbox": [60, 400, 960, 1000]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["changed"] is True
    assert body["quad"] is not None and len(body["quad"]) == 4
    assert quad_matches_bbox(body["quad"], body["bbox"])


def test_snap_box_endpoint_straight_text_has_no_quad(pagestore, client):
    pagestore(make_page(), page_with_line(0.0, W=PW, H=PH))
    r = client.post("/documents/DOC-0001/snap-box", json={"bbox": [60, 600, 960, 800]})
    assert r.status_code == 200, r.text
    assert r.json()["quad"] is None


def test_ocr_region_sends_the_straightened_strip(pagestore, client, monkeypatch):
    g = page_with_line(6.0, W=PW, H=PH)
    pagestore(make_page(), g)
    _, quad = snap_box_to_ink_tilted(g, [60, 400, 960, 1000])
    seen = {}

    from app.pipeline.adapters import mock

    def recognize(self, crops, lang_hint):
        seen["size"] = crops[0].size
        return [("ok", 0.9)]

    monkeypatch.setattr(mock.MockRec, "recognize", recognize)
    r = client.post("/documents/DOC-0001/ocr-region", json={"quad": quad, "family": "mock"})
    assert r.status_code == 200, r.text
    w, h = seen["size"]
    tl, tr, _, bl = quad
    assert w == pytest.approx(math.dist(tl, tr), abs=2)
    assert h == pytest.approx(math.dist(tl, bl), abs=2)
