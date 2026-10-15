"""Uploads are kept in <storage_root>/manual_annotation/<name>/ as
<name>_page<N>.png + <name>_page<N>.jsonl, and reopened from there."""
from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw


def _pdf(pages: int, mark: str = "") -> bytes:
    ims = []
    for i in range(pages):
        im = Image.new("RGB", (850, 1100), "white")
        ImageDraw.Draw(im).text((100, 100), f"page {i + 1} {mark}", fill="black")
        ims.append(im)
    buf = io.BytesIO()
    ims[0].save(buf, format="PDF", save_all=True, append_images=ims[1:])
    return buf.getvalue()


def _jpg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (900, 600), "white").save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DEVALABEL_STORAGE_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    from app.config import get_settings
    get_settings.cache_clear()
    import app.storage as storage_mod
    for name in ("_storage", "_STORAGE", "_instance"):
        if hasattr(storage_mod, name):
            monkeypatch.setattr(storage_mod, name, None)
    if hasattr(storage_mod.get_storage, "cache_clear"):
        storage_mod.get_storage.cache_clear()
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    c.root = tmp_path
    yield c
    get_settings.cache_clear()
    if hasattr(storage_mod.get_storage, "cache_clear"):
        storage_mod.get_storage.cache_clear()


def _project(c) -> int:
    r = c.post("/projects", json={"name": "p"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _upload(c, pid, name, data):
    r = c.post(f"/projects/{pid}/images", files=[("files", (name, data))])
    assert r.status_code == 200, r.text
    return r.json()["results"][0]


def _page(c, name, n, ext="png"):
    d = c.root / "manual_annotation" / name
    return d / f"{name}_page{n}.{ext}", d / f"{name}_page{n}.jsonl"


def test_pdf_pages_saved_as_named_png_and_jsonl(client):
    pid = _project(client)
    res = _upload(client, pid, "pdf_001.pdf", _pdf(2))
    assert res["ok"] and len(res["document_ids"]) == 2 and not res["skipped_duplicate"]
    for n in (1, 2):
        png, jl = _page(client, "pdf_001", n)
        assert png.is_file() and jl.is_file()
        assert Image.open(png).format == "PNG"
        lines = jl.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        doc = json.loads(lines[0])
        assert doc["image"] == f"manual_annotation/pdf_001/pdf_001_page{n}.png"
        assert doc["page_number"] == n and doc["page_count"] == 2
    # the old store is not used for these pages
    assert not list((client.root / "_db" / "documents").glob("*[0-9].json"))


def test_edits_saved_to_jsonl_and_reopened(client):
    pid = _project(client)
    data = _pdf(2)
    ids = _upload(client, pid, "pdf_001.pdf", data)["document_ids"]

    got = client.get(f"/documents/{ids[0]}").json()
    doc, rev = got["doc"], got["revision"]
    doc["doc_type"] = "government_notice"
    r = client.put(f"/documents/{ids[0]}/annotation", json={"doc": doc, "revision": rev})
    assert r.status_code == 200, r.text
    _, jl = _page(client, "pdf_001", 1)
    assert json.loads(jl.read_text())["doc_type"] == "government_notice"

    again = _upload(client, pid, "pdf_001.pdf", data)
    assert again["skipped_duplicate"] and again["document_ids"] == ids
    assert again["warning"] is None
    assert client.get(f"/documents/{ids[0]}").json()["doc"]["doc_type"] == "government_notice"


def test_hand_edit_of_jsonl_is_loaded(client):
    pid = _project(client)
    ids = _upload(client, pid, "pdf_001.pdf", _pdf(1))["document_ids"]
    _, jl = _page(client, "pdf_001", 1)
    d = json.loads(jl.read_text())
    d["doc_type"] = "hand_edited"
    import os, time
    time.sleep(1.1)
    jl.write_text(json.dumps(d, ensure_ascii=False, indent=2))   # pretty-printed on purpose
    got = client.get(f"/documents/{ids[0]}").json()
    assert got["doc"]["doc_type"] == "hand_edited"
    assert len(jl.read_text().strip().splitlines()) == 1        # written back as one line


def test_same_name_different_content_opens_saved(client):
    pid = _project(client)
    ids = _upload(client, pid, "pdf_001.pdf", _pdf(2, "v1"))["document_ids"]
    res = _upload(client, pid, "pdf_001.pdf", _pdf(2, "v2 rescanned"))
    assert res["ok"] and res["skipped_duplicate"] and res["document_ids"] == ids
    assert "different content" in res["warning"]
    assert len(list((client.root / "manual_annotation" / "pdf_001").glob("*.png"))) == 2


def test_folder_adopted_after_index_reset(client):
    pid = _project(client)
    data = _pdf(2)
    ids = _upload(client, pid, "pdf_001.pdf", data)["document_ids"]
    got = client.get(f"/documents/{ids[1]}").json()
    doc = got["doc"]
    doc["doc_type"] = "kept_after_reset"
    client.put(f"/documents/{ids[1]}/annotation", json={"doc": doc, "revision": got["revision"]})

    shutil.rmtree(client.root / "_db" / "documents")          # tool index lost
    res = _upload(client, pid, "pdf_001.pdf", data)
    assert res["ok"] and res["document_ids"] == ids
    assert client.get(f"/documents/{ids[1]}").json()["doc"]["doc_type"] == "kept_after_reset"


def test_single_image_same_scheme(client):
    pid = _project(client)
    res = _upload(client, pid, "scans/scan_12.jpg", _jpg())
    assert res["ok"]
    img, jl = _page(client, "scan_12", 1, "jpg")
    assert img.is_file() and jl.is_file()
    again = _upload(client, pid, "scan_12.jpg", _jpg())
    assert again["skipped_duplicate"] and again["document_ids"] == res["document_ids"]


def test_build_labels_and_export_paths(client, monkeypatch):
    pid = _project(client)
    _upload(client, pid, "pdf_001.pdf", _pdf(2))
    _upload(client, pid, "scan_12.jpg", _jpg())

    import sys
    from scripts import build_labels
    monkeypatch.setattr(sys, "argv", ["build_labels"])
    assert build_labels.main() == 0
    lab = client.root / "manual_annotation" / "labels.jsonl"
    recs = [json.loads(l) for l in lab.read_text().splitlines()]
    assert len(recs) == 3
    for r in recs:
        assert (lab.parent / r["image"]).is_file(), r["image"]

    z = client.get(f"/projects/{pid}/export?format=zip")
    zf = zipfile.ZipFile(io.BytesIO(z.content))
    names = set(zf.namelist())
    recs = [json.loads(l) for l in zf.read("labels.jsonl").decode().splitlines()]
    assert len(recs) == 3 and all(r["image"] in names for r in recs)
