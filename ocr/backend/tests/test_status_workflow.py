"""Saving never changes a page's workflow status; only /status does.

Regression: after "Submit for review" the editor still held
annotation.status == "in_progress", and the next Save wrote that back,
so finished pages kept showing "in progress".
"""
from __future__ import annotations

from test_per_file_folders import _jpg, _project, _upload, client  # noqa: F401  (fixture)


def _page(client):  # noqa: F811
    pid = _project(client)
    [doc_id] = _upload(client, pid, "receipt.jpg", _jpg())["document_ids"]
    got = client.get(f"/documents/{doc_id}").json()
    got["doc"]["doc_type"] = "receipt"           # submit requires a doc_type
    return doc_id, got


def _status(client, doc_id, target):  # noqa: F811
    return client.post(f"/documents/{doc_id}/status", json={"target": target})


def _save(client, doc_id, doc, rev):  # noqa: F811
    return client.put(f"/documents/{doc_id}/annotation", json={"doc": doc, "revision": rev})


def test_first_save_moves_pre_annotated_to_in_progress(client):  # noqa: F811
    doc_id, got = _page(client)
    r = _save(client, doc_id, got["doc"], got["revision"])
    assert r.status_code == 200, r.text
    after = client.get(f"/documents/{doc_id}").json()
    if got["status"] in ("generated", "pre_annotated"):
        assert after["status"] == "in_progress"
    assert after["doc"]["annotation"]["status"] == after["status"]


def test_stale_save_does_not_revert_submitted_or_approved(client):  # noqa: F811
    doc_id, got = _page(client)
    stale_doc = got["doc"]                      # editor copy, says "in_progress"
    stale_doc["annotation"]["status"] = "in_progress"

    r = _save(client, doc_id, stale_doc, got["revision"])
    assert r.status_code == 200, r.text

    r = _status(client, doc_id, "submitted")
    assert r.status_code == 200, r.text
    rev = r.json()["revision"]

    # save again with the stale status in the payload
    r = _save(client, doc_id, stale_doc, rev)
    assert r.status_code == 200, r.text
    now = client.get(f"/documents/{doc_id}").json()
    assert now["status"] == "submitted"
    assert now["doc"]["annotation"]["status"] == "submitted"

    r = _status(client, doc_id, "approved")
    assert r.status_code == 200, r.text
    r = _save(client, doc_id, stale_doc, r.json()["revision"])
    assert r.status_code == 200, r.text
    assert client.get(f"/documents/{doc_id}").json()["status"] == "approved"


def test_pre_annotated_page_can_be_submitted_directly(client):  # noqa: F811
    doc_id, got = _page(client)
    if got["status"] != "pre_annotated":
        return
    r = _status(client, doc_id, "submitted")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "submitted"


def test_approved_page_can_be_rejected(client):  # noqa: F811
    doc_id, got = _page(client)
    _save(client, doc_id, got["doc"], got["revision"])
    assert _status(client, doc_id, "submitted").status_code == 200
    assert _status(client, doc_id, "approved").status_code == 200
    r = _status(client, doc_id, "rejected")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "rejected"
