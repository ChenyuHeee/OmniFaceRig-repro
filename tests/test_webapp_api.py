"""Webapp API tests: /api/health, /api/characters, /api/rigged, /api/rig
validation, and friendly error handlers.

The app is imported from code/webapp/app.py with HOME redirected to a tmp dir
so the server-side ~/work paths never touch the real machine.
"""

import importlib.util
import io
import os

import pytest


@pytest.fixture()
def app(tmp_path, monkeypatch):
    work = tmp_path / "work"
    (work / "glbs").mkdir(parents=True)
    (work / "outputs").mkdir(parents=True)
    # fake base characters
    for i in range(3):
        (work / "glbs" / f"ai3d_{i+1:02d}.glb").write_bytes(b"glb")
    # fake prebuilt rigged products
    (work / "outputs" / "ai3d_01_rigged.glb").write_bytes(b"x" * 100)
    (work / "outputs" / "ai3d_02_rigged.glb").write_bytes(b"y" * 50)
    monkeypatch.setenv("HOME", str(tmp_path))

    spec = importlib.util.spec_from_file_location(
        "webapp_app", os.path.join(os.path.dirname(__file__), "..", "code", "webapp", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # wait for any background threads started by earlier tests
    mod.app.config["TESTING"] = True
    yield mod
    mod.JOBS.clear()


@pytest.fixture()
def client(app):
    return app.app.test_client()


def test_health_ok(client, app):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["checks"]["process"] == "ok"
    assert body["checks"]["outdir"]["writable"] is True
    assert body["checks"]["glbs"]["count"] == 3
    assert body["checks"]["rigged"]["count"] == 2
    assert body["uptime_s"] >= 0


def test_health_degraded_when_outdir_missing(app):
    import shutil
    shutil.rmtree(app.OUTDIR)  # remove outputs dir
    client = app.app.test_client()
    r = client.get("/api/health")
    assert r.status_code == 503
    assert r.get_json()["status"] == "degraded"
    assert r.get_json()["checks"]["outdir"]["exists"] is False


def test_characters_backward_compat(client, app):
    r = client.get("/api/characters")
    assert r.status_code == 200
    assert r.get_json() == ["ai3d_01.glb", "ai3d_02.glb", "ai3d_03.glb"]


def test_characters_empty_when_glbs_missing(app):
    import shutil
    shutil.rmtree(app.GLBDIR)
    client = app.app.test_client()
    assert client.get("/api/characters").get_json() == []


def test_rigged_lists_prebuilt(client, app):
    r = client.get("/api/rigged")
    assert r.status_code == 200
    items = r.get_json()
    assert [i["name"] for i in items] == ["ai3d_01_rigged.glb", "ai3d_02_rigged.glb"]
    assert items[0]["base"] == "ai3d_01.glb"
    assert items[0]["url"] == "/output/ai3d_01_rigged.glb"
    assert items[0]["size"] == 100
    assert "mtime" in items[0]


def test_rig_validation_bad_character(client):
    r = client.post("/api/rig", data={"character": "../../etc/passwd", "text": "hi"})
    assert r.status_code == 400
    assert "角色名不合法" in r.get_json()["error"]


def test_rig_validation_long_text(client):
    r = client.post("/api/rig", data={"character": "ai3d_01.glb", "text": "x" * 2001})
    assert r.status_code == 400
    assert "口型文本过长" in r.get_json()["error"]


def test_rig_validation_bad_image_type(client):
    data = {"character": "ai3d_01.glb", "text": "hi",
            "image": (io.BytesIO(b"\x00\x01\x02\x03not-an-image"), "evil.txt")}
    r = client.post("/api/rig", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    assert "图片格式不支持" in r.get_json()["error"]


def test_rig_starts_job(app, client):
    called = {}

    def fake_run(job_id, payload):
        called["job"] = job_id
        app.JOBS[job_id]["status"] = "done"
        app.JOBS[job_id]["result"] = {"url": f"/output/{job_id}.glb", "size": 1}

    app._run_job = fake_run
    r = client.post("/api/rig", data={"character": "ai3d_01.glb", "text": "你好"})
    assert r.status_code == 200
    job = r.get_json()["job"]
    import time
    for _ in range(50):
        if app.JOBS[job]["status"] == "done":
            break
        time.sleep(0.02)
    assert app.JOBS[job]["status"] == "done"
    assert app.JOBS[job]["result"]["url"].endswith(".glb")


def test_status_unknown_job_404(client):
    assert client.get("/api/status/nope").status_code == 404
    body = client.get("/api/status/nope").get_json()
    assert body["error"] == "资源不存在"


def test_api_404_json(client):
    r = client.get("/api/nonexistent")
    assert r.status_code == 404
    assert r.get_json()["error"] == "资源不存在"


def test_page_404_html(client):
    r = client.get("/nonexistent-page")
    assert r.status_code == 404
    assert "页面不存在" in r.get_data(as_text=True)


def test_index_contains_rigged_section(client):
    html = client.get("/").get_data(as_text=True)
    assert "已有产物" in html
    assert "loadRigged" in html
    assert "validateImage" in html
    assert "MAX_MB" in html


def test_oversize_returns_413_json(client, app):
    app.app.config["MAX_CONTENT_LENGTH"] = 64  # tiny cap for the test
    data = {"character": "ai3d_01.glb", "text": "hi",
            "image": (io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 256), "big.png")}
    r = client.post("/api/rig", data=data, content_type="multipart/form-data")
    assert r.status_code == 413
    assert "32MB" in r.get_json()["error"]


def test_health_endpoint_and_root_serve(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/health").status_code == 200
