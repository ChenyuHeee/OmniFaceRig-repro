"""Web preview service (deliverable 1): upload image -> rigged GLB preview.

Flask app on port 8000 (externally mapped 32170->8000 / 32171->8001 per the
A100 server doc). Endpoints:
  GET  /                    UI (three.js viewer)
  GET  /api/health          health check (process + output dir writable)
  GET  /api/characters      list of available base characters (backward compat)
  GET  /api/rigged          list of prebuilt *_rigged.glb products
  POST /api/rig             start a rig job: {image?, character?, text?}
  GET  /api/status/<job>    job progress
  GET  /output/<file>       serve generated glb
"""

import os
import subprocess
import threading
import time
import uuid

from flask import (Flask, abort, jsonify, render_template_string, request,
                   send_from_directory)

WORK = os.path.expanduser("~/work")
CODE = os.path.join(WORK, "code")
OUTDIR = os.path.join(WORK, "outputs")
GLBDIR = os.path.join(WORK, "glbs")
CONDA_BIN = os.path.expanduser("~/anaconda3/envs/torch2.4_cuda12.1/bin")
RIGGED_SUFFIX = "_rigged.glb"
MAX_UPLOAD_MB = 32

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
JOBS = {}
LOCK = threading.Lock()
START_TS = time.time()


def _list_glbs(directory):
    """Sorted .glb filenames under a directory (missing dir -> [])."""
    try:
        return sorted(f for f in os.listdir(directory) if f.endswith(".glb"))
    except OSError:
        return []


def _characters():
    """Base (T-pose) character glbs — unchanged shape for backward compat."""
    return _list_glbs(GLBDIR)


def _rigged():
    """Prebuilt rigged products (*_rigged.glb / *_im.glb / *_talk_*.glb)."""
    out = []
    for f in _list_glbs(OUTDIR):
        if not (f.endswith(RIGGED_SUFFIX) or "_im.glb" in f or "_talk" in f):
            continue
        p = os.path.join(OUTDIR, f)
        st = os.stat(p)
        base = f[: -len(RIGGED_SUFFIX)] + ".glb"
        out.append({
            "name": f,
            "url": f"/output/{f}",
            "base": base,
            "size": st.st_size,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        })
    return out


def _sniff_image(header):
    """Magic-byte sniff for common image formats; None if unrecognized."""
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"BM"):
        return "image/bmp"
    return None


def _run_job(job_id, payload):
    try:
        JOBS[job_id]["status"] = "running"
        img = payload.get("image")
        audio = payload.get("audio")
        character = payload.get("character", "ai3d_01.glb")
        text = payload.get("text", "你好世界")
        lang = payload.get("lang", "zh" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en")
        glb = os.path.join(GLBDIR, character)
        if not os.path.exists(glb):
            raise RuntimeError(f"character not found: {character}")
        out_name = f"{job_id}.glb"
        out_path = os.path.join(OUTDIR, out_name)
        # Unchanged pipeline invocation (backward compatible): same args as
        # before, only the subprocess PATH is pinned to the conda env so the
        # job works identically under systemd (code/scripts/stage1_real.py is
        # not modified).
        env = dict(os.environ)
        env["PATH"] = CONDA_BIN + os.pathsep + env.get("PATH", "")
        # TRELLIS runtime env (image-to-mesh mode)
        models_dir = os.path.join(WORK, "models")
        env.setdefault("TRELLIS_MODEL_PATH", os.path.join(models_dir, "TRELLIS-image-large"))
        dino = os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vitl14_reg_pretrain.pth")
        env.setdefault("TRELLIS_DINOV2_PTH", dino)
        env.setdefault("ATTN_BACKEND", "sdpa")
        trellis_src = os.path.join(WORK, "src", "TRELLIS-main")
        if os.path.exists(trellis_src):
            env["PYTHONPATH"] = trellis_src + os.pathsep + env.get("PYTHONPATH", "")
        log = []
        cmd = ["python", "-u", "scripts/stage1_real.py",
               "--glb", glb, "--out", out_path, "--text", text]
        if img:
            img_path = os.path.join(OUTDIR, f"{job_id}.img")
            with open(img_path, "wb") as fh:
                fh.write(img)
            if payload.get("image_to_mesh"):
                # FULL deliverable-2 loop: image -> TRELLIS mesh -> rigged glb
                mesh_path = os.path.join(OUTDIR, f"{job_id}_mesh.glb")
                tcmd = ["python", "-u", "scripts/trellis_front.py",
                        "--image", img_path, "--out", mesh_path]
                tproc = subprocess.Popen(
                    tcmd, cwd=CODE,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
                for line in tproc.stdout:
                    log.append(line.rstrip())
                    JOBS[job_id]["log"] = log[-8:]
                trc = tproc.wait()
                if trc != 0:
                    raise RuntimeError("TRELLIS image-to-mesh failed:\n"
                                       + "\n".join(log[-12:]))
                log.append("image-to-mesh OK: " + mesh_path)
                cmd = ["python", "-u", "scripts/stage1_real.py",
                       "--glb", mesh_path, "--out", out_path, "--text", text]
            else:
                cmd += ["--img", img_path]
        proc = subprocess.Popen(
            cmd, cwd=CODE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True,
        )
        for line in proc.stdout:
            log.append(line.rstrip())
            JOBS[job_id]["log"] = log[-8:]
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError("\n".join(log[-15:]))
        # REAL audio lip-sync: audio+transcript -> whisper alignment, or
        # text only -> piper TTS phoneme timestamps; append WEIGHTS animation
        if audio is not None or text:
            audio_path = None
            if audio is not None:
                audio_path = os.path.join(OUTDIR, f"{job_id}.wav")
                with open(audio_path, "wb") as fh:
                    fh.write(audio)
            acmd = ["python", "-u", "scripts/animate_audio.py",
                    "--glb", out_path, "--out", out_path,
                    "--text", text, "--lang", lang]
            if audio_path:
                acmd += ["--audio", audio_path]
            aproc = subprocess.Popen(
                acmd, cwd=CODE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True,
            )
            alog = []
            for line in aproc.stdout:
                alog.append(line.rstrip())
                JOBS[job_id]["log"] = alog[-4:]
            arc = aproc.wait()
            if arc != 0:
                raise RuntimeError("audio lip-sync failed:\n" + "\n".join(alog[-15:]))
            log.append("audio: " + (alog[-1] if alog else ""))
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {"url": f"/output/{out_name}", "size": os.path.getsize(out_path)}
    except Exception as exc:  # noqa
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(exc)


def _api_error(message, code=400):
    return jsonify({"error": message}), code


def _friendly_page(title, message):
    return render_template_string(
        """<!doctype html><html><head><meta charset="utf-8">
<title>{{ title }} · OmniFaceRig-repro</title>
<style>
 body{font-family:system-ui,sans-serif;margin:48px;background:#101418;color:#e8eef4}
 .card{background:#1a222b;border:1px solid #2c3a49;border-radius:10px;padding:24px;max-width:640px}
 a{color:#6fc3ff}
</style></head><body>
<div class="card"><h2>{{ title }}</h2><p>{{ message }}</p>
<p><a href="/">← 返回首页</a></p></div>
</body></html>""", title=title, message=message)


@app.get("/")
def index():
    html = """<!doctype html><html><head><meta charset="utf-8">
<title>OmniFaceRig-repro · A100 preview</title>
<style>
 body{font-family:system-ui,sans-serif;margin:24px;background:#101418;color:#e8eef4}
 .card{background:#1a222b;border:1px solid #2c3a49;border-radius:10px;padding:18px;margin-bottom:16px}
 label{display:block;margin:8px 0 4px;font-size:13px;color:#9fb3c8}
 input,select,button{font-size:14px;padding:7px 10px;border-radius:6px;border:1px solid #33506e;background:#0f151c;color:#e8eef4}
 button{background:#2f6f4f;cursor:pointer;margin-top:12px}
 button:hover{background:#3a8a61}
 .btn2{background:#2f5f8f}.btn2:hover{background:#3a76b3}
 #status{font-family:monospace;font-size:12px;color:#9fb3c8;white-space:pre-wrap;min-height:60px}
 #status.err{color:#ff9d9d}
 #viewer{width:100%;height:560px;border-radius:8px;background:radial-gradient(#1c2631,#0b0f14)}
 .row{display:flex;gap:10px;align-items:end;flex-wrap:wrap}
 .hint{font-size:12px;color:#7d93a8;margin-top:6px}
</style></head><body>
<h2>OmniFaceRig-repro · 图 → 带骨骼+表情的 glb (A100)</h2>
<div class="card">
 <label>已有产物(预 rig 角色,直接预览,无需重新生成)</label>
 <div class="row">
  <select id="rigged" style="min-width:260px"></select>
  <button class="btn2" onclick="loadRigged()">加载预览</button>
 </div>
 <div class="hint" id="riggedHint">加载中…</div>
</div>
<div class="card">
 <div class="row">
  <div><label>上传 2D 角色图(可选,PNG/JPG/WebP/GIF,≤32MB)</label><input type="file" id="img" accept="image/*"></div>
  <div><label>基础角色(T-pose glb)</label><select id="char"></select></div>
  <div><label>口型文本</label><input id="text" value="你好世界" style="width:220px"></div>
 </div>
 <button onclick="start()">生成 rigged glb</button>
 <div id="status">ready</div>
</div>
<div class="card"><div id="viewer"></div>
 <div style="margin-top:8px">
  <label>表情权重预览(拖动 jawOpen / eyeBlinkLeft / mouthSmileRight)</label>
  <div class="row">
   <input type="range" id="w_jawOpen" min="0" max="1" step="0.01" value="0" oninput="setMorph('jawOpen',this.value)">
   <input type="range" id="w_eyeBlinkLeft" min="0" max="1" step="0.01" value="0" oninput="setMorph('eyeBlinkLeft',this.value)">
   <input type="range" id="w_mouthSmileRight" min="0" max="1" step="0.01" value="0" oninput="setMorph('mouthSmileRight',this.value)">
   <label style="margin:0"><input type="checkbox" id="play" onchange="toggleAnim(this.checked)"> 播放口型动画</label>
  </div>
 </div>
</div>
<script type="importmap">{"imports":{"three":"/static/three.module.js","three/addons/":"/static/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
let renderer, scene, camera, controls, model, mixer, clock = new THREE.Clock(), morphNames = [], animClips = [];
const MAX_MB = 32, ALLOWED = ['image/png','image/jpeg','image/webp','image/gif','image/bmp'];
function init(){
  const el = document.getElementById('viewer');
  renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(el.clientWidth, el.clientHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  el.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.2));
  scene.add(new THREE.DirectionalLight(0xffffff, 1.5));
  camera = new THREE.PerspectiveCamera(40, el.clientWidth/el.clientHeight, 0.01, 100);
  camera.position.set(0, 0.9, 2.2);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.5, 0);
  renderer.setAnimationLoop(()=>{
    const dt = clock.getDelta();
    if (mixer) mixer.update(dt);
    controls.update();
    renderer.render(scene, camera);
  });
}
function setStatus(txt, isErr){ const el = document.getElementById('status'); el.textContent = txt; el.className = isErr ? 'err' : ''; }
function setMorph(name, v){
  if (!model) return;
  model.traverse(o=>{ if (o.morphTargetDictionary && o.morphTargetDictionary[name] !== undefined)
    o.morphTargetInfluences[o.morphTargetDictionary[name]] = parseFloat(v); });
}
function toggleAnim(on){
  if (!model) return;
  if (on && animClips.length){
    // GLTFLoader exposes clips on the gltf result, not on mesh objects
    mixer = new THREE.AnimationMixer(model);
    mixer.clipAction(animClips[0]).play();
    setStatus('播放口型动画:' + animClips[0].name + ' · ' + animClips[0].duration.toFixed(2) + 's');
  } else if (mixer){
    mixer.stopAllAction(); mixer = null;
    setStatus('已停止动画');
  }
}
async function loadGlb(url){
  if (model){ scene.remove(model); model = null; mixer = null; }
  setStatus('加载 ' + url + ' …');
  try {
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync(url);
    model = gltf.scene;
    animClips = gltf.animations || [];
    scene.add(model);
    model.traverse(o=>{ if (o.morphTargetDictionary){
      morphNames = Object.keys(o.morphTargetDictionary);
      o.morphTargetInfluences = new Array(morphNames.length).fill(0);
    }});
    setStatus('loaded ' + url + ' · morphs: ' + morphNames.length + ' · anims: ' + animClips.length);
  } catch (e) {
    console.error('[loadGlb] ' + (e && e.stack ? e.stack : e));
    setStatus('加载失败:' + url + ' (' + e + ')', true);
    setTimeout(() => { throw e; }, 0);
  }
}
function loadRigged(){
  const sel = document.getElementById('rigged');
  if (!sel.value){ setStatus('请先选择已有产物', true); return; }
  loadGlb('/output/' + sel.value);
}
function validateImage(){
  const f = document.getElementById('img').files[0];
  if (!f) return null;
  if (!ALLOWED.includes(f.type)) return '仅支持 PNG/JPG/WebP/GIF/BMP 图片(当前:' + (f.type || '未知类型') + ')';
  if (f.size > MAX_MB * 1024 * 1024) return '图片超过 ' + MAX_MB + 'MB 上限';
  return null;
}
async function start(){
  const err = validateImage();
  if (err){ setStatus('❌ ' + err, true); return; }
  setStatus('submitting...');
  const fd = new FormData();
  const img = document.getElementById('img').files[0];
  if (img) fd.append('image', img);
  fd.append('character', document.getElementById('char').value);
  fd.append('text', document.getElementById('text').value);
  let r;
  try { r = await fetch('/api/rig', {method:'POST', body: fd}); }
  catch (e){ setStatus('❌ 网络错误:' + e, true); return; }
  const j = await r.json();
  if (!r.ok){ setStatus('❌ ' + (j.error || ('提交失败 HTTP ' + r.status)), true); return; }
  const poll = setInterval(async ()=>{
    const s = await (await fetch('/api/status/'+j.job)).json();
    setStatus(s.status + '\\n' + (s.log||[]).join('\\n'));
    if (s.status === 'done'){ clearInterval(poll); await loadGlb(s.result.url); }
    if (s.status === 'error'){ clearInterval(poll); setStatus('❌ 生成失败:\\n' + s.error, true); }
  }, 1500);
}
init();
// module-scoped functions are invisible to inline onclick/oninput
// handlers (they resolve in global scope) — expose them explicitly
window.loadRigged = loadRigged;
window.loadGlb = loadGlb;
window.setMorph = setMorph;
window.toggleAnim = toggleAnim;
window.start = start;
Promise.all([
  fetch('/api/characters').then(r=>r.json()),
  fetch('/api/rigged').then(r=>r.json()),
]).then(([chars, rigged])=>{
  const sel = document.getElementById('char');
  const riggedBases = new Set(rigged.map(r=>r.base));
  chars.forEach(c=>{
    const o = document.createElement('option');
    o.value = c; o.textContent = riggedBases.has(c) ? c + ' (已 rig ✓)' : c;
    sel.appendChild(o);
  });
  const rsel = document.getElementById('rigged');
  const hint = document.getElementById('riggedHint');
  if (!rigged.length){
    hint.textContent = '暂无预 rig 产物(批量 rig 进行中…),可先在下方生成。';
    rsel.innerHTML = '<option value="">(无)</option>';
    return;
  }
  rigged.forEach(r=>{
    const o = document.createElement('option');
    o.value = r.name;
    o.textContent = r.name + ' · ' + (r.size/1048576).toFixed(1) + 'MB · ' + r.mtime;
    rsel.appendChild(o);
  });
  hint.textContent = rigged.length + ' 个预 rig 角色,点击"加载预览"直接查看(免重新生成)。';
});
</script></body></html>"""
    return render_template_string(html)


@app.get("/api/health")
def health():
    outdir_ok = os.path.isdir(OUTDIR) and os.access(OUTDIR, os.W_OK)
    glbs_ok = os.path.isdir(GLBDIR)
    checks = {
        "process": "ok",
        "outdir": {"exists": os.path.isdir(OUTDIR), "writable": outdir_ok},
        "glbs": {"exists": glbs_ok, "count": len(_characters())},
        "rigged": {"count": len(_rigged())},
    }
    ok = outdir_ok and glbs_ok
    body = {
        "status": "ok" if ok else "degraded",
        "service": "webapp",
        "uptime_s": int(time.time() - START_TS),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checks": checks,
    }
    return jsonify(body), (200 if ok else 503)


@app.get("/api/characters")
def characters():
    return jsonify(_characters())


@app.get("/api/rigged")
def rigged():
    return jsonify(_rigged())


@app.post("/api/rig")
def rig():
    character = (request.form.get("character") or "ai3d_01.glb").strip()
    text = (request.form.get("text") or "").strip() or "你好世界"
    if not character.endswith(".glb") or "/" in character or "\\" in character:
        return _api_error("角色名不合法")
    if len(text) > 2000:
        return _api_error("口型文本过长(≤2000 字符)")
    img = request.files.get("image")
    img_bytes = None
    if img and img.filename:
        img_bytes = img.read()
        if len(img_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
            return _api_error(f"图片超过 {MAX_UPLOAD_MB}MB 上限")
        detected = _sniff_image(img_bytes[:16])
        if detected is None:
            return _api_error("图片格式不支持,请上传 PNG/JPG/WebP/GIF/BMP")
    audio = request.files.get("audio")
    audio_bytes = None
    if audio and audio.filename:
        audio_bytes = audio.read()
        if len(audio_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
            return _api_error(f"音频超过 {MAX_UPLOAD_MB}MB 上限")
    job = str(uuid.uuid4())[:8]
    payload = {"character": character, "text": text,
               "image_to_mesh": request.form.get("image_to_mesh") == "1"}
    if img_bytes is not None:
        payload["image"] = img_bytes
    if audio_bytes is not None:
        payload["audio"] = audio_bytes
    with LOCK:
        JOBS[job] = {"status": "queued", "log": [], "created": time.time()}
    threading.Thread(target=_run_job, args=(job, payload), daemon=True).start()
    return jsonify({"job": job})


@app.get("/api/status/<job>")
def status(job):
    if job not in JOBS:
        abort(404)
    return jsonify(JOBS[job])


@app.get("/output/<path:name>")
def output(name):
    return send_from_directory(OUTDIR, name)


@app.errorhandler(413)
def too_large(_e):
    if request.path.startswith("/api"):
        return _api_error(f"上传文件超过 {MAX_UPLOAD_MB}MB 上限", 413)
    return _friendly_page("文件过大", f"上传文件超过 {MAX_UPLOAD_MB}MB 上限,请压缩后重试。"), 413


@app.errorhandler(404)
def not_found(_e):
    if request.path.startswith("/api"):
        return _api_error("资源不存在", 404)
    return _friendly_page("页面不存在", "你访问的页面不存在。"), 404


@app.errorhandler(500)
def server_error(_e):
    if request.path.startswith("/api"):
        return _api_error("服务器内部错误", 500)
    return _friendly_page("服务器错误", "服务器内部错误,请稍后重试。"), 500


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    app.run(host="0.0.0.0", port=8000, threaded=True)
