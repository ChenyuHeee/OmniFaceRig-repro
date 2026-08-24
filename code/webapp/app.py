"""Web preview service (deliverable 1): upload image -> rigged GLB preview.

Flask app on port 8000 (externally mapped 32170->8000 / 32171->8001 per the
A100 server doc). Endpoints:
  GET  /                    UI (three.js viewer)
  GET  /api/characters      list of available base characters
  POST /api/rig             start a rig job: {image?, character?, text?}
  GET  /api/status/<job>    job progress
  GET  /output/<file>       serve generated glb
"""

import json
import os
import subprocess
import threading
import time
import uuid

from flask import Flask, jsonify, render_template_string, request, send_from_directory, abort

WORK = os.path.expanduser("~/work")
CODE = os.path.join(WORK, "code")
OUTDIR = os.path.join(WORK, "outputs")
GLBDIR = os.path.join(WORK, "glbs")
CONDA = os.path.expanduser("~/anaconda3/etc/profile.d/conda.sh")
ENV = "torch2.4_cuda12.1"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
JOBS = {}
LOCK = threading.Lock()


def _characters():
    out = []
    for f in sorted(os.listdir(GLBDIR)):
        if f.endswith(".glb"):
            out.append(f)
    return out


def _run_job(job_id, payload):
    try:
        JOBS[job_id]["status"] = "running"
        img = payload.get("image")
        character = payload.get("character", "ai3d_01.glb")
        text = payload.get("text", "你好世界")
        glb = os.path.join(GLBDIR, character)
        if not os.path.exists(glb):
            raise RuntimeError(f"character not found: {character}")
        out_name = f"{job_id}.glb"
        out_path = os.path.join(OUTDIR, out_name)
        cmd = ["python", "-u", "scripts/stage1_real.py",
               "--glb", glb, "--out", out_path, "--text", text]
        if img:
            img_path = os.path.join(OUTDIR, f"{job_id}.img")
            with open(img_path, "wb") as fh:
                fh.write(img)
            cmd += ["--img", img_path]
        env = dict(os.environ)
        proc = subprocess.Popen(
            cmd, cwd=CODE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True,
        )
        log = []
        for line in proc.stdout:
            log.append(line.rstrip())
            JOBS[job_id]["log"] = log[-8:]
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError("\n".join(log[-15:]))
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {"url": f"/output/{out_name}", "size": os.path.getsize(out_path)}
    except Exception as exc:  # noqa
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(exc)


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
 #status{font-family:monospace;font-size:12px;color:#9fb3c8;white-space:pre-wrap;min-height:60px}
 #viewer{width:100%;height:560px;border-radius:8px;background:radial-gradient(#1c2631,#0b0f14)}
 .row{display:flex;gap:10px;align-items:end}
</style></head><body>
<h2>OmniFaceRig-repro · 图 → 带骨骼+表情的 glb (A100)</h2>
<div class="card">
 <div class="row">
  <div><label>上传 2D 角色图(可选)</label><input type="file" id="img" accept="image/*"></div>
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
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
let renderer, scene, camera, controls, model, mixer, clock = new THREE.Clock(), morphNames = [];
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
function setMorph(name, v){
  if (!model) return;
  model.traverse(o=>{ if (o.morphTargetDictionary && o.morphTargetDictionary[name] !== undefined)
    o.morphTargetInfluences[o.morphTargetDictionary[name]] = parseFloat(v); });
}
function toggleAnim(on){
  if (!model) return;
  model.traverse(o=>{ if (o.animations && o.animations.length){
    if (on){ mixer = new THREE.AnimationMixer(o); mixer.clipAction(o.animations[0]).play(); }
    else if (mixer){ mixer.stopAllAction(); mixer = null; }
  }});
}
async function loadGlb(url){
  if (model){ scene.remove(model); model = null; mixer = null; }
  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync(url);
  model = gltf.scene;
  scene.add(model);
  model.traverse(o=>{ if (o.morphTargetDictionary){
    morphNames = Object.keys(o.morphTargetDictionary);
    o.morphTargetInfluences = new Array(morphNames.length).fill(0);
  }});
  document.getElementById('status').textContent = 'loaded ' + url + ' · morphs: ' + morphNames.length;
}
async function start(){
  const status = document.getElementById('status');
  status.textContent = 'submitting...';
  const fd = new FormData();
  const img = document.getElementById('img').files[0];
  if (img) fd.append('image', img);
  fd.append('character', document.getElementById('char').value);
  fd.append('text', document.getElementById('text').value);
  const r = await fetch('/api/rig', {method:'POST', body: fd});
  const j = await r.json();
  const poll = setInterval(async ()=>{
    const s = await (await fetch('/api/status/'+j.job)).json();
    status.textContent = s.status + '\n' + (s.log||[]).join('\n');
    if (s.status === 'done'){ clearInterval(poll); await loadGlb(s.result.url); }
    if (s.status === 'error'){ clearInterval(poll); status.textContent = 'ERROR: ' + s.error; }
  }, 1500);
}
init();
fetch('/api/characters').then(r=>r.json()).then(cs=>{
  const sel = document.getElementById('char');
  cs.forEach(c=>{ const o = document.createElement('option'); o.value = o.textContent = c; sel.appendChild(o); });
});
</script></body></html>"""
    return render_template_string(html)


@app.get("/api/characters")
def characters():
    return jsonify(_characters())


@app.post("/api/rig")
def rig():
    job = str(uuid.uuid4())[:8]
    payload = {"character": request.form.get("character", "ai3d_01.glb"),
               "text": request.form.get("text", "你好世界")}
    img = request.files.get("image")
    if img:
        payload["image"] = img.read()
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


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    app.run(host="0.0.0.0", port=8000, threaded=True)
