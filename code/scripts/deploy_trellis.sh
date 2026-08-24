#!/usr/bin/env bash
# deploy_trellis.sh — deploy the TRELLIS (v1) image-to-mesh front-end
# (issue #3: 图→mesh 前端). Creates a conda env, installs the TRELLIS
# source checkout, and fetches all weights with resume-capable curl.
#
# Usage:
#   ./deploy_trellis.sh            # full deploy (env + code + weights)
#   ./deploy_trellis.sh --weights  # only fetch/verify weights (no GPU needed)
#   ./deploy_trellis.sh --vram     # only print GPU/VRAM check
#   ./deploy_trellis.sh --help
#
# Environment (all optional):
#   HF_ENDPOINT         HF endpoint for weights, default https://hf-mirror.com
#                       (huggingface.co is unreachable from the dev machine;
#                        hf-mirror measured ~8-11 MB/s on 2026-08-25).
#   TRELLIS_WEIGHTS_DIR where TRELLIS-image-large lands,
#                       default $HOME/.cache/trellis/TRELLIS-image-large
#   TRELLIS_SRC         where the TRELLIS source checkout goes,
#                       default $HOME/trellis
#   TORCH_HOME          torch hub cache (dinov2 image encoder),
#                       default $HOME/.cache/torch
#   CONDA_ENV           conda env name, default trellis
#
# Network reality measured 2026-08-25 (dev machine):
#   huggingface.co direct         0 B/s (connection timeout)
#   hf-mirror.com                 ~8-11 MB/s  (verified: 141 MB in 18 s,
#                                              1.2 GB in 114 s, full 3.3 GB ok)
#   codeload.github.com           ~8 MB/s  (TRELLIS source tarball)
#   github release assets         ~137 KB/s (rembg u2net.onnx, 176 MB ~21 min)
#   dl.fbaipublicfiles.com        403      (dinov2 -> use HF mirror instead)
#   gh API                        ok (fallback for source tarballs when
#                                      github.com direct is blocked)
#
# So: the 3.3 GB TRELLIS-image-large weights are *not* a blocker on this
# network (hf-mirror is fast); only the small aux models (dinov2 / rembg)
# needed workarounds, handled below.

set -euo pipefail

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
WEIGHTS_DIR="${TRELLIS_WEIGHTS_DIR:-$HOME/.cache/trellis/TRELLIS-image-large}"
TRELLIS_SRC="${TRELLIS_SRC:-$HOME/trellis}"
TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
CONDA_ENV="${CONDA_ENV:-trellis}"

REPO="microsoft/TRELLIS-image-large"
BASE="$HF_ENDPOINT/$REPO/resolve/main"
DINO_PTH_SRC="https://hf-mirror.com/showstarpro/dinov2_vitl14_reg4_pretrain/resolve/main/dinov2_vitl14_reg4_pretrain.pth"
U2NET_ONNX_URL="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
U2NET_HOME="${U2NET_HOME:-$HOME/.u2net}"

# (name, expected bytes) — sizes from HF API on 2026-08-25
declare -a WEIGHT_FILES=(
  "ckpts/ss_dec_conv3d_16l8_fp16.json 245"
  "ckpts/ss_dec_conv3d_16l8_fp16.safetensors 147591972"
  "ckpts/ss_enc_conv3d_16l8_fp16.json 244"
  "ckpts/ss_enc_conv3d_16l8_fp16.safetensors 119068016"
  "ckpts/ss_flow_img_dit_L_16l8_fp16.json 385"
  "ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors 1130770840"
  "ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.json 843"
  "ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors 171450952"
  "ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.json 372"
  "ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors 181903412"
  "ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.json 396"
  "ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.safetensors 171450488"
  "ckpts/slat_enc_swin8_B_64l8_fp16.json 321"
  "ckpts/slat_enc_swin8_B_64l8_fp16.safetensors 173242816"
  "ckpts/slat_flow_img_dit_L_64l8p2_fp16.json 442"
  "ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors 1203755136"
  "pipeline.json 1987"
  "README.md 464"
  ".gitattributes 1519"
)

log() { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# weights (hf-mirror, curl -C - resume, size-verified)
# ---------------------------------------------------------------------------
download_weights() {
  log "weights -> $WEIGHTS_DIR  (endpoint: $HF_ENDPOINT)"
  mkdir -p "$WEIGHTS_DIR"
  local fail=0
  for entry in "${WEIGHT_FILES[@]}"; do
    local rel size
    rel="${entry%% *}"; size="${entry##* }"
    local dest="$WEIGHTS_DIR/$rel"
    mkdir -p "$(dirname "$dest")"
    if [[ -f "$dest" ]] && [[ "$(stat -c %s "$dest" 2>/dev/null || echo 0)" == "$size" ]]; then
      log "ok   $rel ($size bytes)"
      continue
    fi
    log "get  $rel ($size bytes) [resume-capable]"
    # -C - resumes partial downloads; --retry helps on flaky CDNs
    if ! curl -sSL --retry 3 -C - -o "$dest" "$BASE/$rel"; then
      warn "download failed for $rel — will retry next run (curl -C - resumes)"
      fail=1
      continue
    fi
    local got; got="$(stat -c %s "$dest" 2>/dev/null || echo 0)"
    if [[ "$got" != "$size" ]]; then
      warn "size mismatch for $rel: got $got, want $size (will re-fetch)"
      rm -f "$dest"
      fail=1
    fi
  done
  # sanity: every inference checkpoint must be present (json + safetensors).
  # Names = file basenames under ckpts/ (pipeline.json model values), NOT the
  # pipeline model keys (slat_decoder_gs_* -> files are slat_dec_gs_*).
  local missing=0
  for name in ss_dec_conv3d_16l8_fp16 ss_flow_img_dit_L_16l8_fp16 \
              slat_dec_gs_swin8_B_64l8gs32_fp16 slat_dec_rf_swin8_B_64l8r16_fp16 \
              slat_dec_mesh_swin8_B_64l8m256c_fp16 slat_flow_img_dit_L_64l8p2_fp16; do
    for ext in json safetensors; do
      [[ -s "$WEIGHTS_DIR/ckpts/$name.$ext" ]] || { warn "inference ckpt missing: ckpts/$name.$ext"; missing=1; }
    done
  done
  if [[ "$missing" == 1 ]]; then die "weights incomplete — re-run this script"; fi
  log "weights complete: $(du -sh "$WEIGHTS_DIR" | cut -f1)"
}

# ---------------------------------------------------------------------------
# dinov2 image encoder (torch hub cache pre-seed)
# ---------------------------------------------------------------------------
download_dinov2() {
  local dst="$TORCH_HOME/hub/checkpoints/dinov2_vitl14_reg_pretrain.pth"
  mkdir -p "$(dirname "$dst")"
  # clear stale partial from an interrupted run (final file takes precedence)
  if [[ -s "$dst" ]]; then rm -f "$dst.tmp"; log "dinov2 already cached: $dst"; return 0; fi
  log "get  dinov2_vitl14_reg (~1.2 GB) -> $dst"
  # dl.fbaipublicfiles.com is blocked (403); mirror hosts the identical
  # native torch.hub checkpoint (1_217_607_321 bytes, verified keys:
  # cls_token/pos_embed/register_tokens/blocks.* — no prefix).
  curl -sSL --retry 3 -C - -o "$dst.tmp" "$DINO_PTH_SRC" || die "dinov2 download failed"
  mv "$dst.tmp" "$dst"
  log "dinov2 cached: $dst ($(stat -c %s "$dst") bytes)"
}

# ---------------------------------------------------------------------------
# rembg u2net (background removal in TRELLIS preprocess_image)
# ---------------------------------------------------------------------------
download_rembg() {
  mkdir -p "$U2NET_HOME"
  if [[ -s "$U2NET_HOME/u2net.onnx" ]]; then log "u2net already present"; return 0; fi
  log "get  rembg u2net.onnx (176 MB) -> $U2NET_HOME/u2net.onnx"
  log "     (github release ~137 KB/s here ≈ 21 min; alternative mirrors:"
  log "      hf-mirror.com/Gulraiz00/u2net u2net.onnx — same size, fast,"
  log "      but a slightly different build; fine for smoke tests)"
  # optional: cap at 3 min so a slow link cannot stall the deploy; on
  # failure trellis_front.py falls back to naive preprocessing.
  if timeout 180 curl -sSL --retry 3 -C - -o "$U2NET_HOME/u2net.onnx" "$U2NET_ONNX_URL" \
     && [[ -s "$U2NET_HOME/u2net.onnx" ]]; then
    log "u2net done: $U2NET_HOME/u2net.onnx"
  else
    warn "u2net not fetched (slow/blocked link) — trellis_front.py will fall"
    warn "back to naive preprocessing (no background removal) and log a warning."
  fi
}

# ---------------------------------------------------------------------------
# TRELLIS source checkout (github direct ~8 MB/s via codeload tarball)
# ---------------------------------------------------------------------------
fetch_source() {
  if [[ -d "$TRELLIS_SRC/trellis" ]]; then log "source already at $TRELLIS_SRC"; return 0; fi
  log "fetch TRELLIS source -> $TRELLIS_SRC"
  mkdir -p "$TRELLIS_SRC"
  # primary: codeload tarball (fast, ~8 MB/s). Submodules are fetched
  # separately below (they are not in the tarball).
  local tarball="$TRELLIS_SRC/trellis-main.tar.gz"
  curl -sSL --retry 3 -o "$tarball" "https://codeload.github.com/microsoft/TRELLIS/tar.gz/refs/heads/main" \
    || die "TRELLIS tarball download failed"
  tar -xzf "$tarball" -C "$TRELLIS_SRC" --strip-components=1
  rm -f "$tarball"

  # fallback when github.com direct is blocked but the gh API channel works:
  #   gh api repos/microsoft/TRELLIS/tarball/main > trellis.tar.gz
  #   (same layout; then tar -xzf with --strip-components=1)

  # submodules (CUDA kernels): diffoctreerast (radiance field rendering) and
  # modified FlexiCubes (mesh extraction). Each is a small tarball; try the
  # repo's default branch (JeffreyXiang/diffoctreerast=master,
  # MaxtirError/FlexiCubes=main).
  mkdir -p "$TRELLIS_SRC/submodules"
  for sm in "JeffreyXiang/diffoctreerast" "MaxtirError/FlexiCubes"; do
    local name="${sm##*/}"
    local dst="$TRELLIS_SRC/submodules/$name"
    if [[ -d "$dst" ]]; then continue; fi
    log "submodule $name"
    local got=""
    for br in main master; do
      if curl -sSL --retry 2 -o "$TRELLIS_SRC/$name.tar.gz" \
           "https://codeload.github.com/$sm/tar.gz/refs/heads/$br" \
         && tar -tzf "$TRELLIS_SRC/$name.tar.gz" >/dev/null 2>&1; then
        got="$br"; break
      fi
    done
    if [[ -z "$got" ]]; then
      warn "submodule $name tarball failed (try: gh api repos/$sm/tarball)"
      continue
    fi
    mkdir -p "$dst"
    tar -xzf "$TRELLIS_SRC/$name.tar.gz" -C "$dst" --strip-components=1
    rm -f "$TRELLIS_SRC/$name.tar.gz"
    log "  $name from $got branch"
  done
  # NOTE: if git works (ssh:443), the simplest source fetch is:
  #   git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git "$TRELLIS_SRC"
}

# ---------------------------------------------------------------------------
# conda env + pip deps
# ---------------------------------------------------------------------------
install_env() {
  if ! command -v conda >/dev/null 2>&1; then
    warn "conda not found — skipping env creation. Install deps manually:"
    warn "  python>=3.10, torch (CUDA 11.8/12.2), trellis source deps (see setup.sh)"
    return 0
  fi
  if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    log "conda create -n $CONDA_ENV python=3.10"
    conda create -y -n "$CONDA_ENV" python=3.10
  fi
  log "conda env '$CONDA_ENV' ready — install TRELLIS deps inside it:"
  log "  conda activate $CONDA_ENV"
  log "  cd $TRELLIS_SRC && . ./setup.sh --basic --xformers --diffoctreerast --spconv"
  log "  # (add --flash-attn on Ampere+; see setup.sh --help; the server's"
  log "  #  torch2.4_cuda12.1 env can also be reused by pip install -e $TRELLIS_SRC)"
  log "  # CPU-only smoke test (no CUDA kernels): pip install -e $TRELLIS_SRC"
}

# ---------------------------------------------------------------------------
# VRAM check
# ---------------------------------------------------------------------------
vram_check() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found — no NVIDIA GPU visible on this host."
    warn "TRELLIS v1 needs >= 16 GB VRAM (A100/A6000 verified by authors)."
    return 0
  fi
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
  local gb
  gb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
  if [[ "${gb:-0}" -lt 16384 ]]; then
    warn "GPU has ${gb} MiB VRAM < 16384 MiB — TRELLIS v1 may OOM;"
    warn "use the A100-40GB server for real inference (mock mode needs no GPU)."
  else
    log "GPU VRAM OK for TRELLIS v1 (>= 16 GB)."
  fi
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
MODE="${1:---all}"
case "$MODE" in
  --weights) download_weights; download_dinov2; download_rembg ;;
  --source)  fetch_source ;;
  --vram)    vram_check ;;
  --all|"")  vram_check; fetch_source; download_weights; download_dinov2;
             download_rembg; install_env ;;
  --help|-h) sed -n '1,40p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) die "unknown flag $MODE (see --help)" ;;
esac
log "done."
