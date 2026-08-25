"""Animate a rigged glb with REAL audio-driven lip-sync.

Usage:
  # text -> piper TTS (real phoneme timestamps)
  python animate_audio.py --glb in.glb --out out.glb --text "Hello world" --lang en

  # real audio + transcript -> faster-whisper word timestamps
  python animate_audio.py --glb in.glb --out out.glb --audio speech.wav \
      --text "Hello world" --lang en

The WEIGHTS animation (52 ARKit channels, real seconds) is appended to the
existing rigged glb; original skeleton/body animation is untouched.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omnifacerig_repro import audio_lipsync


def _whisper_model(models_dir):
    d = os.path.join(models_dir, "faster-whisper-base")
    f = os.path.join(models_dir, "faster-whisper-base.bin")
    return d if os.path.isdir(d) else f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True, help="input rigged glb (52 morphs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", required=True, help="transcript / TTS text")
    ap.add_argument("--audio", default=None, help="real audio for whisper alignment")
    ap.add_argument("--lang", default="en", choices=["en", "zh"])
    ap.add_argument("--models", default=os.path.expanduser("~/work/models"),
                    help="dir with piper models + faster-whisper bins")
    args = ap.parse_args()

    if args.audio:
        # REAL audio + transcript -> faster-whisper word timestamps
        model = _whisper_model(args.models)
        track = audio_lipsync.whisper_viseme_track(
            args.audio, args.text, lang=args.lang, model_path=model)
        src = f"whisper({os.path.basename(args.audio)})"
    else:
        # text -> piper TTS (real speech) -> whisper alignment (unified)
        piper_model = (os.path.join(args.models, "zh_CN-huayan-medium.onnx")
                       if args.lang == "zh" else
                       os.path.join(args.models, "en_US-lessac-medium.onnx"))
        if not os.path.exists(piper_model):
            raise SystemExit(f"piper model missing: {piper_model}")
        wav = os.path.join(os.path.dirname(args.out), os.path.basename(args.out) + ".tts.wav")
        audio_lipsync.piper_tts_wav(args.text, piper_model, wav)
        track = audio_lipsync.whisper_viseme_track(
            wav, args.text, lang=args.lang, model_path=_whisper_model(args.models))
        os.unlink(wav)
        src = f"piper-tts({args.lang})"

    if not track:
        raise SystemExit("empty viseme track - check audio/transcript match")

    audio_lipsync.animate_glb(args.glb, args.out, track)
    anim = audio_lipsync.viseme_track_to_animation(track)
    print(json.dumps({
        "out": args.out, "source": src, "visemes": len(track),
        "duration": round(float(anim["times"][-1]), 2),
        "first": track[:4],
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
