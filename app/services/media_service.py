# app/services/media_service.py
import os, subprocess, uuid
from typing import Optional, Tuple

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def join_image_audio_to_video(image_path: str, audio_path: str, output_dir: str) -> Tuple[str, float]:
    """
    Create an mp4 video from a still image and an audio track.
    Returns (output_path, duration_seconds).
    """
    ensure_dir(output_dir)
    out_name = f"merge_{uuid.uuid4().hex[:8]}.mp4"
    out_path = os.path.join(output_dir, out_name)

    # Probe duration (optional)
    def probe_duration(p: str) -> float:
        try:
            cmd = ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",p]
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            return float(r.stdout.strip())
        except Exception:
            return 0.0

    audio_dur = probe_duration(audio_path)

    # Simple still-image video, scaled and with audio; -shortest cuts to audio length
    cmd = [
        "ffmpeg","-y",
        "-loop","1","-i", image_path,
        "-i", audio_path,
        "-c:v","libx264","-tune","stillimage",
        "-pix_fmt","yuv420p",
        "-c:a","aac","-b:a","192k",
        "-shortest",
        "-movflags","+faststart",
        "-vf","scale=1280:-2,format=yuv420p",
        out_path
    ]
    subprocess.run(cmd, check=True)
    duration = audio_dur if audio_dur > 0 else 0.0
    return out_path, duration
