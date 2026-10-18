"""Create brief, pod-only presentation excerpts from the supplied proposal assets."""
import os
from pathlib import Path
import subprocess
import imageio_ffmpeg

source = Path(__file__).resolve().parent.parent / 'proposal assets'
output = Path(os.environ.get('PROPOSAL_MEDIA', '/home/dev/proposal_state/media'))
output.mkdir(parents=True, exist_ok=True)
clips = {
    'globo': 'exploration/globo/old/s60_c000006_s19_behavior_000040001.mp4',
    'maze': 'exploration/maze/random.webm',
    'maze-continual': 'exploration/maze/continual.webm',
    'robotics': 'planning and representation/Robotics/m3_demo.gif',
    'reconfiguration': 'misc/demo_reconfig_corr_1k.gif',
}
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
for name, relative in clips.items():
    target = output / f'{name}.mp4'
    if not target.exists():
        temp = output / f'{name}.tmp.mp4'
        subprocess.run([ffmpeg, '-v', 'error', '-y', '-i', str(source/relative),
            '-t', '8', '-an', '-vf', r'scale=min(960\,iw):-2,fps=12',
            '-c:v', 'libx264', '-threads', '2', '-crf', '25', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(temp)], check=True)
        temp.replace(target)
    poster = output / f'{name}.jpg'
    if not poster.exists():
        subprocess.run([ffmpeg, '-v', 'error', '-y', '-i', str(target),
            '-frames:v', '1', str(poster)], check=True)
    print(name, target.stat().st_size, 'bytes', flush=True)
