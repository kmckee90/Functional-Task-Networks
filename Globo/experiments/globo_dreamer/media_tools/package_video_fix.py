"""Package only the updated headline montage and the two pages using it."""
import hashlib,json,zipfile
from pathlib import Path
root=Path('/home/dev/globo_dreamer_state/export_video_v3/proposal_offline')
manifest=json.loads((root/'manifest.json').read_text())
assets=manifest['assets']
video=next(v for k,v in assets.items() if k.endswith('/nascent-annotated-preview-v3.mp4'))
poster=next(v for k,v in assets.items() if k.endswith('/nascent-annotated-preview.jpg'))
files=['index.html','projects-globo.html',video,poster]
for name in files[:2]:
    page=(root/name).read_text()
    assert video in page and 'montage-smooth-climbing-preview.mp4' not in page
assert hashlib.sha256((root/video).read_bytes()).hexdigest()=='e735b6c149b7c2365803ea11219382e56d798725810a9e203fadfb49694bee48'
out=root.parent/'nascent-github-video-update.zip'
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name in files:z.write(root/name,name)
print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':files}),flush=True)
