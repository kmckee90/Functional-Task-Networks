"""Make a complete, smaller Notion upload copy; originals remain untouched."""
import json,sys
from pathlib import Path
import av
root=Path('/home/dev/globo_dreamer_state/notion_proposal')
bundle=json.loads((root/'bundle.json').read_text())
a=next(a for a in bundle['assets'] if a['key']=='ASSET000')
source_path=Path(sys.argv[1]) if len(sys.argv)>1 else Path(a['path'])
dest=root/(sys.argv[2] if len(sys.argv)>2 else 'nascent-globo-montage.mp4')
assert source_path.resolve()!=dest.resolve(), 'Source and output must differ'
count=0
with av.open(str(source_path)) as source,av.open(str(dest),'w',options={'movflags':'+faststart'}) as out:
    vs=source.streams.video[0];vs.thread_count=2
    target=out.add_stream('libx264',rate=vs.average_rate)
    target.width=vs.width;target.height=vs.height;target.pix_fmt='yuv420p';target.bit_rate=480000
    target.options={'preset':'slow','threads':'2','maxrate':'520k','bufsize':'1040k'}
    for frame in source.decode(video=0):
        for packet in target.encode(frame):out.mux(packet)
        count+=1
    for packet in target.encode():out.mux(packet)
assert dest.stat().st_size<20*1024**2
a['path']=str(dest);a['filename']=dest.name
(root/'bundle.json').write_text(json.dumps(bundle))
print(json.dumps({'frames':count,'bytes':dest.stat().st_size}),flush=True)
