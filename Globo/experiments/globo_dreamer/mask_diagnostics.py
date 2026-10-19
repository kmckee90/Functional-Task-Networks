"""Small overwrite-only mask pop-out, generated separately from live workers."""
import html
import json


def publish(live,metadata,title,status):
    import av
    from .search import atomic
    live=live.resolve();step=metadata['step']
    stamp=live/'mask_video.json'
    try:previous=json.loads(stamp.read_text())
    except (OSError,ValueError):previous={}
    if previous.get('step')!=step:
        source=av.open(str(live/'obs.mp4'))
        target=av.open(str(live/'bw_pending.mp4'),'w')
        stream=target.add_stream('libx264',rate=6);stream.width=256;stream.height=280;stream.pix_fmt='yuv420p'
        for frame in source.decode(video=0):
            tile=frame.to_image().crop((256,0,512,280))
            for packet in stream.encode(av.VideoFrame.from_image(tile)):target.mux(packet)
        for packet in stream.encode():target.mux(packet)
        source.close();target.close();(live/'bw_pending.mp4').replace(live/'bw.mp4')
        atomic(stamp,dict(step=step,frame_start=metadata['frame_start'],frame_end=metadata['frame_end']))
    atomic(live/'mask_status.json',dict(step=status.get('step',0),**status.get('ftn',{})))
    text=f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · mask diagnostic</title>
<style>body{{background:#141820;color:#eee;font:16px system-ui;max-width:950px;margin:24px auto;padding:16px}}
a{{color:#9bd0ff}}video{{width:512px;max-width:100%;image-rendering:pixelated}}img{{width:256px;max-width:100%}}
.goals{{display:flex;gap:24px;flex-wrap:wrap}}pre{{white-space:pre-wrap}}</style>
<h1>{html.escape(title)}</h1><p>White: active neurons. Purple/blue: fading activity. Display permutation changes neither neuron identities nor their overlaps.</p>
<video controls autoplay muted loop playsinline src="bw.mp4?v={step}"></video>
<p>Recorded observations {metadata['frame_start']:,}–{metadata['frame_end']:,}. This is the action mask, recorded concurrently with the observation.</p>
<div class="goals" id="goals"></div><pre id="stats"></pre><p><a href="../">Back to monitor</a></p>
<script>
async function update(){{
 const s=await fetch('mask_status.json',{{cache:'no-store'}}).then(r=>r.json());
 document.getElementById('stats').textContent=JSON.stringify(s,null,2);
 const r=await fetch('goal_status.json',{{cache:'no-store'}});if(!r.ok)return;
 const g=await r.json();let out='';
 if(g.active_goal)out+=`<div><h3>Active goal ${{g.active_goal}}</h3><img src="active_goal.jpg?v=${{g.step}}"></div>`;
 if(g.frontier_goal&&g.frontier_goal!==g.active_goal)out+=`<div><h3>Final frontier ${{g.frontier_goal}}</h3><img src="frontier_goal.jpg?v=${{g.step}}"></div>`;
 if(!g.active_goal)out='<p>Currently collecting new experience with density exploration.</p>';
 document.getElementById('goals').innerHTML=out+`<p>Current goal state at observation ${{g.step}}. Goals inside the main video correspond to that clip’s historical frames.</p>`;
}}
update().catch(console.error);setInterval(()=>update().catch(console.error),10000);
setInterval(async()=>{{const s=await fetch('mask_video.json',{{cache:'no-store'}}).then(r=>r.json());if(s.step!=={step})location.reload()}},30000);
</script>'''
    tmp=live/'mask.tmp';tmp.write_text(text);tmp.replace(live/'mask.html')
