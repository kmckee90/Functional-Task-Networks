"""Build a portable, interactive before/after gallery from actual model samples."""
import json
from pathlib import Path


def main():
    out = Path('/home/dev/workspace/continual/artifacts')
    manifest = {}
    for name in ['bayes_replay_s123','bayes_noreplay_s123','plain_lora_s123','bayes_replay_reverse_s123']:
        history = json.loads((out/name/'lifecycle.json').read_text())
        manifest[name] = [r['stage'] for r in history]
    page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SiT continual-learning demo</title><style>
body{font:17px system-ui;background:#f4f4ee;color:#183b45;max-width:1500px;margin:35px auto;padding:0 25px}
select,button{font:inherit;padding:8px;margin:4px;border:1px solid #8ba1a3;border-radius:6px;background:white}
.controls{position:sticky;top:0;background:#f4f4ee;padding:10px 0;border-bottom:1px solid #ccd5d4}
img{width:100%;background:white;border-radius:6px}figure{margin:20px 0}figcaption{font-weight:650;margin:8px 0}
input{width:100%;margin:15px 0}a{color:#006b83}.hint{font-size:14px;color:#586a6e}h1{margin-bottom:10px}
</style><h1>Can SiT remember earlier image domains?</h1>
<p>These are actual samples from one pretrained SiT with an evolving shared adapter bank.
Choose a domain and move through training. Every grid uses the same noise seeds and class labels.
No images are manually selected or retouched.</p>
<p><a href="report.html">Results report</a> · <a href="index.html">All matrices and samples</a></p>
<div class="controls"><label>Method <select id="method"></select></label>
<label>Requested domain <select id="domain"><option>monochrome</option><option selected>poster</option><option>sketch</option><option>photo</option></select></label>
<button id="play">Play lifecycle</button><input id="stage" type="range" min="0" max="3" value="3"><strong id="stageText"></strong></div>
<figure><figcaption>Recall through cold Bayesian mask search — 40 reference images, frozen weights</figcaption><a id="coldLink"><img id="cold"></a></figure>
<figure id="storedFigure"><figcaption>Recall through the mask saved when this domain was learned — same current weights</figcaption><a id="storedLink"><img id="stored"></a></figure>
<p id="storedMissing"></p>
<figure><figcaption>Before any post-training — same requested domain and inference protocol</figcaption><a id="initialLink"><img id="initial"></a></figure>
<p class="hint">Columns: tench, English springer, cassette player, chainsaw, church, French horn, garbage truck, gas pump, golf ball, parachute. Two fixed noise draws per class. The photo domain is never post-trained. Ordinary LoRA has one active adapter, so its “stored mask” cannot isolate past domains. Missing stored grids mean that domain has not yet been trained. Click a grid to inspect the full image.</p>
<script>const data=MANIFEST;
const names={bayes_replay_s123:'Bayes FTN + mask-filtered replay',bayes_noreplay_s123:'Bayes FTN, no replay',plain_lora_s123:'Ordinary LoRA',bayes_replay_reverse_s123:'Bayes FTN + replay, reversed order'};
const method=document.querySelector('#method'),domain=document.querySelector('#domain'),stage=document.querySelector('#stage');
for(const name of Object.keys(data)){const o=document.createElement('option');o.value=name;o.textContent=names[name];method.append(o)}
function setImage(id,url){document.getElementById(id).src=url;document.getElementById(id+'Link').href=url}
function update(){const name=method.value,d=domain.value,s=data[name][Number(stage.value)];
document.querySelector('#stageText').textContent=s==='initial'?'Before post-training':s.replaceAll('_',' ');
setImage('cold',`${name}/samples/${s}/${d}_cold.png`);setImage('initial',`${name}/samples/initial/${d}_cold.png`);
const seen=data[name].slice(1,Number(stage.value)+1).some(x=>x.endsWith('_'+d));
document.querySelector('#storedFigure').hidden=!seen;document.querySelector('#storedMissing').textContent=seen?'':'No saved training mask exists for this domain at this stage.';
if(seen)setImage('stored',`${name}/samples/${s}/${d}_stored.png`)}
method.onchange=domain.onchange=stage.oninput=update;let timer=null;
document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.querySelector('#play').textContent='Play lifecycle';return}stage.value=0;update();document.querySelector('#play').textContent='Pause';timer=setInterval(()=>{stage.value=(Number(stage.value)+1)%4;update()},3000)};update();</script>'''
    (out/'demo.html').write_text(page.replace('MANIFEST',json.dumps(manifest)))


if __name__ == '__main__':
    main()
