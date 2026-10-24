"""Baseline and scaled unlimited single-stream experiments; all state is pod-only."""
import fcntl
import html
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .search import atomic, alive
from .search60 import FIXED
from .achievement_charts import COLORS

STATE=Path(os.environ.get('GLOBO_STATE','/home/dev/globo_dreamer_state'))
ROOT=STATE/'continuous_runs'
OUT=STATE/'reviews/config24/continuous'


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def comparison_label(t):
    if t.get('frontier_ftn'):return 'FTN–M2E · '+('retained frontiers' if t['frontier_ftn']=='frontier' else 'waypoint planning')
    if t.get('m2e_ftn'):return 'FTN–M2E · '+('flat MLP' if t['m2e_ftn']=='flat' else 'convolutional')
    if t.get('mf_init_weights'):return 'M2E warm-start · 256px render'
    if t.get('plan2explore'):return 'Plan2Explore (P2E)'
    if t.get('dreamer_baseline'):
        return 'Dreamer+Novelty' + (' (stopped)' if t.get('status')=='paused' else '')
    if t.get('model_free_only'):return 'MetaLearning2Explore (M2L)'
    if t.get('ftn'):return 'FTN+M2L+Dreamer'
    return f'Dreamer + explorer · weight {t["weight"]:g}' + (' (stopped)' if t.get('status')=='paused' else '')


def comparison_groups(trials):
    eligible=[t for t in trials if t.get('seed') is not None and
        ((t.get('dreamer_baseline') and not t.get('plan2explore')) or
         ((t.get('compare') or t.get('status') in ('running','pending')) and (t.get('plan2explore') or
          t.get('ftn') or t.get('model_free_only') or (t.get('scaled') and t.get('weight') in (4,16)))))]
    for seed in sorted({t['seed'] for t in eligible}):
        yield seed,sorted((t for t in eligible if t['seed']==seed),key=lambda t:(
            0 if t.get('dreamer_baseline') else 4 if t.get('plan2explore') else
            3 if t.get('ftn') else 2 if t.get('model_free_only') else 1,t.get('weight',0)))


def spatial_comparisons(trials, focused=False):
    if focused:
        trials=[t for t in trials if t.get('m2e_ftn') or t.get('plan2explore') or
            (t.get('model_free_only') and not t.get('mf_init_weights') and not t.get('ftn'))]
    parts=[('<h2>FTN–M2E experiments versus base M2E and Plan2Explore</h2><p>'
        'Cumulative spatial discoveries at matched observation counts, one chart per seed. '
        'The axis follows the longest new FTN–M2E run; archived baseline traces are clipped to that window. '
        'Uncheck legend entries to isolate methods.</p>') if focused else
        '<h2>Seed-matched spatial progress</h2><p>Cumulative discovered spatial cells versus observations. '
        'The shared axis includes the compared runs; shorter traces end at their latest observation.</p>']
    for seed,matched in comparison_groups(trials):
        # Preserve the useful early comparison scale, extending it as the new
        # baselines/FTNs advance beyond the now-stopped Dreamer baseline.
        xmax=max([1]+[read(STATE/'runs'/t['name']/'status.json').get('step',0)
            for t in matched if t.get('dreamer_baseline') or t.get('model_free_only') or t.get('ftn') or t.get('plan2explore')])
        if focused:
            xmax=max([1]+[read(STATE/'runs'/t['name']/'status.json').get('step',0)
                for t in matched if t.get('m2e_ftn')])
        series=[]
        for t in matched:
            run=STATE/'runs'/t['name']; rows=[]
            try:
                for line in (run/'live/history.jsonl').read_text().splitlines():
                    try:rows.append(json.loads(line))
                    except ValueError:continue
            except OSError:pass
            points={0:0}
            for row in rows:
                if 0<=row['step']<=xmax:points[row['step']]=row['total_cells']
            status=read(run/'status.json')
            if 0<status.get('step',0)<=xmax and 'cells_lifetime' in status.get('environment',{}):
                points[status['step']]=status['environment']['cells_lifetime']
            label=comparison_label(t)
            series.append((label,sorted(points.items())))
        ymax=max(1,max(y for _,points in series for x,y in points))
        parts.append(f'<section class="spatial-comparison" data-seed="{seed}" data-max-step="{xmax}"><h3>Seed {seed} · 0–{xmax:,} observations</h3>'
            f'<svg role="img" aria-label="Seed {seed}: cumulative spatial discoveries" viewBox="0 0 900 300" style="height:auto;max-width:100%">')
        for i in range(5):
            x=65+810*i/4;y=240-205*i/4
            parts.append(f'<path d="M65 {y} H875" stroke="#343c48"/><text x="55" y="{y+4}" text-anchor="end" fill="#bbb" font-size="12">{ymax*i/4:.0f}</text>'
                f'<text x="{x}" y="260" text-anchor="middle" fill="#bbb" font-size="12">{xmax*i/4:,.0f}</text>')
        for (label,points),color in zip(series,(COLORS[i%len(COLORS)] for i in range(len(series)))):
            # Stair steps reflect cumulative counts; never include future samples.
            path=f'M65 {240-205*points[0][1]/ymax:.2f}'
            for x,y in points[1:]:path+=f' H{65+810*x/xmax:.2f} V{240-205*y/ymax:.2f}'
            parts.append(f'<path class="comparison-line" data-run="{seed}:{html.escape(label,quote=True)}" d="{path}" stroke="{color}" stroke-width="2" fill="none"><title>{label}</title></path>')
        parts.append('<text x="470" y="289" text-anchor="middle" fill="#ddd" font-size="13">Observations</text>'
            '<text x="65" y="19" fill="#ddd" font-size="13">Cumulative spatial cells</text></svg><p>')
        for (label,_),color in zip(series,(COLORS[i%len(COLORS)] for i in range(len(series)))):
            parts.append(f'<label style="color:{color};margin-right:24px;white-space:nowrap;cursor:pointer">'
                f'<input type="checkbox" class="run-toggle" data-run="{seed}:{html.escape(label,quote=True)}" checked> ━ {label}</label>')
        parts.append('</p></section>')
    return ''.join(parts)


def gallery(trials, archived=False):
    # Phase 2 has its own federated monitor; this page remains Phase 1.
    trials=[t for t in trials if t.get('phase') not in ('phase2','phase3','phase4','phase5')]
    comparison_html=spatial_comparisons(trials,focused=True) if any(t.get('m2e_ftn') for t in trials) else ''
    comparison_html+=spatial_comparisons(trials)
    from .achievement_charts import achievement_charts,achievement_checklist,run_charts
    comparison_html+=achievement_charts(list(comparison_groups(trials)),STATE,comparison_label)
    trials=sorted((t for t in trials if (t.get('archive_display') if archived else t['status']=='running' and alive(t))),
        key=lambda t:(3 if t.get('mf_init_weights') else 0 if t.get('ftn') else 2 if t.get('dreamer_baseline') or t.get('model_free_only') or t.get('plan2explore') else 1,t['gpu']))
    OUT.mkdir(parents=True,exist_ok=True)
    parts=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Continuous density + model exploration</title><style>body{background:#141820;color:#eee;font:16px system-ui;'
        'max-width:1200px;margin:24px auto;padding:16px}a{color:#9bd0ff}video{width:100%;max-width:768px}'
        'video.obs{width:50%;max-width:384px}section{border-bottom:1px solid #555;padding:20px 0}'
        'td,th{padding:8px}svg{width:100%;max-width:900px;height:160px}'
        '.run-charts{display:grid;grid-template-columns:1fr 1fr;gap:20px}.run-charts>div{min-width:0}'
        '.achievement-checklist summary{cursor:pointer;list-style:none}.achievement-checklist summary:before{content:"> ";}'
        '.achievement-checklist[open] summary:before{content:"⌄ ";}'
        '@media(max-width:650px){.run-charts{grid-template-columns:1fr}}</style>'
        f'<h1>{"Archived exploration" if archived else "Phase 1 — retained explorers"}</h1><p><a href="/phase2/">Phase 2 — fresh comparison ↗</a></p><p>No step limit. Dreamer training ratio 256 where enabled. '
        'Model-free agents train the config 24 density explorer. Dreamer+Novelty and P2E train their world models and novelty-driven actors. '
        'FTN–M2E experiments replace only the policy decoder; the frozen reservoir and density explorer remain.</p>'
        f'<p><b>{len(trials)} {"archived runs" if archived else "live agents"}</b>. Ordered by method and GPU.</p>'
        '<p><label for="clip-length">Video length (all live agents): </label>'
        '<select id="clip-length"><option value="64">10.7 seconds</option><option value="128">21.3 seconds</option>'
        '<option value="192" selected>32 seconds</option><option value="256">42.7 seconds</option></select> '
        '<button id="clip-apply" type="button">Apply</button> <span id="clip-message" role="status"></span></p>'
        '<p>Current real trajectory: actual | reconstruction | first-half context, second-half open-loop predictions. '
        'Length changes take effect without restarting training, once enough frames from the current life are collected. '
        'Movies overwrite themselves every 1,000 observations or five minutes. '
        'Recent training data, not a held-out evaluation. Spatial cells are 4m bins, not RECODE centroids.</p>'
        f'<p>Updated {time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime())}</p>']
    parts.insert(1,comparison_html)
    phase=read(ROOT/'phase.json')
    banner=('<h1>Phase 0 — preliminary results (frozen archive)</h1><p>Training stopped; videos and curves are frozen snapshots. '
        '<a href="/continuous/">Current phase</a> · <a href="inventory.json">Archive inventory</a> · '
        '<a href="README.md">Documentation and resume instructions</a> · <a href="/">Earlier experiment galleries</a></p>' if archived else
        f'<h1>{html.escape(phase.get("title","Continuous exploration"))}</h1>'
        '<p><a href="/phase0/">Phase 0 archive</a></p>')
    parts.insert(1,banner)
    previous_group=None
    for t in trials:
        group='Plan2Explore (P2E)' if t.get('plan2explore') else 'FTN+M2L+Dreamer' if t.get('ftn') else 'MetaLearning2Explore (M2L)' if t.get('model_free_only') else 'Dreamer+Novelty' if t.get('dreamer_baseline') else ('Scaled runs' if t.get('scaled') else 'Baseline runs')
        if t.get('mf_init_weights'):group='M2E warm-start · high-resolution rendering'
        if t.get('m2e_ftn'):group='FTN–M2E policy decoders'
        if group!=previous_group:
            parts.append(f'<h2 style="margin-top:48px;padding-bottom:12px;border-bottom:3px solid #9bd0ff">{group}</h2>')
            previous_group=group
        run=STATE/'runs'/t['name']; live=run/'live';st=read(run/'status.json');p=read(live/'progress.json');v=read(live/'video.json')
        link=OUT/t['name']
        if live.exists() and not link.exists():link.symlink_to(live,target_is_directory=True)
        prefix=t['name'];step=st.get('step',0)
        variant='Scaled' if t.get('scaled') else 'Baseline'
        if t.get('reservoir_variant') == 'tuned_v1':variant += ' · tuned reservoir'
        if t.get('ftn'):variant=f'FTN+M2L+Dreamer · K={t.get("ftn_k",256)}'
        architecture=('Reservoir 128; RSSM hidden 1024, stochastic 128×32; image embedding 2048. '
            'Replay 2M; frontier 500k; RECODE 100k.' if t.get('scaled') else
            'Reservoir 64; RSSM hidden 256, stochastic 32×16; image embedding 1024. '
            'Replay 1M; frontier 100k; RECODE 10k.')
        if t.get('reservoir_variant') == 'tuned_v1':
            architecture += ' Tuned recurrence: 20% connections, radius 0.99, input gain 0.04, leak 1. Fresh replay and models.'
        if t.get('mf_replay_dtype')=='float16':
            architecture += ' Reservoir replay stored in float16; sampled features and DQN computation use float32.'
        architecture += f' Dreamer decoder base channels: {t.get("decoder_depth",16)}.'
        if t.get('dreamer_size')=='size12m':
            variant=f'FTN+M2L+Dreamer · K={t.get("ftn_k",256)}'
            architecture=('Original Dreamer size12m: RSSM hidden 256, stochastic 32×16; image embedding 1024; '
                'encoder and decoder base channels 16. Tuned reservoir 128; replay 2M; frontier 500k; RECODE 100k. '
                'Reservoir replay stored in float16; DQN computation uses float32.')
        if t.get('dreamer_baseline'):
            variant='Dreamer+Novelty'
            architecture='RSSM hidden 1024, stochastic 128×32; image embedding 2048; decoder base channels 64. Dreamer replay 100k. Dreamer chooses 100% of actions; novelty-only actor objective. No model-free explorer.'
        if t.get('model_free_only'):
            variant='MetaLearning2Explore (M2L)'
            architecture='Tuned reservoir 128; replay 2M; frontier 500k; RECODE 100k; float16 replay. Config 24 density learning, matching the FTN explorer. Model-feedback input fixed at zero; no Dreamer or FTN. Updates below are DQN updates.'
            if t.get('mf_init_weights'):
                variant='M2E warm-start · 256px render'
                architecture+=' Weights transferred from same-seed Phase 1 M2E; empty replay, frontier and RECODE, fresh optimizer; epsilon fixed at its configured minimum from the start. Human render 256×256 → area-downsampled 64×64 → 32×64 RGB reservoir input.'
        if t.get('plan2explore'):
            variant='Plan2Explore (P2E)'
            architecture='DreamerV3 world model and exploration actor/critic. Five independent embedding predictors; mean-variance intrinsic reward. No density explorer, FTN, or task-reward actor objective.'
        if t.get('m2e_ftn'):
            variant=comparison_label(t)
            spec=read(run/'architecture.json')
            architecture=('Frozen tuned reservoir 128; original 256×256 render → area-downsampled 64×64 → 32×64 RGB + feedback. '
                'One sheet / one Bayesian mask; independent eight-layer, width-eight neurons. '
                f'K={spec.get("k","…")}; bank={spec.get("neurons","…")}; active parameters={spec.get("active_parameters","…")}. '
                'Two reconfiguration steps per batch and every eight actions; mask-filtered replay. No Dreamer.')
            if t.get('frontier_ftn'):
                architecture=('Frozen tuned reservoir and downsampled 256×256 observations; matched convolutional FTN decoder. '
                    'K=296 of 2,368 independent eight-layer neurons. Goal identity retrieves a retained mask and posterior; '
                    'two-step offline reconfiguration with reaching targets and overlap replay. '
                    +('Direct frontier-reaching policies.' if t['frontier_ftn']=='frontier' else 'Directed observed-waypoint graph and short reusable reaching policies.'))
        weight_label='' if t.get('dreamer_baseline') or t.get('model_free_only') or t.get('plan2explore') else f' · model weight {t["weight"]}'
        parts.append(f'<section><h2>{variant} · GPU {t["gpu"]} · seed {t.get("seed",7)}{weight_label}</h2>'
            f'<p>{architecture}</p><p>{t["status"]}; '
            f'observations <b>{step:,}</b>; updates {st.get("updates",0):,}; process starts {t["attempts"]}. '
            f'Status age {int(time.time()-st.get("time",time.time()))} seconds.</p>')
        from .survival import summary as survival_summary,panel as survival_panel
        parts.append(survival_panel(survival_summary(run,st),individual_chart=True))
        if t.get('achievements'):
            parts.append(achievement_checklist(st.get('environment',{})))
        if st.get('ftn'):
            f=st['ftn']
            parts.append(f'<p>FTN: TD loss EMA {f.get("ftn_td_ema",0):.4f}; '
                f'mask turnover {f.get("ftn_mask_turnover",0):.1%}; '
                f'overlap replay updates {f.get("ftn_replay_updates",0):,}; '
                f'change points {f.get("ftn_changepoints",0):,}.</p>')
            if 'ftn_goals' in f:
                parts.append(f'<p>Retained goals {f["ftn_goals"]}; active goal {f.get("ftn_active_goal",0)}; '
                    f'goal retrievals {f.get("ftn_goal_retrievals",0)}; reached {f.get("ftn_goal_successes",0)}; '
                    f'timeouts {f.get("ftn_goal_timeouts",0)}; plans {f.get("ftn_plans",0)}.</p>')
        if p:
            parts.append(f'<p>Spatial cells {p["total_cells"]}; new in last window {p["new_cells"]}; '
                f'steps since discovery {p["steps_since_discovery"]}; stagnant: {p["stalled"]}; '
                f'first stagnation onset {p["first_stagnation"]}; longest stall {p["longest_stall"]} steps.</p>')
            if t.get('dreamer_baseline') or t.get('plan2explore'):
                parts.append(f'<p>Window means: raw disagreement {p["mean_model_novelty"]:.4f}; novelty reward {p["mean_model_reward"]:.4f}.</p>')
            else:parts.append(f'<p>Window means: density novelty {p["mean_density_novelty"]:.4f}; '
                f'raw disagreement {p["mean_model_novelty"]:.4f}; RNN input {p["mean_model_input"]:.4f}; '
                f'model reward {p["mean_model_reward"]:.4f}; weighted model reward {p["mean_weighted_model_novelty"]:.4f}. '
                f'Reward clipping {p["model_reward_clip_fraction"]:.1%}; input saturation {p["model_input_saturation_fraction"]:.1%}.</p>')
            rows=[]
            try:rows=[json.loads(x) for x in (live/'history.jsonl').read_text().splitlines()]
            except (OSError,ValueError):pass
            if rows:
                parts.append(run_charts(rows))
        if v and t.get('model_free_only'):
            stamp=v['step']
            parts.append(f'<p>Current observations {v["frame_start"]:,}–{v["frame_end"]:,}; clip {v["playback_seconds"]:.1f} seconds.</p>')
            if v.get('m2e_ftn'):
                goal_caption=(' | offline frontier (density view)' if v.get('offline_goal') else
                    ' | goal observation (waypoint / frontier)' if v.get('goal_panels') else '')
                parts.append(f'<p>Original observation | concurrent FTN action mask (EMA; spatially consolidated){goal_caption}</p>'
                    f'<video style="max-width:768px" controls playsinline preload="metadata" src="{prefix}/obs.mp4?v={stamp}"></video>')
                from .mask_diagnostics import publish as publish_masks
                try:publish_masks(live,v,f'{variant} · seed {t.get("seed")}',st)
                except Exception as exc:print('Mask diagnostic error',t['name'],repr(exc),flush=True)
                parts.append(f'<p><a target="_blank" rel="noopener" href="{prefix}/mask.html">Open mask diagnostic ↗</a></p>'
                    f'<details><summary>Mask variability diagnostic</summary><video style="max-width:384px" controls playsinline '
                    f'preload="none" src="{prefix}/bw.mp4?v={stamp}"></video></details>')
            elif v.get('exact_inputs'):
                from .split_human_video import publish
                publish(live,v)
                parts.append(f'<p>Human view · original 256×256 render</p><video class="obs" controls playsinline preload="metadata" src="{prefix}/human.mp4?v={stamp}"></video>'
                    f'<p>Agent input · RGB and feedback</p><video class="obs" controls playsinline preload="metadata" src="{prefix}/agent_input.mp4?v={stamp}"></video>'
                    f'<p>Both videos cover the same frames. Agent input: 32×64 RGB above the projected feedback channel, stacked into a 64×64 display tile. The actual reservoir tensor is 4×32×64, concatenated by channel. Feedback grayscale uses the labeled symmetric clip-wide range. <a href="{prefix}/inputs.npz">Exact float inputs and feedback vectors for this clip</a>.</p>')
                if v.get('offline_goal'):
                    parts.append(f'<p>Offline frontier · density observation used for prioritized training</p>'
                        f'<video class="obs" controls playsinline preload="metadata" src="{prefix}/offline_goal.mp4?v={stamp}"></video>')
            else:
                parts.append(f'<video class="obs" controls playsinline preload="metadata" src="{prefix}/obs.mp4?v={stamp}"></video>')
        elif v:
            panel_title='Actual | reconstruction | context / prediction | live mask' if t.get('ftn') else 'Actual | reconstruction | context / prediction'
            video_width=1024 if t.get('ftn') else 768
            stamp=v['step'];parts.append(f'<p>Video model step <b>{stamp:,}</b>; actual frames '
                f'{v["frame_start"]:,}–{v["frame_end"]:,}; age at rendering {v["frame_age"]} observations. '
                f'Clip {v["playback_seconds"]:.1f} seconds ({v["frames"]} frames; {v["context"]} context + {v["prediction_frames"]} prediction). '
                f'Reconstruction MSE {v["reconstruction_mse"]:.5f}; forecast MSE {v["forecast_mse"]:.5f}.</p>'
                f'<p>Actual observations</p><video class="obs" controls playsinline preload="metadata" src="{prefix}/obs.mp4?v={stamp}"></video>'
                f'<p>{panel_title}</p><video style="max-width:{video_width}px" controls playsinline preload="metadata" src="{prefix}/recent.mp4?v={stamp}&panels={len(v.get("panels",[]))}"></video>')
        else:parts.append('<p>Preparing first current-trajectory movie.</p>')
        if t.get('ftn'):
            ftn=read(live/'ftn.json')
            parts.append(f'<h3>BayesLoRA mask activity</h3><p>{t.get("ftn_side",48)}×{t.get("ftn_side",48)} atoms; {t.get("ftn_k",256)} active; rank 1; α={t.get("ftn_alpha",64)}. '
                'White: active now; purple/blue: fading activity. Spatial permutation groups co-active atoms '
                'for display only; it does not move weights or change overlaps. Unsupervised routes are not semantic labels.</p>')
            if ftn:
                parts.append('<p>The fourth video panel records the live action mask at each observation. '
                    'Reconstruction and prediction evaluate those observations using the current model and selected mask.</p>'
                    f'<details><summary>Training-mask activity diagnostic</summary><img width="384" height="384" style="max-width:100%;image-rendering:pixelated" '
                    f'src="{prefix}/bw.gif?v={ftn["updates"]}" alt="FTN training mask activity with EMA trails"></details>'
                    f'<p>Mask update {ftn["updates"]:,}; action reconfigurations {ftn["action_reconfigs"]:,}; '
                    f'action change points {ftn["action_changes"]}; archived routes {ftn["routes"]}; '
                    f'route hits {ftn["route_hits"]}; overlap replay updates {ftn["replay_updates"]:,}.</p>')
            else:parts.append('<p>Preparing mask activity visualization.</p>')
        links=[f'<a href="{prefix}/{file}">{label}</a>' for file,label in (
            ('history.jsonl','Full progress history'),('events.jsonl','Stagnation / recovery events'),
            ('video.json','Video provenance')) if (live/file).exists()]
        parts.append('<p>'+' · '.join(links)+'</p></section>')
    parts.append('<p>Stagnation means no new spatial cell for 5,000 observations; it is a diagnostic, not a stopping rule or proof of semantic stagnation.</p>'
        '''<script>
const clipSelect=document.getElementById('clip-length'), clipMessage=document.getElementById('clip-message'), clipApply=document.getElementById('clip-apply');
// Shared controls for spatial and achievement curves; survive page refreshes.
let hiddenRuns={};
try{hiddenRuns=JSON.parse(localStorage.getItem('globo-hidden-runs-v1')||'{}')||{}}catch(e){}
function applyRunVisibility(){
 document.querySelectorAll('path[data-run]').forEach(line=>{
  line.style.display=hiddenRuns[line.dataset.run]?'none':'';
 });
 document.querySelectorAll('.run-toggle').forEach(box=>{box.checked=!hiddenRuns[box.dataset.run]});
}
document.querySelectorAll('.run-toggle').forEach(box=>box.addEventListener('change',()=>{
 hiddenRuns[box.dataset.run]=!box.checked;
 try{localStorage.setItem('globo-hidden-runs-v1',JSON.stringify(hiddenRuns))}catch(e){}
 applyRunVisibility();
}));
applyRunVisibility();
let clipEditing=false;
clipSelect.addEventListener('change',()=>{clipEditing=true});
fetch('/api/clip-settings',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Control unavailable');return r.json()}).then(s=>{
 if(!clipEditing)clipSelect.value=s.frames;
}).catch(e=>{clipMessage.textContent=e.message});
clipApply.addEventListener('click',async()=>{
 clipApply.disabled=true;clipEditing=true;
 try{
  const response=await fetch('/api/clip-settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({frames:Number(clipSelect.value)})});
  const data=await response.json();if(!response.ok)throw Error(data.error||'Could not update');
  clipMessage.textContent=`Saved: ${data.seconds.toFixed(1)} seconds. Applies to new clips as frames become available.`;clipEditing=false;
 }catch(e){clipMessage.textContent=e.message}finally{clipApply.disabled=false}
});
setInterval(()=>{if(!clipEditing&&document.activeElement!==clipSelect&&[...document.querySelectorAll('video')].every(v=>v.paused))location.reload()},60000);
</script>''')
    tmp=OUT/'index.tmp';tmp.write_text(''.join(parts));tmp.replace(OUT/'index.html')
    if archived:
        page=(OUT/'index.html').read_text().replace('No step limit.', 'Archived unlimited runs.')
        page=page.replace('<select id="clip-length">','<select id="clip-length" disabled>')
        page=page.replace('<button id="clip-apply" type="button">','<button id="clip-apply" type="button" disabled>')
        page=page.replace('The shared axis includes current FTN and model-free runs plus the checkpointed Dreamer baseline;',
            'The shared axis includes archived FTN and model-free runs plus the checkpointed Dreamer baseline;')
        (OUT/'index.html').write_text(page)


def check_progress(t, now):
    """Recover live-but-hung workers; allow startup and checkpoint time."""
    run=STATE/'runs'/t['name']
    status=read(run/'status.json');launch=read(run/'launch.json')
    heartbeat=max(status.get('time',0),launch.get('started',0))
    if not heartbeat or now-heartbeat<1800:return
    if not alive(t):return
    if not t.get('watchdog_term_at'):
        action='terminate';sig=signal.SIGTERM;t['watchdog_term_at']=now
    elif now-t['watchdog_term_at']>=300 and not t.get('watchdog_kill_at'):
        action='force_stop';sig=signal.SIGKILL;t['watchdog_kill_at']=now
    else:return
    event=dict(time=now,run=t['name'],pid=t['pid'],event='stale_progress',
        action=action,step=status.get('step'),heartbeat_age=now-heartbeat)
    with (ROOT/'watchdog.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    print('Watchdog',json.dumps(event),flush=True)
    try:os.kill(t['pid'],sig)
    except ProcessLookupError:pass


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/'supervisor.pid').write_text(str(os.getpid()))
    path=ROOT/'state.json'
    # An explicit run list is authoritative: never silently fill free GPUs.
    if not path.exists():
        raise RuntimeError('Create continuous_runs/state.json with the intended run list first')
    trials=json.loads(path.read_text())
    stop=[False];children={}
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    try:
        while not stop[0]:
            for t in trials:
                from .continuous_monitor import bound_logs
                bound_logs(STATE/'runs'/t['name'])
                if t['name'] in children:children[t['name']].poll()
                if t['status']=='running':check_progress(t,time.time())
                if t['status']=='running' and not alive(t):
                    t.update(status='pending',retry_after=time.time()+60)
                if t['status']=='pending' and time.time()>=t.get('retry_after',0):
                    run=STATE/'runs'/t['name'];run.mkdir(parents=True,exist_ok=True)
                    cmd=[sys.executable,'-u','-m','globo_dreamer.train','--name',t['name'],'--seed',str(t.get('seed',7)),
                        '--spawn',t.get('spawn','12,5.1,120'),'--steps','0',
                        '--train-ratio','256',
                        '--continuous-monitor','--checkpoint-seconds','600',
                        '--dreamer-replay-capacity','100000']
                    if t.get('agent_seed') is not None:
                        cmd[cmd.index('--seed')+1]=str(t['agent_seed'])
                        cmd.extend(['--world-seed',str(t['seed'])])
                    if t.get('achievements'):cmd.append('--achievements')
                    if t.get('render_size'):cmd.extend(['--render-size',str(t['render_size'])])
                    if t.get('mf_init_weights'):cmd.extend(['--mf-init-weights',t['mf_init_weights']])
                    if t.get('m2e_ftn'):cmd.extend(['--m2e-ftn',t['m2e_ftn']])
                    if t.get('reduced_actions'):cmd.append('--reduced-actions')
                    if t.get('inventory_actions'):cmd.append('--inventory-actions')
                    if t.get('m2e_ftn_k'):cmd.extend(['--m2e-ftn-k',str(t['m2e_ftn_k'])])
                    if t.get('frontier_ftn'):cmd.extend(['--frontier-ftn',t['frontier_ftn']])
                    if t.get('plan2explore'):
                        cmd.extend(['--plan2explore','--p2e-reward-scale',str(t.get('p2e_reward_scale',10000.))])
                    elif t.get('dreamer_baseline'):
                        cmd.extend(['--dreamer-novelty-only','--density','0','--disagreement','1',
                            '--prediction-error','0','--no-feedback','--dreamer-action-probability','1'])
                    else:cmd.extend(['--model-free',
                        *(['--model-free-only'] if t.get('model_free_only') else ['--world-model-only']),
                        '--model-feedback','--model-reward-weight',str(t['weight']),
                        '--mf-replay-capacity','2000000' if t.get('scaled') else '1000000',
                        '--mf-goal-capacity','500000' if t.get('scaled') else '100000',
                        '--recode-capacity','100000' if t.get('scaled') else '10000',
                        ])
                    if t.get('scaled'):
                        cmd.extend(['--reservoir-channels','128'])
                        if not t.get('dreamer_size') and not t.get('model_free_only'):
                            cmd.extend(['--rssm-hidden','1024','--rssm-stoch','128','--rssm-classes','32','--encoder-depth','32'])
                    if t.get('dreamer_size'):cmd.extend(['--size',t['dreamer_size']])
                    if t.get('reservoir_variant'):
                        cmd.extend(['--reservoir-variant',t['reservoir_variant']])
                    if t.get('decoder_depth'):
                        cmd.extend(['--decoder-depth',str(t['decoder_depth'])])
                    if t.get('ftn'):cmd.append('--ftn')
                    if not t.get('dreamer_baseline') and not t.get('plan2explore'):
                        settings=dict(FIXED,dreamer_action_probability=0.)
                        for k in ('mf_gamma','mf_offline_updates','mf_lr'):
                            if k in t:settings[k]=t[k]
                        for k,v in settings.items():cmd.extend(['--'+k.replace('_','-'),str(v)])
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(t['gpu']),OMP_NUM_THREADS='4',LP_NUM_THREADS='2',
                        PYTHONDONTWRITEBYTECODE='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                        GLOBO_ENGINE_ROOT=f'/tmp/globo_continuous_engines/{t["name"]}')
                    if t.get('ftn'):
                        env.update(GLOBO_FTN_SIDE=str(t.get('ftn_side',48)),
                            GLOBO_FTN_K=str(t.get('ftn_k',256)),GLOBO_FTN_ALPHA=str(t.get('ftn_alpha',64)))
                    env['GLOBO_MF_REPLAY_DTYPE']=t.get('mf_replay_dtype','float32')
                    if t.get('software_renderer'):env['GLOBO_SOFTWARE_RENDERER']='1'
                    if t.get('visitation'):env['GLOBO_VISITATION']='1'
                    if t.get('event_capture'):env['GLOBO_EVENT_CAPTURE']='1'
                    if t.get('life_capture'):
                        env['GLOBO_LIFE_CAPTURE']='1'
                        env['GLOBO_LIFE_VIDEO_ROOT']=str(Path(t.get('artifact_root','/mnt/obelisk-shared/kmckee90/globo_lifetime_v2'))/'runs'/t['name']/'life_videos')
                    if t.get('hard_survival'):env['GLOBO_HARD_SURVIVAL']='1'
                    if t.get('natural_cycle'):env['GLOBO_NATURAL_CYCLE']='1'
                    if 'move_speed' in t:env['GLOBO_MOVE_SPEED']=str(t['move_speed'])
                    if 'move_accel' in t:env['GLOBO_MOVE_ACCEL']=str(t['move_accel'])
                    if t.get('life_settings'):env['GLOBO_LIFE_SETTINGS']=t['life_settings']
                    if t.get('survival'):
                        env['GLOBO_SURVIVAL']='1'
                        env['GLOBO_SURVIVAL_NOVELTY_SCALE']=str(t.get('survival_novelty_scale',.25))
                    if t.get('single_place'):env['GLOBO_SINGLE_PLACE']='1'
                    if t.get('lifetime_v2'):env['GLOBO_LIFETIME_V2']='1'
                    if t.get('block_stack_capacity'):env['GLOBO_BLOCK_STACK_CAPACITY']=str(t['block_stack_capacity'])
                    if t.get('collected_stack_capacity'):env['GLOBO_COLLECTED_STACK_CAPACITY']=str(t['collected_stack_capacity'])
                    if t.get('frontier_rotation') is False:env['GLOBO_FRONTIER_ROTATION']='0'
                    if t.get('phase')=='phase2':env['GLOBO_CLIP_SETTINGS']='/mnt/obelisk-shared/kmckee90/globo_phase2/clip_settings.json'
                    if t.get('phase')=='phase3':env['GLOBO_CLIP_SETTINGS']='/mnt/obelisk-shared/kmckee90/globo_phase3/clip_settings.json'
                    if t.get('phase')=='phase4':env['GLOBO_CLIP_SETTINGS']='/mnt/obelisk-shared/kmckee90/globo_phase4/clip_settings.json'
                    if t.get('phase')=='phase5':env['GLOBO_CLIP_SETTINGS']='/mnt/obelisk-shared/kmckee90/globo_phase5/clip_settings.json'
                    if t.get('clip_settings'):env['GLOBO_CLIP_SETTINGS']=t['clip_settings']
                    cpus=sorted(os.sched_getaffinity(0))[t['gpu']*4:t['gpu']*4+4]
                    if len(cpus)==4:cmd=['taskset','-c',','.join(map(str,cpus)),*cmd]
                    with (run/'console.log').open('a') as f:
                        child=subprocess.Popen(cmd,env=env,cwd=STATE,stdout=f,stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL,start_new_session=True)
                    children[t['name']]=child;t.update(pid=child.pid,status='running',attempts=t['attempts']+1)
                    t.pop('watchdog_term_at',None);t.pop('watchdog_kill_at',None)
                    atomic(path,trials)
            atomic(path,trials)
            try:
                if not any(t.get('phase')=='phase3' for t in trials):gallery(trials)
            except Exception as exc:print('Gallery error',repr(exc),flush=True)
            time.sleep(20)
    finally:
        for t in trials:
            if t['status']=='running' and alive(t):os.kill(t['pid'],signal.SIGTERM)
        atomic(path,trials)


if __name__=='__main__':main()
