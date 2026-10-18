"""Create a dated, read-only proposal snapshot of all Phase 2 runs on a pod."""
import json,shutil,html,datetime
from pathlib import Path
from .phase2 import SHARED,experiments,read
from .phase2_monitor import rows
from .achievement_catalog import LABELS

OUT=Path('/home/dev/proposal_state/phase2-results')
NAMES={'Dreamer':'Dreamer','Plan2Explore':'Plan2Explore','M2E':'M2E','M2E+convFTN':'M2E+FTN'}
COLORS={'Dreamer':'#b39ddb','Plan2Explore':'#ffb56b','M2E':'#72b7ff','M2E+convFTN':'#7de1bf'}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seeds':{}}
    for seed in (19,101):
        trials=[t for t in experiments() if t['seed']==seed]
        data={t['name']:rows(t) for t in trials}
        cap=min(rs[-1]['step'] for rs in data.values())
        entry={'comparison_through':cap,'runs':[]}
        for t in trials:
            rs=data[t['name']];last=rs[-1];matched=[r for r in rs if r['step']<=cap][-1]
            status=read(SHARED/'runs'/t['name']/'status.json');e=status.get('environment',{})
            entry['runs'].append(dict(name=t['name'],model=NAMES[t['model']],step=last['step'],cells=last['total_cells'],
                achievements=last['achievement_checked'],matched=matched,
                checklist=[LABELS.get(k,k) for k,v in e.get('achievement_counts',{}).items() if v>0]+list(e.get('achievement_discoveries',{}))))
            shutil.copyfile(SHARED/'monitor/maps'/f'{t["name"]}.png',OUT/f'{t["name"]}.png')
        entry['map']=read(SHARED/'monitor/maps'/f'seed{seed}.json')
        for metric,title,ylabel in [('total_cells','Spatial exploration','Distinct ground cells'),('achievement_checked','Unique achievements','Achievement boxes checked')]:
            series=[(t,[r for r in data[t['name']] if r['step']<=cap and (metric!='achievement_checked' or 'achievement_counts' in r)]) for t in trials]
            peak=max(1,max(r.get(metric,0) for _,rs in series for r in rs))*1.08
            p=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 440"><rect width="760" height="440" rx="12" fill="#111b27"/><style>text{font-family:DejaVu Sans,sans-serif;fill:#b5c5d7;font-size:14px}</style>',f'<text x="70" y="32" style="font-size:20px;fill:#e4edf5">{title} · world {seed}</text>']
            for i,t in enumerate(trials):
                x=70+i*168;c=COLORS[t['model']]
                p.append(f'<path d="M{x} 58h20" stroke="{c}" stroke-width="3"/><text x="{x+26}" y="63">{NAMES[t["model"]]}</text>')
            for i in range(5):
                y=355-i*260/4;x=70+i*655/4
                p.append(f'<path d="M70 {y}H725" stroke="#304052"/><text x="60" y="{y+5}" text-anchor="end">{peak*i/4:.0f}</text><text x="{x}" y="380" text-anchor="middle">{cap*i/4000:.0f}k</text>')
            for t,rs in series:
                path=''
                for r in rs:
                    x=70+655*r['step']/cap;y=355-260*r.get(metric,0)/peak
                    path+=f'M{x:.2f},{y:.2f}' if not path else f'H{x:.2f}V{y:.2f}'
                p.append(f'<path d="{path}" stroke="{COLORS[t["model"]]}" stroke-width="3" fill="none"/>')
            p.append(f'<text x="400" y="416" text-anchor="middle">Environment observations</text><text transform="translate(20 225) rotate(-90)" text-anchor="middle">{ylabel}</text></svg>')
            (OUT/f'seed{seed}-{metric}.svg').write_text(''.join(p))
        report['seeds'][str(seed)]=entry
    (OUT/'snapshot.json').write_text(json.dumps(report,indent=2))
    for seed,e in report['seeds'].items():
        print(seed,'matched through',e['comparison_through'])
        for r in e['runs']:print(r['model'],r['step'],r['cells'],r['achievements'],'matched',r['matched']['total_cells'],r['matched']['achievement_checked'])

if __name__=='__main__':main()
