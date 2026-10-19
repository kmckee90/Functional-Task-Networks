"""Read-only snapshot of seed-19 monitor data; standalone proposal figure."""
import json
from pathlib import Path
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

STATE=Path('/home/dev/globo_dreamer_state')
OUT=STATE/'reviews/config24/proposal_figures'


def main():
    import sys
    saved=json.loads((OUT/'globo-ftn-seed19.json').read_text()) if '--reuse-snapshot' in sys.argv else None
    OUT.mkdir(exist_ok=True)
    runs=[('m2eftn_flat_s19','FTN+Metalearn2Explore','#7de1bf'),
          ('phase1_m2e_s19','Metalearn2Explore','#c7a1fa'),
          ('phase1_p2e_s19','Plan2Explore','#70cfff'),
          ('phase1_dreamer_s19','Dreamer','#f5bd63')]
    statuses={name:json.loads((STATE/'runs'/name/'status.json').read_text()) for name,_,_ in runs}
    end=statuses[runs[0][0]]['step']
    if saved:end=saved['observations']
    snapshot=dict(seed=19,observations=end,captured=datetime.now(timezone.utc).isoformat(),runs=[])
    plt.rcParams.update({'figure.facecolor':'#111820','axes.facecolor':'#111820',
        'text.color':'#e4edf5','axes.labelcolor':'#bacbdd','xtick.color':'#bacbdd',
        'ytick.color':'#bacbdd','axes.edgecolor':'#465463','font.size':12,
        'savefig.facecolor':'#111820','font.family':'DejaVu Sans'})
    fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
    for name,label,color in runs:
        root=STATE/'runs'/name;points={0:0};latest=0
        for line in (root/'live/history.jsonl').read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            latest=max(latest,row['step'])
            if row['step']<=end:points[row['step']]=row['total_cells']
        status=statuses[name];latest=max(latest,status['step'])
        if status['step']<=end:points[status['step']]=status['environment']['cells_lifetime']
        # Carry the last observed cumulative count to the cutoff only when the
        # recorded run actually extends to/beyond it. Never extrapolate a run.
        if latest>=end:points[end]=points[max(points)]
        values=sorted(points.items());x,y=zip(*values)
        if saved:
            values=next(r['points'] for r in saved['runs'] if r['name']==name)
            x,y=zip(*values)
        ax.step(x,y,where='post',label=label,color=color,lw=3 if name=='m2eftn_flat_s19' else 2.3)
        label_x=min(end*.96,x[-1])
        label_y=next(count for obs,count in reversed(values) if obs<=label_x)
        ax.annotate(label,(label_x,label_y),xytext=(0,7),textcoords='offset points',
            ha='right',va='bottom',color=color,fontsize=12,fontweight='bold',
            bbox=dict(facecolor='#111820',edgecolor='none',alpha=.85,pad=1.5))
        snapshot['runs'].append(dict(name=name,label=label,points=values))
    ax.set(xlim=(0,end),ylim=(0,None),xlabel='Observations',ylabel='Cumulative spatial cells discovered')
    ax.set_title('Exploring Globo',loc='left',pad=18,fontweight='bold')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda n,_:f'{n/1000:g}k'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda n,_:f'{n:,.0f}'))
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#34404d',alpha=.7);ax.set_axisbelow(True)
    for extension in ('png','svg'):fig.savefig(OUT/f'globo-ftn-seed19.{extension}',dpi=180,bbox_inches='tight',pad_inches=.22)
    plt.close(fig)
    if saved:
        labels={name:label for name,label,_ in runs}
        for run in saved['runs']:run['label']=labels[run['name']]
    (OUT/'globo-ftn-seed19.json').write_text(json.dumps(saved or snapshot))
    print(json.dumps(dict(observations=end,final_cells={r['label']:r['points'][-1][1] for r in snapshot['runs']})))


if __name__=='__main__':main()
