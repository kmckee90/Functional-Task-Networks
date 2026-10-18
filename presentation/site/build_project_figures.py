"""Render proposal figures on the base pod. Inputs are preserved; outputs overwrite.

Runtime: /home/dev/proposal_state/venv/bin/python build_project_figures.py
Globo input is a snapshot of the six original Phase 1 history.jsonl files,
exported as method/seed/points into media/globo-comparison.json.
"""
import json
from pathlib import Path
import pymupdf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image, ImageDraw, ImageFont

STATE = Path('/home/dev/proposal_state')
OUT = STATE / 'media'
ASSETS = Path(__file__).resolve().parent.parent / 'proposal assets/results'
plt.rcParams.update({'figure.facecolor':'#111820','axes.facecolor':'#111820',
    'text.color':'#e4edf5','axes.labelcolor':'#bacbdd','xtick.color':'#bacbdd',
    'ytick.color':'#bacbdd','axes.edgecolor':'#465463','font.size':11,
    'savefig.facecolor':'#111820','font.family':'DejaVu Sans'})

def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches='tight', pad_inches=.22)
    plt.close(fig)

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', color='#34404d', alpha=.7)
    ax.set_axisbelow(True)

def globo():
    snapshot = json.loads((OUT / 'globo-comparison.json').read_text())
    summaries=[]
    for seed in [19,101]:
        runs=[r for r in snapshot['runs'] if r['seed']==seed]
        assert {r['name'] for r in runs} == {f'phase1_{m}_s{seed}' for m in ['m2e','dreamer','p2e']}
        end=min(r['points'][-1][0] for r in runs)
        fig,ax=plt.subplots(figsize=(10,4.7),layout='constrained')
        final={}
        for method,label,color in [('m2e','M2E','#7de1bf'),('dreamer','Dreamer + novelty','#f5bd63'),('p2e','Plan2Explore','#70cfff')]:
            r=next(r for r in runs if r['method']==method)
            points=[p for p in r['points'] if p[0]<=end]
            x,y=zip(*points)
            ax.step(x,y,where='post',label=label,color=color,lw=2.7)
            final[method]=y[-1]
        ax.set(xlim=(0,end),ylim=(0,None),xlabel='Observations',ylabel='Distinct ground cells visited (4 m × 4 m)')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda n,_:f'{n/1000:g}k'))
        ax.set_title(f'World seed {seed} · equal observation budget',loc='left',pad=17,fontweight='bold')
        ax.legend(frameon=False,loc='upper left')
        clean(ax)
        save(fig,f'globo-seed-{seed}.png')
        summaries.append(dict(seed=seed,observations=end,cells=final))
    (OUT/'globo-comparison-summary.json').write_text(json.dumps(dict(captured=snapshot['captured'],results=summaries),indent=2))
    print('Globo matched comparisons:', summaries)

def maze():
    doc=pymupdf.open(STATE/'m2e.pdf')
    # Figure 1: the simpler discrete-action DQN architecture.
    doc[4].get_pixmap(matrix=pymupdf.Matrix(5,5),clip=pymupdf.Rect(144,94.04,468,241.78),alpha=False).save(OUT/'maze-dqn.png')
    # Figure 9 in Appendix A, the continuous-action DDPG architecture.
    doc[15].get_pixmap(matrix=pymupdf.Matrix(4,4),clip=pymupdf.Rect(90,359.98,522.01,585.60),alpha=False).save(OUT/'maze-ddpg.png')
    # Figure 2: crop each original vector panel, preserving the original curves/legend.
    for name,y0,y1 in [('fixed',96,225),('random',257,386),('continual',419,548),('noisy',568,698)]:
        doc[7].get_pixmap(matrix=pymupdf.Matrix(4,4),clip=pymupdf.Rect(144,y0,463,y1),alpha=False).save(OUT/f'maze-{name}.png')

def ftn():
    # Exact means from manuscripts/arxiv/main.tex, table tab:mnist_suite.
    names=['Sequential','Replay','SupSup','Progressive','Fixed masks','Mask retrieval','FTN']
    acc=[44.4,44.2,88.6,97.9,97.9,97.8,96.0]
    hs=[10.397,8.524,3.535,8.330,2.309,2.311,.562]
    colors=['#72849c']*6+['#7de1bf']
    for vals,title,filename in zip([acc,hs],['Retained accuracy (%) · higher is better','New-task starting loss · lower is better'],['ftn-accuracy.png','ftn-headstart.png']):
        fig,ax=plt.subplots(figsize=(6,4.3),layout='constrained')
        bars=ax.barh(names,vals,color=colors,height=.66)
        ax.invert_yaxis();ax.set_title(title,loc='left',pad=15,fontsize=12)
        ax.bar_label(bars,labels=[f'{v:g}' for v in vals],padding=5,color='#e4edf5',fontsize=10)
        ax.set_xlim(0,max(vals)*1.2);clean(ax);ax.grid(False)
        save(fig,filename)
    # Exact means from main.tex, table tab:lora_main (five seeds).
    fig,ax=plt.subplots(figsize=(8,3.7),layout='constrained')
    names=['Sequential LoRA','LoRA + replay','FTN-LoRA']
    values=[-47,-9,19]
    bars=ax.barh(names,values,color=['#a3a6b8','#a3a6b8','#7de1bf'],height=.55)
    ax.invert_yaxis();ax.axvline(0,color='#bacbdd',lw=1)
    ax.bar_label(bars,labels=['−47','−9','+19'],padding=6,color='#e4edf5')
    ax.set(xlim=(-57,29),xlabel='Backward transfer (percentage points)')
    ax.set_title('Later learning improves earlier abilities',loc='left',pad=16,fontweight='bold')
    clean(ax);ax.grid(False);save(fig,'ftn-backward-transfer.png')

def sit():
    # First four generated samples, in their original order; no quality selection.
    samples=[('monochrome','Monochrome'),('sketch','Sketch'),('Whoopi_Goldberg','A learned identity'),('Giant_anteater','Giant anteater'),('BMW_i8','BMW i8')]
    canvas=Image.new('RGB',(1024,len(samples)*310),'#111820')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.load_default(size=24)
    for i,(name,label) in enumerate(samples):
        draw.text((8,i*310+8),label,font=font,fill='#e4edf5')
        im=Image.open(ASSETS/f'{name}.png').convert('RGB')
        canvas.paste(im.crop((0,0,1024,256)),(0,i*310+45))
    canvas.save(OUT/'sit-retained-samples.jpg',quality=92)
    records=json.loads((ASSETS/'sit-source.json').read_text())
    result=records['replay_p05_s125'][-1]['targets']
    print('SiT 5% final metrics:',{n:v['metrics'] for n,v in result.items()})

if __name__=='__main__':
    globo();maze();ftn();sit()
