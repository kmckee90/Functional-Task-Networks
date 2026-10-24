"""One chronological navigation chain, injected without changing archived reports."""
import html
import re
from pathlib import Path

ROOT=Path('/mnt/obelisk-shared/kmckee90')
EXPERIMENTS=[
 ('phase0','Phase 0 · preliminary results',ROOT/'globo_archives/phase0/monitor'),
 ('phase1','Phase 1 · archived exploration',ROOT/'globo_archives/phase1/monitor'),
 ('phase2','Phase 2 · long-run comparison',ROOT/'globo_phase2/monitor'),
 ('phase3','Phase 3 · multi-seed comparison',ROOT/'globo_phase3/monitor'),
 ('phase4','Phase 4 · M2E and FTN-M2E',ROOT/'globo_phase4/monitor'),
 ('phase5','Phase 5 · Plan2Explore',ROOT/'globo_phase5/monitor'),
 ('actions','Reduced actions · 14-action comparison',ROOT/'globo_actions/monitor'),
 ('inputs','Corrected inputs · checkpoint continuations',ROOT/'globo_input_v2/monitor'),
 ('engine','Corrected engine · fresh runs',ROOT/'globo_engine_v2/monitor'),
 ('lifetime','Continuing lives · discount and update comparison',ROOT/'globo_lifetime_v2/monitor'),
]

def prepare(out):
    folder=out/'experiments';folder.mkdir(exist_ok=True)
    for key,_,path in EXPERIMENTS:
        if key=='lifetime':continue
        link=folder/key
        if not link.is_symlink() and not link.exists():link.symlink_to(path,target_is_directory=True)
        # Historical reports use these absolute URLs for phase links and media.
        if key.startswith('phase'):
            alias=out/key
            if not alias.exists() and not alias.is_symlink():alias.symlink_to(path,target_is_directory=True)

def url(key):
    return '/' if key=='lifetime' else '/experiments/'+key+'/'

def decorate(path,body):
    resolved=path.resolve()
    index=next((i for i,(_,_,p) in enumerate(EXPERIMENTS)
                if resolved==(p/'index.html').resolve()),None)
    if index is None:return body
    def link(i,label):
        if not 0<=i<len(EXPERIMENTS):
            return f'<span aria-disabled="true" style="opacity:.45">{label}</span>'
        key,title,_=EXPERIMENTS[i]
        return f'<a href="{url(key)}" title="{html.escape(title)}">{label}</a>'
    nav='<nav aria-label="Experiment navigation" style="display:flex;flex-wrap:wrap;align-items:center;gap:20px;padding:16px 0;margin-bottom:20px;border-bottom:1px solid #435064;font:16px system-ui">'
    nav+=link(index-1,'← Previous experiment')
    nav+=f'<strong style="flex:1">{html.escape(EXPERIMENTS[index][1])}</strong>'
    nav+=link(index+1,'Next experiment →')+'</nav>'
    # Some early reports omit explicit body tags.
    match=re.search(r'<body\b[^>]*>',body,re.I)
    if match:return body[:match.end()]+nav+body[match.end():]
    match=re.search(r'</style>',body,re.I)
    pos=match.end() if match else 0
    return body[:pos]+nav+body[pos:]
