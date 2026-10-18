"""Snapshot stopped phase results/checkpoints on shared storage; keep originals."""
import argparse,hashlib,json,os,shutil,signal,time
from pathlib import Path
from .phase2 import STATE,read,atomic,PUBLISH_FILES

ARCHIVE=Path('/mnt/obelisk-shared/kmckee90/globo_archives')

def main(pod):
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try: args=p.read_bytes().split(b'\0')
        except OSError: continue
        if b'globo_dreamer.phase4' in args or b'globo_dreamer.phase4_monitor' in args:
            os.kill(int(p.parent.name),signal.SIGTERM)
    time.sleep(1)
    trials=read(STATE/'continuous_runs/state.json',[])
    for phase in ('phase3','phase4'):
        dest=ARCHIVE/phase/pod;dest.mkdir(parents=True,exist_ok=True)
        selected=[t for t in trials if t.get('phase')==phase]
        shared=Path(f'/mnt/obelisk-shared/kmckee90/globo_{phase}')
        for t in selected:
            run=STATE/'runs'/t['name'];target=dest/'runs'/t['name']
            proc=Path(f'/proc/{t.get("pid")}/cmdline')
            assert not proc.exists() or not proc.read_bytes(),t['name']+' still alive'
            assert read(run/'status.json').get('running') is False,t['name']
            latest=(run/'ckpt/latest').read_text().strip()
            assert (run/'ckpt'/latest/'done').exists(),t['name']+' checkpoint incomplete'
            for rel in (*PUBLISH_FILES,'console.log'):
                src=run/rel
                if not src.is_file():continue
                dst=target/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
                # Refresh frozen monitor data with the final stopped status.
                if rel in PUBLISH_FILES:
                    dst=shared/'runs'/t['name']/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            for name in ('ckpt','event_clips'):
                if (run/name).exists():shutil.copytree(run/name,target/name,dirs_exist_ok=True)
            atomic(target/'preservation.json',dict(original_run=str(run),pod=pod,
                checkpoint=latest,step=read(run/'status.json')['step'],
                replay='Retained at original_run; not duplicated in this results/checkpoint snapshot.',time=time.time()))
            print('Saved',phase,pod,t['name'],flush=True)
        atomic(dest/'trials.json',selected)
        atomic(shared/'pods'/f'{pod}.json',dict(time=time.time(),pod=pod,trials=selected))
        source=dest/'source_at_archive';source.mkdir(exist_ok=True)
        for p in Path(__file__).parent.iterdir():
            if p.is_file() and p.suffix in ('.py','.lua','.js','.md','.yaml'):
                shutil.copy2(p,source/p.name)
        manifest=[]
        for p in sorted(dest.rglob('*')):
            if p.is_file() and p.name!='manifest.json':
                h=hashlib.sha256()
                with p.open('rb') as f:
                    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
                manifest.append(dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=h.hexdigest()))
        atomic(dest/'manifest.json',dict(time=time.time(),files=manifest))
        print('Verified snapshot',phase,pod,len(manifest),'files',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pod',required=True);main(p.parse_args().pod)
