"""Dedicated CPU-only, playerless original-terrain servers for two seeds."""
import os
import fcntl
import signal
from pathlib import Path
import shutil
import subprocess
import time
from .phase2 import STATE,SHARED,atomic

def main():
    children=[];commands={};stopping=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True))
    root=STATE/'terrain_reference';root.mkdir(parents=True,exist_ok=True)
    lock=(root/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    games=root/'games';games.mkdir(exist_ok=True)
    link=games/'globo'
    if not link.exists():link.symlink_to(Path(__file__).parent/'legacy/globo_game',target_is_directory=True)
    binary=STATE/'venv/lib/python3.11/site-packages/craftium/luanti/bin/luanti'
    for seed in (19,101):
        world=root/f'seed{seed}';mod=world/'worldmods/terrain_probe';mod.mkdir(parents=True,exist_ok=True)
        (mod/'mod.conf').write_text('name = terrain_probe\ndepends = main\n')
        shutil.copyfile(Path(__file__).with_name('terrain_probe.lua'),mod/'init.lua')
        (world/'world.mt').write_text('gameid = globo\nbackend = sqlite3\nplayer_backend = sqlite3\nauth_backend = sqlite3\nload_mod_terrain_probe = true\n')
        out=SHARED/'terrain'/str(seed);out.mkdir(parents=True,exist_ok=True)
        config=world/'mapper.conf'
        config.write_text(f'fixed_map_seed = {seed}\nport = {31000+seed}\nbind_address = 127.0.0.1\nserver_announce = false\nterrain_output = {out}\nnum_emerge_threads = 1\nactive_block_range = 0\nsecure.enable_security = false\nsync_env_mode = false\n')
        env=dict(os.environ,MINETEST_GAME_PATH=str(games),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
        with (world/'console.log').open('a') as f:
            child=subprocess.Popen([str(binary),'--server','--gameid','globo','--world',str(world),'--config',str(config)],
                cwd=world,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
        commands[seed]=(world,env,list(child.args))
        children.append((seed,child));print('Terrain server',seed,child.pid,flush=True)
    atomic(root/'servers.json',{str(seed):p.pid for seed,p in children})
    while not stopping[0]:
        for i,(seed,p) in enumerate(children):
            if p.poll() is not None:
                print('Restarting terrain server',seed,p.returncode,flush=True)
                world,env,cmd=commands[seed]
                with (world/'console.log').open('a') as f:
                    p=subprocess.Popen(cmd,cwd=world,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
                children[i]=(seed,p)
                atomic(root/'servers.json',{str(seed):p.pid for seed,p in children})
        time.sleep(10)
    for _,p in children:
        if p.poll() is None:p.terminate()

if __name__=='__main__':main()
