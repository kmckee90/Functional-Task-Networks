"""Read-only bindings to the four existing experiment families."""
import json
from pathlib import Path

import torch

ASSETS = Path('/home/dev/sit_state')
ROOT = ASSETS / 'ic4_recall'
ART = Path(__file__).parent / 'artifacts'
SUITES = {
    'filters': dict(data=ASSETS, experiment=ASSETS,
                    names=['monochrome', 'poster', 'sketch'], H=64, k=16, alpha=4.),
    'lfw': dict(data=ASSETS/'identity_lfw', experiment=ASSETS/'identity_lfw',
                names=['George_W_Bush', 'Colin_Powell', 'Tony_Blair'], H=64, k=16, alpha=4.),
    'distinct': dict(data=ASSETS/'identity_distinct', experiment=ASSETS/'identity_distinct_1024_k256',
                     names=['Whoopi_Goldberg', 'Bruce_Willis', 'Nicole_Kidman'], H=1024, k=256, alpha=64.),
    'mixed': dict(data=ASSETS/'mixed_objects', experiment=ASSETS/'mixed_objects',
                  names=['Whoopi_Goldberg', 'Hoatzin', 'BMW_i8'], H=1024, k=256, alpha=64.),
}


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def run_path(suite, arm, seed):
    if suite == 'filters':
        name = f'bayes_reactive_low_sd3_s{seed}' if arm == 'noreplay' else f'bayes_replay_s{seed}'
    else:
        name = f'{"noreplay" if arm == "noreplay" else "representative"}_s{seed}'
    return SUITES[suite]['experiment']/'runs'/name


def load_data(suite, splits=('train', 'support')):
    cfg = SUITES[suite]
    return {n: {s: torch.load(cfg['data']/f'{n}_{s}.pt', map_location='cpu', weights_only=True)
                for s in splits} for n in cfg['names']}


def image_rows(suite, domain, split):
    cfg = SUITES[suite]
    if suite == 'filters':
        from ..prepare import styled
        from PIL import Image
        manifest = json.loads((cfg['data']/'data_manifest.json').read_text())
        return [styled(Image.open(path), domain) for path, _ in manifest['splits'][split]]
    from PIL import Image
    manifest = json.loads((cfg['data']/'manifest.json').read_text())
    return [Image.open(cfg['data']/r['image']).convert('RGB')
            for r in manifest['splits'][domain][split]]


def build_model(suite, seed=123):
    from ..model import BayesSiT
    cfg = SUITES[suite]
    torch.manual_seed(seed)
    return BayesSiT(ASSETS/'SiT-XL-2-256.pt', H=cfg['H'], k=cfg['k'], rank=1,
                   alpha=cfg['alpha'], reactive_low=True).cuda().eval()
