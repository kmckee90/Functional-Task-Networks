"""Independent completion and exact-data-overlap audit, executed only on the pod."""
import hashlib
import json
from pathlib import Path
import math


def main():
    root=Path('/home/dev/sit_state')
    status=json.loads((root/'supervisor_status.json').read_text())
    assert len(status)==8 and all(v['exit_code']==0 for v in status.values())
    manifest=json.loads((root/'data_manifest.json').read_text())
    seen={}
    overlaps=[]
    for split,rows in manifest['splits'].items():
        for filename,label in rows:
            digest=hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            if digest in seen and seen[digest][0]!=split:
                overlaps.append([seen[digest],[split,filename]])
            seen[digest]=(split,filename)
    quality_records=0
    for name in status:
        folder=root/'runs'/name
        records=[json.loads(x) for x in (folder/'train.jsonl').read_text().splitlines()]
        assert len(records)==4500
        assert all(sum(r['block']==b for r in records)==1500 for b in range(3))
        assert all(math.isfinite(r['loss']) and r['domain']!='photo' for r in records)
        lifecycle=json.loads((folder/'lifecycle.json').read_text())
        assert len(lifecycle)==4 and lifecycle[0]['stage']=='initial'
        for row in lifecycle:
            for domain,variants in row['domains'].items():
                for mode,v in variants.items():
                    if mode=='inference':
                        continue
                    assert len(v['per_example'])==200 and math.isfinite(v['loss'])
        quality=json.loads((folder/'quality.json').read_text())
        expected=20 if name in ['bayes_replay_s123','bayes_noreplay_s123'] else 4
        assert len(quality)==expected, (name,len(quality),expected)
        assert all(r['generated_count']==200 and r['reference_count']==400 and math.isfinite(r['kid']) for r in quality)
        quality_records+=len(quality)
    result={'workers_successful':8,'training_updates':36000,'domains_per_run':3,
            'quality_records':quality_records,'quality_generated_images':quality_records*200,
            'exact_file_content_cross_split_duplicates':overlaps,
            'split_counts':{k:len(v) for k,v in manifest['splits'].items()},
            'note':'Exact byte duplicates only; this does not prove absence of perceptual near-duplicates in upstream Imagenette.'}
    dest=Path('/home/dev/workspace/continual/artifacts/verification.json')
    dest.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    assert not overlaps, 'Exact cross-split image duplicates found; report must disclose.'


if __name__=='__main__':
    main()
