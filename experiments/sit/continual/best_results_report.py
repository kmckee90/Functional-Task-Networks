"""Build a portable, read-only exhibition of completed unified-direct4 results.

Run on the SiT pod: python -m continual.best_results_report
"""
import base64
import csv
import html
import json
from pathlib import Path

BASE = Path(__file__).parent
OUT = BASE / 'best_results_2026_10_06'


def read(path):
    return json.loads(path.read_text())


def image(path, alt):
    assert path.exists(), path
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img loading="lazy" alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def value(x, suite):
    return f'{x:.5f}' if suite == 'filters' else f'{x:.1%}'


def main():
    OUT.mkdir(exist_ok=True)
    original = read(BASE / 'ic4_recall/artifacts/summary_atoms.json')
    anteater = read(BASE / 'mixed_anteater/artifacts/summary.json')['unified_final']
    broad = read(BASE / 'mixed_anteater/artifacts/broad_label_audit.json')['unified']
    suites = {
        'filters': ('Image filters', original['filters'], 64, 16),
        'lfw': ('LFW faces', original['lfw'], 64, 16),
        'distinct': ('Distinct faces', original['distinct'], 1024, 256),
        'anteater': ('Mixed objects — giant anteater', anteater, 1024, 256),
    }
    selected = []; allrows = []; sections = []; overview = []
    for suite, (title, runs, atoms, active) in suites.items():
        assert len(runs) == 6
        names = list(next(iter(runs.values()))['summary'])
        scores = {tag: sum(v['mean'] for v in r['summary'].values()) / 3 for tag, r in runs.items()}
        best = min(scores, key=scores.get) if suite == 'filters' else max(scores, key=scores.get)
        section = [f'<section id="{suite}"><h2>{title}</h2>',
                   f'<p>Training order: {" → ".join(n.replace("_", " ") for n in names)}. '
                   f'Shared rank-one LoRA bank: {atoms} atoms, {active} active. '
                   + ('Clean Inception KID, lower is better; raw values, not multiplied by 1,000.' if suite == 'filters' else 'Independent evaluator acceptance, higher is better.') + '</p>']
        notes = {
            'filters': 'Visual limitation: some images in the best monochrome grid retain visible color. Low KID does not guarantee that every image follows the requested filter.',
            'lfw': 'Replay is materially stronger here. Even the best no-replay run accepts only 39.1% of Bush samples, despite successful recovery of its selected mask.',
            'distinct': 'Perfect evaluator acceptance in the best no-replay run means all scored samples pass target recognition; it does not establish perfect image quality, diversity, or absence of memorization.',
            'anteater': 'The best replay run scores 90.6% anteater acceptance with the broader distractor vocabulary. Some generated animals still have anatomical artifacts or repeated compositions.',
        }
        section.append('<p><b>Interpretation:</b> ' + notes[suite] + '</p>')
        for arm in ['replay', 'noreplay']:
            candidates = [tag for tag in runs if tag.startswith(arm + '_')]
            tag = (min if suite == 'filters' else max)(candidates, key=scores.get)
            r = runs[tag]
            assert r['complete'] and r['support_images_per_draw'] == 4
            assert all(v['exact_teacher_rate'] == 1 and v['unique_masks'] == 1 for v in r['summary'].values())
            folder = (BASE / 'mixed_anteater/artifacts/ic4/anteater' if suite == 'anteater' else BASE / 'ic4_recall/artifacts' / suite) / tag / 'evaluation_fixed_atoms'
            label = ('Replay' if arm == 'replay' else 'No replay') + ', seed ' + tag.split('_s')[1]
            record = dict(dataset=suite, run=tag, mean=scores[tag], targets=r['summary'],
                          support=4, refinement=False, offline_repair=True,
                          images_per_target=r['images_per_mask'], source=str(folder.relative_to(BASE)))
            selected.append(record)
            overview.append(f'<tr><td>{title}</td><td>{label}{" ★" if tag == best else ""}</td>'
                            + ''.join(f'<td>{value(r["summary"][n]["mean"], suite)}</td>' for n in names)
                            + f'<td><b>{value(scores[tag], suite)}</b></td><td>4 / no</td></tr>')
            section.append(f'<h3>{label}{" — best overall" if tag == best else ""}</h3><p>Mean: <b>{value(scores[tag], suite)}</b>. '
                           f'{r["images_per_mask"]} scored generated images per target; {r["support_draws"]} support draws. '
                           'Exact repaired-mask recall: 100% of evaluated draws.</p>')
            if suite == 'anteater':
                section.append(f'<p>Broader 1,019-label anteater acceptance: <b>{broad[tag]["Giant_anteater"]["target_top1_rate"]:.1%}</b>. '
                               'This supplementary score did not select the displayed run.</p>')
            for n in names:
                v = r['summary'][n]
                section.extend([f'<article><h4>{n.replace("_", " ")} · {value(v["mean"], suite)}</h4>',
                                '<div class="support"><p>Four support images — first draw</p>',
                                image(folder / f'{n}_support4.png', f'{n}: four support images'), '</div>',
                                f'<p>First {20 if suite == "filters" else 16} generated images, in saved order; no image selection</p>',
                                image(folder / v['first_grid'], f'{n}: generated samples'), '</article>'])
            section.append(f'<p class="source">Source: {record["source"]}/summary.json</p>')
        section.append('<details><summary>All three seeds in both arms</summary><table><tr><th>Run</th>'
                       + ''.join(f'<th>{n.replace("_", " ")}</th>' for n in names) + '<th>Mean</th></tr>')
        for tag, r in runs.items():
            section.append(f'<tr><td>{tag}</td>' + ''.join(f'<td>{value(r["summary"][n]["mean"], suite)}</td>' for n in names) + f'<td>{value(scores[tag], suite)}</td></tr>')
            for n, v in r['summary'].items():
                allrows.append(dict(dataset=suite, run=tag, target=n, metric=v['metric'], score=v['mean'],
                                    support=4, refinement=False, offline_repair=True, images=r['images_per_mask'], selected=any(z['dataset']==suite and z['run']==tag for z in selected)))
        section.append('</table></details></section>'); sections.extend(section)
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Best four-image continual recall results</title><style>',
             'body{font:17px/1.55 system-ui,sans-serif;color:#182b38;background:#f5f7fa;max-width:1180px;margin:auto;padding:32px}h1{font-size:38px;line-height:1.2}h2{margin-top:0}section,.panel{background:white;padding:28px;margin:28px 0;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #ddd}th{background:#edf2f6}img{max-width:100%;height:auto}article{border-top:1px solid #ddd;padding-top:12px;margin:24px 0}.support{max-width:560px}.source{font:12px monospace;overflow-wrap:anywhere}a{color:#1659a0}.muted{color:#526577}nav{display:flex;gap:22px;flex-wrap:wrap}summary{cursor:pointer;font-weight:600}@media print{body{background:white;padding:0;font-size:11pt}section{padding:0;break-before:page}article{break-inside:avoid}details{display:block}nav{display:none}}',
             '</style></head><body><h1>Best four-image continual recall results</h1>',
             '<p class="muted">Completed experiments · 6 October 2026 · Unified atom-wise input conditioning</p>',
             '<p>All four datasets support direct recall from four images using the same frozen feature pipeline and atom-wise readout method. '
             'The best generator training arm varies by dataset. The mixed-object experiment uses giant anteater in place of Hoatzin.</p>',
             '<nav>' + ''.join(f'<a href="#{s}">{v[0]}</a>' for s, v in suites.items()) + '</nav>',
             '<div class="panel"><h2>Best complete runs</h2><p>★ marks the best arm/seed for that dataset. '
             'Targets 1–3 follow the training orders listed below. Means are unweighted across the three targets.</p>',
             '<table><tr><th>Dataset</th><th>Generator training</th><th>Target 1</th><th>Target 2</th><th>Target 3</th><th>Mean</th><th>Support / refine</th></tr>',
             *overview, '</table><p>Filters: monochrome / poster / sketch (KID ↓). LFW: Bush / Powell / Blair. '
             'Distinct: Whoopi / Willis / Kidman. Mixed: Whoopi / anteater / BMW i8 (acceptance ↑).</p></div>',
             '<div class="panel"><h2>Method and support</h2><ol>',
             '<li><b>Generator:</b> the original block-sequential Bayes FTN shared rank-one LoRA bank, with replay or no replay during generator weight updates. Each run finishes all three task blocks before the recall evaluation shown here.</li>',
             '<li><b>Offline preparation:</b> search/repair masks using training images and choose teachers using an internal training-validation split with DINO/style distribution matching. Fit the conditioner sequentially from training-image episodes. This preparation uses more than four images and can revisit earlier-task images in both arms.</li>',
             '<li><b>Frozen unified features:</b> DINOv2 embeddings, appearance/style features, and CASIA FaceNet embeddings, applied to every image in every dataset. Average four image representations; add a frozen 128-dimensional random ReLU feature map.</li>',
             '<li><b>Distributed mask prediction:</b> a separate scalar readout per atom, fitted by ridge regression with accumulated sufficient statistics; select the top-k atom scores. Inference does not look up a stored whole-task mask.</li>',
             '<li><b>Recall:</b> exactly four support images, direct4, no online refinement. Each suite has its own generator bank and fitted readout; the feature pipeline and fitting rule are shared across suites.</li></ol>',
             '<p><b>Scope:</b> final-checkpoint recall with offline repaired-mask supervision. This is not an end-to-end replay-free method, four-image discovery of a new generative task, or a jointly online-trained generator/conditioner demonstration.</p></div>',
             '<div class="panel"><h2>How to read the evidence</h2><p>We select the best of seeds 123–125 separately for each arm by mean primary score across all three targets: minimum KID for filters, maximum acceptance otherwise. '
             'These are explicitly best observed development runs, selected after evaluator scoring; all seeds are included below. No target-wise seed mixing or manual image cherry-picking is used.</p>',
             '<p>Face acceptance uses an independent VGGFace2 FaceNet evaluator, distinct from the CASIA conditioner. Mixed objects use FaceNet for Whoopi and CLIP for animal/car recognition. Acceptance measures target recognition, not anatomical correctness or visual quality. Filter KID measures distribution similarity and is not comparable numerically with acceptance.</p>',
             '<p>Ten overlapping support draws per target all retrieve the same selected teacher mask in the exhibited runs. Identical masks reuse the same paired generated batch: 200 images per filter target or 64 per face/object target, not ten independent batches. Exact mask recovery does not imply perfect generation.</p>',
             '<p>Mixed-object supplementary audit: anteater scoring adds 1,000 ImageNet distractors. Strict held-out real-anteater calibration is 21/23; one original test reference overlapped a preliminary recognition screen and was excluded from that calibration. The support pool is independent of those screens. Animal anatomy, diversity, and possible training-image resemblance still require visual inspection.</p></div>',
             *sections,
             '<div class="panel"><h2>Sources and reproducibility</h2><p>This report embeds all displayed images and can be shared as one HTML file. '
             'Adjacent selected_results.json and all_results.csv contain the numerical records. Existing experiment artifacts were read without modification; no training or new generation was performed for this report.</p>',
             '<p>Primary sources: continual/ic4_recall/artifacts/summary_atoms.json (filters, LFW, distinct); '
             'continual/mixed_anteater/artifacts/summary.json and broad_label_audit.json (mixed objects). '
             'The original Hoatzin study remains available in continual/ic4_recall/artifacts/report_atoms.html but is superseded in this four-dataset exhibition.</p></div></body></html>']
    (OUT / 'report.html').write_text('\n'.join(parts))
    (OUT / 'selected_results.json').write_text(json.dumps(selected, indent=2))
    with (OUT / 'all_results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(allrows[0])); writer.writeheader(); writer.writerows(allrows)
    assert len(selected) == 8 and len(allrows) == 72
    print(json.dumps({'report': str(OUT / 'report.html'), 'selected_runs': len(selected), 'target_results': len(allrows),
                      'embedded_images': (OUT / 'report.html').read_text().count('data:image/png;base64,')}, indent=2))


if __name__ == '__main__':
    main()
