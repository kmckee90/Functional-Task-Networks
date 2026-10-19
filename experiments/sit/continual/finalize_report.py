"""Add outcome-led interpretation and cross-seed tables to the generated report."""
import json
from pathlib import Path
import shutil
import numpy as np


def main():
    root=Path('/home/dev/sit_state')
    out=Path('/home/dev/workspace/continual/artifacts')
    summary=json.loads((out/'summary.json').read_text())
    runs=summary['runs']
    cells=[]
    quality_summary={}
    for condition in ['bayes_replay','bayes_noreplay','plain_lora']:
        names=[n for n in runs if n.startswith(condition+'_s')]
        quality_summary[condition]={}
        values=[]
        for d in ['photo','monochrome','poster','sketch']:
            vv=[]
            for name in names:
                q=runs[name].get('quality',[])
                vv.extend(r['kid'] for r in q if r['stage'].startswith('after_3_') and r['domain']==d and r['mode']=='cold40')
            if not vv:
                values.append('pending')
                continue
            mean=float(np.mean(vv)); sd=float(np.std(vv,ddof=1)) if len(vv)>1 else None
            quality_summary[condition][d]={'mean':mean,'sd':sd,'n':len(vv)}
            values.append(f'{mean:.4f}'+(f' ± {sd:.4f}' if sd is not None else ''))
        cells.append('<tr><td>'+condition+'</td><td>'+'</td><td>'.join(values)+'</td></tr>')
    (out/'quality_summary.json').write_text(json.dumps(quality_summary,indent=2))
    inference_rows=[]
    modes=['cold_4_30','cold_40_30','cold_80_30','cold_40_120','random_search30','ic_direct4','ic_refine4','retrieve_stored40']
    for mode in modes:
        vals=[]
        for d in ['monochrome','poster','sketch']:
            a=[runs[f'bayes_replay_s{s}']['diagnostics'][d][mode]['loss'] for s in [123,124,125]]
            vals.append(f'{np.mean(a):.4f} ± {np.std(a,ddof=1):.4f}')
        inference_rows.append('<tr><td>'+mode+'</td><td>'+'</td><td>'.join(vals)+'</td></tr>')
    lead='''<section style="background:#e6eeeb;padding:22px;border-left:5px solid #1c6a68">
<h2>Outcome: partial success, not learning without forgetting</h2>
<p><strong>The shared Bayes LoRA bank learned new styles and cold mask inference recovered useful domain-specific generation after sequential training. Mask-filtered replay helped, especially on poster retention. The experiment does not establish zero forgetting or reliable recall from arbitrarily sparse support.</strong></p>
<p>Three distinctions matter: (1) keeping the original frozen base intact is automatic; (2) keeping a previously saved mask functional is not automatic and often failed; (3) finding a new useful mask in the final bank often worked despite that staleness. The last case is the strongest positive result here.</p>
<p><a href="demo.html"><strong>Interactive lifecycle demo: compare cold recall with stale saved masks</strong></a></p>
</section>
<h2>Main findings</h2>
<ol>
<li><strong>New domains were acquired.</strong> Cold-inferred held-out loss improved after training each domain. The largest effect was sketch: mean loss fell from about 0.7812 to 0.7081 with replay. Samples show actual sketch and poster-style generation, not merely a lower training objective.</li>
<li><strong>Replay improved poster retention but did not win every loss comparison.</strong> Across three seeds, poster domain loss degradation after later sketch training averaged about 0.0039 with replay versus 0.0131 without. The mean fraction of the initial poster learning gain retained was about 81% versus 25%. Monochrome cold-inference MSE retention was weaker with replay, even though generated-image KID favored replay. These metrics measure different aspects and should not be collapsed into a single success claim.</li>
<li><strong>Saved masks became stale.</strong> Without replay, the monochrome and poster masks saved during training ended near losses 0.901 and 0.973, versus roughly 0.705 and 0.717 when saved. Replay reduced some damage but did not preserve all saved masks. Cold search on the final replay bank instead recovered roughly 0.7094 and 0.7236. A remembered mask ID is not a stable memory of a function when shared atoms continue changing.</li>
<li><strong>Support size matters for recall.</strong> On final replay models, sketch MSE averaged 0.7164 with four reference images versus 0.7081 with forty. Without replay that gap was larger: 0.7454 versus 0.7072. More support and more search steps were not uniformly better on every domain; 40-image, 30-step inference was a useful baseline, not a guaranteed optimizer.</li>
<li><strong>Input-conditioned retrieval was not needed for the main positive result.</strong> The optional pooled-latent PCA router picked the correct archived entry for all three learned domains in the tested support sets, but its old logits were stale. Poster loss was approximately 0.950 under retrieved priors, versus 0.724 under cold inference. Thirty deterministic refinement steps selected the same masks. This tests unrepaired priors; the original project's additional replay-based mask-repair mechanism was not evaluated here.</li>
<li><strong>The original reactive detector missed the domain boundaries.</strong> Every Bayes run recorded only the initial change-point refit. No forced domain-boundary mask resets were added. In seed123, replay ended with boundary-mask overlaps 0.625 (monochrome/poster) and 0.6875 (poster/sketch); without replay the latter pair was identical. Replay genuinely reached old data: reconstructed seeded selections came from earlier domains on 50.8% of poster updates and 78.5% of sketch updates.</li>
<li><strong>Prospective transfer was small and mixed.</strong> Future-domain columns were evaluated before training those domains. For replay seed123, sketch loss changed only from 0.7812 initially to 0.7782 after monochrome, then 0.7809 after poster, before falling to 0.7078 after sketch training. There is no broad zero-shot-transfer claim here: evaluation itself includes support-conditioned mask inference.</li>
</ol>
<h2>Generated-image check</h2>
<p>Final KID, mean ± sample SD over three seeds for the Bayes conditions; the ordinary-LoRA control has one seed. Lower is better. All use the same 200 generation seeds/class labels and 400 reference images per domain. These are small-sample diagnostics, not FID-50K results or statistical proof of superiority.</p>
<table><tr><th>Condition</th><th>Photo</th><th>Monochrome</th><th>Poster</th><th>Sketch</th></tr>'''+''.join(cells)+'''</table>
<p>For replay seed123, KID after each domain was acquired versus at the end was: monochrome 0.01895 → 0.01767; poster 0.02889 → 0.02625. Thus the image-distribution check did not show degradation for those two domains in this seed, despite some velocity-loss forgetting. Without replay, poster KID returned from 0.03734 to 0.03944, almost its initial 0.03974. Final replay quality varies across seeds, especially for sketch.</p>
<p>The ordinary-LoRA control learned sketch well but generated sketch-like outputs when earlier styles or photos were requested after the final block. Its final old-domain losses and KIDs degraded sharply. This is a matched <em>active-rank</em> control, not a parameter- or compute-matched superiority test: it has 8.55M trainable parameters versus 51.61M for the bank and does not pay the mask-search/replay cost.</p>
<h2>Frozen-final-model inference diagnostics</h2>
<p>All entries below freeze the same replay-trained weights. Cold names encode support-image count and optimization steps. Random search selects the lowest support-loss mask from 30 random masks. Stored retrieval selects among archived training masks by support loss, without a domain ID. PCA priors use four clean reference-image latent features. These additional diagnostics do not change the primary lifecycle results or train any weights.</p>
<table><tr><th>Inference method</th><th>Monochrome MSE</th><th>Poster MSE</th><th>Sketch MSE</th></tr>'''+''.join(inference_rows)+'''</table>
<p>Exact old-mask recovery is not the right success criterion: multiple masks can implement useful functions, and some old masks no longer do. The appropriate question is held-out loss and generated quality under the inferred mask.</p>
<h2>What this experiment establishes</h2>
<p>There is credible evidence of <strong>recoverable continual adaptation</strong> in this controlled setting: a shared bank plus support-conditioned inference retained useful earlier styles much better than a single sequential adapter. There is no evidence for strict preservation of all learned functions, universal benefit from replay, reliable four-image inference in every domain, or successful automatic boundary detection. The reversed-order seed retained useful sketch generation after learning the other styles, but one reverse-order run is not an order-invariance study.</p>
<p>Important remaining limits: three training seeds, one fixed support set per budget, ten ImageNet classes, synthetic style shifts rather than independent real-world datasets, finite training/replay capacity, and a smaller atom bank than the toy configuration. The four-image support set covers four classes; the forty-image set covers all ten. Query losses reuse fixed noise/time draws for paired comparisons and do not exhaustively integrate all transport times. No adaptation hyperparameters were selected against the reported query losses.</p>
<p><a href="verification.json">Independent completion audit</a>: all eight workers completed successfully, with 36,000 training updates and 12,800 metric-generation images across 64 quality evaluations. Source-file SHA256 checks found no exact byte duplicates across training, support and test splits; this does not exclude perceptual near-duplicates in upstream Imagenette. All runtime work was performed on the pod.</p>
'''
    path=out/'report.html'
    report=path.read_text()
    marker='<h2>Question and protocol</h2>'
    report=report.replace(marker,lead+marker)
    if (out/'repair'/'report.html').exists():
        report=report.replace(marker,
            '<p><strong>Follow-up:</strong> <a href="repair/report.html">Input conditioning with replay-based mask repair has now been tested.</a> The original input-prior results above describe unrepaired masks; see the follow-up for the updated comparison.</p>'+marker)
    path.write_text(report)
    for source,name in [(root/'environment_freeze.txt','environment_freeze.txt'),
                        (root/'data_manifest.json','data_manifest.json'),
                        (Path(__file__).with_name('protocol.json'),'protocol.json')]:
        shutil.copy2(source,out/name)
    # Verify report and demo asset links without running a browser locally.
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ('src','href') and v and not v.startswith(('http','#')) and '${' not in v:
                    assert (out/v).exists(), (tag,v)
    Links().feed(report)
    Links().feed((out/'demo.html').read_text())
    print('Final report, manifest, environment and links verified.')


if __name__ == '__main__':
    main()
