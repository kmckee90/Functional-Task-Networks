"""Combine the two completed generation shards and the transparent factual review."""
import csv
import html
import json
from pathlib import Path
ROOT=Path('/home/dev/ftn_chat_state/evaluations/qa_ic_20261015')
OUT=ROOT/'generation'
review=json.loads(Path('evaluation_results/qa_ic_20261015/generation/review.json').read_text())
read=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
rows=read(OUT/'answers.jsonl')[:15]+read(ROOT/'generation_part2/answers.jsonl')
assert len(rows)==30
assert len({(r['task'],r['index']) for r in rows})==30
annotations={(r['task'],r['index']):r for r in review['rows']}
assert set(annotations)=={(r['task'],r['index']) for r in rows}
summary={}
for task in ['base_ftn','explorer','rl_mt10']:
    rs=[r for r in rows if r['task']==task]
    entry={'questions':len(rs),'correct_routes':sum(r['route']['reason']=='matched' and r['route']['task']==task for r in rs),'routing_misses':sum(r['route']['reason']!='matched' for r in rs),'methods':{}}
    for method in ['base','ic','oracle']:
        counts={s:0 for s in ['correct','partial','incorrect','unscorable']}
        for r in rs:
            a=annotations[(task,r['index'])]
            grade=a.get(method,a['ic']) if method=='oracle' else a[method]
            counts[grade]+=1
        counts['scored']=len(rs)-counts['unscorable']
        counts['fully_correct_fraction']=counts['correct']/counts['scored']
        entry['methods'][method]=counts
    summary[task]=entry
result={'reviewer':review['reviewer'],'rubric':review['rubric'],'topics':summary,
    'limitations':['30 cached training questions, not held-out generalization.','Assistant review is not blinded or independently verified.','Three questions excluded for defective or insufficient reference/source.','Greedy generation, thinking disabled as in current app, 384-token cap. Outputs that hit the cap are explicitly marked.','Oracle uses the topic label for diagnosis only.','A separate four-question probe used the training open-think prefix; all four exhausted 384 tokens without a final answer. It is not included in correctness scores and does not settle whether a larger thinking budget would help.']}
(OUT/'correctness_summary.json').write_text(json.dumps(result,indent=2))
(OUT/'review.json').write_text(json.dumps(review,indent=2))
with (OUT/'correctness.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['topic','index','question','method','grade','truncated','answer','reference','reason'])
    for r in rows:
        a=annotations[(r['task'],r['index'])]
        for method,answer in r['answers'].items():
            grade=a.get(method,a['ic']) if method=='oracle' else a[method]
            w.writerow([r['task'],r['index'],r['question'],method,grade,answer['hit_token_limit'],answer['text'],r['reference'],a['reason']])
esc=lambda x:html.escape(str(x))
parts=['<!doctype html><meta charset="utf-8"><title>FTN-Chat recall audit</title><style>body{max-width:1100px;margin:40px auto;font:16px/1.5 system-ui;padding:0 20px}table{border-collapse:collapse}td,th{padding:8px 15px;border:1px solid #ccc}pre{white-space:pre-wrap;font:14px/1.5 system-ui}article{border-top:2px solid #ddd;margin-top:30px}.answers{display:grid;grid-template-columns:1fr 1fr;gap:30px}@media(max-width:700px){.answers{display:block}}</style><h1>FTN-Chat closed-book recall audit</h1>',
    '<p>'+esc(result['reviewer'])+'. '+esc(' '.join(result['limitations']))+'</p>',
    '<table><tr><th>Topic</th><th>Scored</th><th>Base correct</th><th>IC correct</th><th>IC partial</th><th>IC incorrect</th></tr>']
for task,d in summary.items():
    a=d['methods']['ic'];b=d['methods']['base']
    parts.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in [task,a['scored'],b['correct'],a['correct'],a['partial'],a['incorrect']])+'</tr>')
parts.append('</table>')
for r in rows:
    a=annotations[(r['task'],r['index'])]
    parts.append('<article><h2>'+esc(r['task']+' / '+str(r['index']))+'</h2><p><b>'+esc(r['question'])+'</b></p><p>'+esc(a['reason'])+'</p><div class="answers">')
    for method in ['base','ic']:
        answer=r['answers'][method]
        parts.append('<section><h3>'+esc(method+' — '+a[method])+('</h3><p>Token limit reached.</p>' if answer['hit_token_limit'] else '</h3>')+'<pre>'+esc(answer['text'])+'</pre></section>')
    parts.append('</div><details><summary>Reference and source excerpt</summary><pre>'+esc(r['reference'])+'</pre><pre>'+esc(r.get('context',''))+'</pre></details></article>')
(OUT/'report.html').write_text('\n'.join(parts))
print(json.dumps(summary,indent=2))
