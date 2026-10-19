function selectedAggregates(runs, hidden, column) {
  const selected=runs.filter(r=>!hidden[r.group||r.name]);
  function value(rows,x){let lo=0,hi=rows.length;while(lo<hi){const mid=(lo+hi)>>1;if(rows[mid][0]<=x)lo=mid+1;else hi=mid;}return rows[Math.max(0,lo-1)][column];}
  const groups=[];
  for(const model of [...new Set(selected.map(r=>r.model))]){
    const members=selected.filter(r=>r.model===model&&r.rows.length);
    if(!members.length)continue;
    const cap=Math.max(...members.map(r=>r.rows.at(-1)[0]));
    const start=Math.min(...members.map(r=>r.rows[0][0]));
    const xs=[...new Set([start,cap,...members.flatMap(r=>r.rows.filter(p=>p[0]>=start&&p[0]<=cap).map(p=>p[0]))])].sort((a,b)=>a-b);
    const points=xs.map(x=>{const ys=members.filter(r=>r.rows[0][0]<=x&&r.rows.at(-1)[0]>=x).map(r=>value(r.rows,x));return {x,n:ys.length,mean:ys.reduce((a,b)=>a+b,0)/ys.length,min:Math.min(...ys),max:Math.max(...ys)};}).filter(p=>p.n);
    groups.push({model,color:members[0].color,n:members.length,cap,points});
  }
  return {groups,selected};
}
let hidden={};try{hidden=JSON.parse(localStorage.getItem('phase5-hidden')||'{}');}catch{}
function redrawSelected(){
  document.querySelectorAll('.curve').forEach(p=>p.classList.toggle('hidden',!!hidden[p.dataset.run]));
  let result;
  for(const svg of document.querySelectorAll('.comparison-chart')){
    const column=svg.dataset.metric==='total_cells'?1:2;
    result=selectedAggregates(comparisonRuns,hidden,column);
    const xy=(x,y)=>[70+655*x/Number(svg.dataset.xmax),315-230*y/Number(svg.dataset.ymax)];
    const layer=svg.querySelector('.aggregate');layer.replaceChildren();
    function add(tag,attrs,title){const el=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);if(title){const t=document.createElementNS(el.namespaceURI,'title');t.textContent=title;el.append(t);}layer.append(el);}
    for(const g of result.groups){
      const polygon=[...g.points.map(p=>xy(p.x,p.max)),...[...g.points].reverse().map(p=>xy(p.x,p.min))];
      if(g.n>1)add('polygon',{points:polygon.map(p=>p.join(',')).join(' '),fill:g.color,opacity:'.12'},`${g.model}: min–max of runs with observations at each step; contributing count varies`);
      const path=g.points.map((p,i)=>(i?'L':'M')+xy(p.x,p.mean).join(',')).join(' ');
      add('path',{d:path,fill:'none',stroke:g.color,'stroke-width':4},`${g.model}: mean of ${g.n} selected runs`);
    }
  }
  const status=document.getElementById('selection-stats');
  if(!result.selected.length)status.textContent='No runs selected.';
  else if(!result.groups.length)status.textContent='Waiting for observations.';
  else status.textContent=result.groups.map(g=>`${g.model}: ${g.n} runs, through ${g.cap.toLocaleString()} observations (${g.points.at(-1).n} contributing at the end)`).join(' · ');
}
for(const c of document.querySelectorAll('.toggle')){
  c.checked=!hidden[c.dataset.run];
  c.onchange=()=>{hidden[c.dataset.run]=!c.checked;try{localStorage.setItem('phase5-hidden',JSON.stringify(hidden));}catch{}redrawSelected();};
}
redrawSelected();
