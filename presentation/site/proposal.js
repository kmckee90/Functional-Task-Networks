'use strict';
// The live proposal is authored in presentation/proposal.md.
// The legacy draft renderer remains below for reference; it no longer controls the page.
const proposalEscape = value => String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const proposalAsset = path => '/assets/proposal/'+path.split('/').map(encodeURIComponent).join('/');
const proposalMedia = (file,title,caption,video=false,project='') => `<figure class="proposal-film">${video
  ? `<video controls playsinline preload="none" poster="/assets/preview/${file}.jpg" aria-label="${proposalEscape(title)}" src="/assets/preview/${file}.mp4"></video>`
  : `<img loading="lazy" src="${proposalAsset(file)}" alt="${proposalEscape(title)}">`}
  <figcaption><strong>${project?`<a href="#/projects/${project}">${proposalEscape(title)} ↗</a>`:proposalEscape(title)}</strong>${proposalEscape(caption)}</figcaption></figure>`;
const proposalReports = {
  'agents': {
    link:'Watch the current agents explore', title:'Learning through experience.', category:'Demonstration',
    copy:'Globo gives an agent a first-person world to explore: terrain, animals, resources, and hazards. The research asks whether experience can become a growing repertoire of useful abilities.',
    note:'Selected recordings at their original playback speed. The fighting segment comes from an earlier joint Dreamer/explorer run; only its observation pane is shown. These examples illustrate behavior rather than establish reliable mastery of each skill.',
    media:['globo-m2e','Meta-Learning to Explore','Wandering, cliffside exploration, hilltop surveying, tree climbing, and a wolf encounter.',true],
    next:'phase-zero'
  },
  'related-work': {
    link:'Related approaches and the problems this project targets', title:'What should an explorer learn?', category:'Research context',
    copy:'Memory density measures familiarity. Model disagreement measures uncertainty about predicted outcomes. Goal-reaching methods address another question: can the agent deliberately return to a useful situation? This project investigates how discovery can become retained capability.',
    note:'The detailed comparison will distinguish assumptions, model sizes, experience budgets, and evaluation tasks. This clip is an illustration, not a benchmark comparison.',
    media:['maze','Exploration with changing walls','Eight-second excerpt from the supplied randomized-maze demonstration.',true],
    next:'m2e'
  },
  'm2e': {
    link:'How M2E uses memory and novelty feedback', title:'Learn how to explore.', category:'M2E · exploration',
    copy:'M2E feeds a memory-based familiarity signal into its exploration policy. As experience accumulates, the policy can respond to what is becoming familiar. Offline learning from stored trajectories helps it seek promising frontiers without first requiring an accurate world model.',
    note:'The full report will explain online density feedback, offline novelty rewards, and tests involving noise and repeated observations.',
    media:['maze-m2e','Exploration progress across three maze settings','Four-second excerpts at 0, 25k, 50k, and 75k steps, plus the final four seconds labeled 95k; original playback speed.',true],
    next:'ftn'
  },
  'ftn': {
    link:'FTN demonstrations of retention, recovery, and transfer', title:'Keep what has been learned.', category:'FTN · continual learning',
    copy:'FTN represents different learned functions through overlapping selections of shared neural resources. Examples from a situation help recover an appropriate selection. The aim is to retain distinct solutions while allowing shared components to support transfer.',
    note:'Selected robotics demonstrations. The expanded report will explain the training sequence and support used to recover each task; this is not evidence of unsupervised goal discovery.',
    media:['ftn-robotics','Functional Task Networks · robotics','Plate slide, drawer open, button press, peg insert, and faucet close. Task execution and active neural assemblies; original GIF timing.',true],
    next:'hierarchy'
  },
  'phase-zero': {
    link:'Phase 0 preliminary experiments, results, and limitations', title:'A working platform. An open question.', category:'Phase 0 · preliminary results',
    copy:'The preliminary experiments established continuous exploration, world-model training, and inspectable trajectories. Spatial discovery helps measure coverage; it does not establish that the agent has acquired reusable skills.',
    note:'This supplied trace is illustrative. The complete archive will attach configurations, seeds, budgets, and limitations. Current Dreamer+Novelty and P2E runs use different model sizes.',
    media:['exploration/globo/old/2026-10-10_11-31.png','Spatial discovery in an earlier Globo run','Coverage trace supplied with the proposal; not a controlled comparison between methods.'],
    next:'hierarchy'
  },
  'hierarchy': {
    link:'The next experiment—retained behaviors and hierarchical exploration', title:'A discovery becomes an ability.', category:'Next experiment · proposed',
    copy:'Learn a reusable behavior from a trajectory that reaches a useful situation. Retain it through FTN. Then let a higher-level explorer decide when to invoke it. The test is whether these abilities make later discoveries faster or enable new outcomes.',
    note:'This robotics example motivates the next Globo experiment. It does not demonstrate the proposed autonomous hierarchy in the current Globo agents.',
    media:['robotics','Movement and active subspaces','Eight-second excerpt from a supplied robotics demonstration.',true],
    next:'funding'
  },
  'funding': {
    link:'Budget, timeline, team, and institutional arrangements', title:'Fund the next decisive experiment.', category:'Research program',
    copy:'The sequence is concrete: establish repeatable exploration, demonstrate retained goal-reaching abilities, test their cumulative benefit, and then investigate composition and transfer.',
    note:'Funding amount, duration, personnel, compute budget, and institutional arrangements remain to be specified. No investment terms or delivery dates are committed here.',
    media:['planning and representation/CSCG/diagram.png','From experience to a usable representation','Supplied research diagram illustrating one part of the broader program; not a delivery schedule.'],
    next:'safety'
  },
  'neuroscience': {
    link:'Biological motivation and prospective neuroscience tests', title:'A computational hypothesis about learning.', category:'Neuroscience',
    copy:'FTN draws inspiration from complex cellular units and overlapping neural populations. The research will examine interference, recovery, and shared resource use, then ask which predictions can distinguish this account from alternatives.',
    note:'The spatial display deliberately groups co-active units. Its appearance is not independent evidence that biological cortical columns have emerged.',
    media:['ftn-classification','Learning and recall of classifiers','Three classification functions are trained into one network in sequence. Each is then recalled and trained further, without additional loss from interference with the other functions.',true],
    next:'ftn'
  },
  'safety': {
    link:'Experimental boundaries, monitoring, and safety plan', title:'Expand capability with evidence.', category:'Experimental control',
    copy:'Current experiments use simulated worlds with a limited action set, logged behavior, and checkpoints. Expansion into external tools or physical systems will require separate capability and safety evaluations.',
    note:'The fuller plan will specify intervention conditions and tests for unwanted behavior. First-person grounding and a novelty objective do not by themselves establish alignment.',
    media:['globo','An observable, bounded experiment','Representative simulation footage; monitoring is one control, not a complete safety argument.',true],
    next:'funding'
  }
};
const proposalDestinations={
  agents:['/projects/globo','Globo: exploration behavior and comparisons'],
  m2e:['/projects/maze','M2E: blind maze exploration and results'],
  ftn:['/projects/classification','FTN: retention, recovery, and knowledge transfer'],
  'phase-zero':['/projects/globo','Globo: current comparisons with Dreamer and Plan2Explore'],
  hierarchy:['/projects/robotics','Robotics: movement segments and transferable control'],
  neuroscience:['/projects/classification','FTN: learning through neural assemblies'],
  funding:['/research-plan','Research plan and funding']
};
const proposalLink = (id,label) => {
  const target=proposalDestinations[id];
  return target?`<a href="#${target[0]}">${proposalEscape(label||target[1])} <span aria-hidden="true">↗</span></a>`:'';
};
const explorationVideos = () => `<div class="exploration-videos">${[['globo-m2e','Globo','globo'],['maze-m2e','Blind Maze Exploration','maze'],['body-m2e','Native locomotion','locomotion']].map(([file,title,id])=>proposalMedia(file,title,'',true,id)).join('')}</div>`;
const ftnVideos = () => `<div class="exploration-videos ftn-videos">${[['ftn-robotics','Flexible robotic arm control','robotics'],['ftn-classification','Learning and recall of classifiers','classification'],['ftn-planning','Learning abstract actions for planning','planning']].map(([file,title,id])=>proposalMedia(file,title,'',true,id)).join('')}</div>`;
let proposalLoopId=0;
const proposalLoop = () => {
  const id='exploration-loop-'+(++proposalLoopId);
  const point=degrees=>{const a=degrees*Math.PI/180;return [300+185*Math.cos(a),270+185*Math.sin(a)];};
  const labels=[['Explore'],['Discover'],['Learn and','retain'],['Plan with','acquired abilities']];
  // Intersect the circular path with each rounded box, rather than trimming
  // every arrow by the same angle (horizontal and vertical edges differ).
  const boxEdge=(angle,direction)=>{
    const [cx,cy]=point(angle);
    let inside=0,outside=45;
    for(let n=0;n<40;n++){
      const offset=(inside+outside)/2,[x,y]=point(angle+direction*offset);
      const dx=Math.abs(x-cx)-(81-18),dy=Math.abs(y-cy)-(40-18);
      const distance=Math.hypot(Math.max(dx,0),Math.max(dy,0))+Math.min(Math.max(dx,dy),0)-18;
      if(distance<=0)inside=offset;else outside=offset;
    }
    return point(angle+direction*(inside+outside)/2);
  };
  const arrows=labels.map((_,i)=>{const a=-90+i*90;const p=boxEdge(a,1),q=boxEdge(a+90,-1);return `<path d="M ${p.join(' ')} A 185 185 0 0 1 ${q.join(' ')}" marker-end="url(#${id}-arrow)"/>`;}).join('');
  const nodes=labels.map((lines,i)=>{const [x,y]=point(-90+i*90);return `<g transform="translate(${x},${y})"><rect x="-81" y="-40" width="162" height="80" rx="18"/><text class="loop-label">${lines.map((line,j)=>`<tspan x="0" y="${lines.length===1?6:-5+j*21}">${line}</tspan>`).join('')}</text></g>`;}).join('');
  return `<figure class="proposal-flow"><svg viewBox="0 0 600 540" role="img" aria-labelledby="${id}-title ${id}-desc"><title id="${id}-title">The explore–learn loop</title><desc id="${id}-desc">Explore, discover, learn and retain, plan with acquired abilities, and return to explore. Arrows connect the stages clockwise in a repeating cycle.</desc><defs><marker id="${id}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 1 1 L 9 5 L 1 9" fill="none" stroke="#7de1bf" stroke-width="1.5"/></marker></defs><circle class="loop-orbit" cx="300" cy="270" r="185"/><circle class="loop-center" cx="300" cy="270" r="82"/><text class="loop-center-label" x="300" y="264">Explore–learn<tspan x="300" dy="25">loop</tspan></text><g class="loop-nodes">${nodes}</g><g class="loop-arrows">${arrows}</g></svg></figure>`;
};
let proposalDirectory = () => ''; // Populated from the project catalog in projects.js.

pages['/overview-unlisted'] = () => `<div class="wrap">
  <section class="proposal-hero"><div class="eyebrow">A proposal for naturalistic AGI</div><h1>Intelligence from Experience</h1><p class="lead">The goal is a perpetual, fully autonomous explore-learn loop.</p><div class="proposal-actions"><a href="#/theory">Read the full proposal ↗</a>${proposalLink('agents','Watch an exploration recording')}</div></section>
  <section class="proposal-layout"><div class="proposal-copy"><h2>The developmental loop.</h2><p>I propose to build an artificial intelligence that develops through its own life: exploring an unfamiliar world, discovering what it can do, and retaining the knowledge and skills that allow it to discover more.</p><p>As its repertoire grows, the agent can attempt things that were beyond its reach at the beginning. People should not have to choose every lesson, define every task, or continually reconstruct what the agent has forgotten.</p>${proposalLoop()}<p class="proposal-note">The central hypothesis: retained abilities can make further learning possible.</p></div><div>${proposalMedia(...proposalReports.agents.media)}<p class="proposal-note">Illustrative recording from the preliminary work. Reusable, hierarchical capabilities are the next experiment.</p></div></section>
  <section class="proposal-section proposal-copy"><h2>Two ingredients, built for this purpose.</h2><p>I developed <strong>meta-learning-based exploration (M2E)</strong> and <strong>functional task networks (FTN)</strong> at the Astera Institute to target naturalistic, brain-like AGI.</p><h3>M2E learns how to Explore</h3>${explorationVideos()}<p>Memory-based feedback tells the agent what is becoming familiar. It can learn to seek promising frontiers before it has an accurate predictive model of its world.</p>${proposalLink('m2e')}<h3>FTN retains different learned solutions.</h3>${ftnVideos()}<p>Overlapping selections of neural resources allow distinct learned functions to coexist. Evidence from a situation helps recover an appropriate selection, while shared components can support transfer.</p>${proposalLink('ftn')}</section>
  ${proposalDirectory()}</div>`;

pages['/innovations'] = () => `<div class="wrap"><section class="proposal-hero"><div class="eyebrow">Supporting the proposal</div><h1>Projects & demonstrations.</h1><p class="lead">Short previews of the methods, preliminary evidence, and proposed next steps.</p></section>${proposalDirectory()}</div>`;
pages['/research-plan'] = () => {
  const plan=revisedProposalText?.match(/^## Plan\n[\s\S]*?(?=^## |$(?![\s\S]))/m)?.[0];
  const content=plan ? renderRevisedProposal('# Summary\n'+plan.replace('[Research plan and funding ↗](#/research-plan)','')).replace('<h2>Summary</h2>','') : '<p>Loading the timeline…</p>';
  return `<div class="wrap"><article class="proposal-paper"><h1>Research plan and funding</h1>${content}<p><a href="#/">Back to proposal ↗</a></p></article></div>`;
};
// Old bookmarks resolve to current summaries, never the retired note previews.
for(const id of Object.keys(proposalReports))pages['/notes/'+id]=()=>pages[proposalDestinations[id]?.[0]||'/projects']();

let revisedProposalText = null;
let revisedProposalError = false;
function renderRevisedProposal(text) {
  const lines=text.slice(text.indexOf('# Summary')).split('\n');
  const links=new Map(Object.entries(proposalReports).map(([id,r])=>[r.link,id]));
  const output=[];let paragraph=[];let excluded=false;let globoAfterParagraph=false;let section='';
  function draftLink(label){
    if(label==='read more')return proposalLink(section.includes('exploration')?'m2e':'ftn','read more');
    if(label==='maze')return proposalLink('m2e','Maze exploration');
    if(label==='maze project')return proposalLink('m2e','Maze exploration');
    if(label==='m2e paper')return '<a href="https://arxiv.org/pdf/2503.02831" target="_blank" rel="noopener">M2E paper ↗</a>';
    if(label==='Experimental boundaries, monitoring, and safety plan')return '<a href="#/experimental-control">Experimental boundaries, monitoring, and safety plan ↗</a>';
    if(label==='the paper')return `<a href="${proposalAsset('FTN.pdf')}" target="_blank" rel="noopener">FTN manuscript ↗</a>`;
    if(label==='solving continual learning page')return proposalLink('ftn','Solving continual learning');
    const id=links.get(label);return id?proposalLink(id):'';
  }
  function flush(){if(paragraph.length){output.push('<p>'+proposalEscape(paragraph.join(' '))
    .replace(/\[bold\](.*?)\[bold\]/g,'<strong>$1</strong>')
    .replace(/transformers\[text link to LLM project\]/g,'<a href="#/projects/chatftn">transformers</a>')
    .replace(/diffusion \[text link to SiT project\]/g,'<a href="#/projects/diffusion">diffusion</a>')
    .replace(/\[text link to SiT page\]\[([^\]]+)\]/g,'<a href="#/projects/diffusion">$1</a>')
    .replace(/\[link: ([^\]]+)\]/g,(_,label)=>draftLink(label))+'</p>');paragraph=[];if(globoAfterParagraph){output.push(proposalMedia('globo-m2e','Exploring the world of Globo','Selected exploration behaviors: wandering, surveying, climbing, and interacting with wildlife. The wolf encounter comes from an earlier combined Dreamer/explorer agent.',true));globoAfterParagraph=false;}}}
  for(const raw of lines){
    const line=raw.trim();
    if(line.startsWith('# ')){
      flush();section=line.slice(2);excluded=line.includes('####EXCLUDE FOR NOW');globoAfterParagraph=line==='# From exploration to cumulative capability';
      if(excluded){output.push('<div class="proposal-inline-link">'+proposalLink('funding',line.slice(2).split('####')[0].trim())+'</div>');continue;}
    }
    if(excluded)continue;
    if(!line){flush();continue;}
    if(line==='[m2e paper link, title as link text]'){
      flush();output.push('<p class="result-source"><a href="https://arxiv.org/pdf/2503.02831" target="_blank" rel="noopener">Meta-Learning to Explore via Memory Density Feedback ↗</a></p>');continue;
    }
    if(line==='[ftn paper link, title as link text]'){
      flush();output.push(`<p class="result-source"><a href="${proposalAsset('FTN.pdf')}" target="_blank" rel="noopener">Fully Autonomous Continual Learning and Knowledge Transfer via Cortex-Inspired Neural Assemblies ↗</a></p>`);continue;
    }
    if(line==='[Chart: FTN+M2E vs M2E vs Dreamer vs P2E]'){
      flush();output.push(resultFigure('/assets/preview/globo-ftn-seed19.png?v=labels2',
        'Spatial discoveries for FTN+Metalearn2Explore, Metalearn2Explore, Plan2Explore, and Dreamer',
        'Cumulative spatial discovery at matched observation counts. The comparison ends at FTN+Metalearn2Explore’s observation count when this snapshot was captured.'));continue;
    }
    if(line==='[LLM result: continual learning/llm/c794f9c6-35bd-4abe-9580-c5c49bd79791.png]'){
      flush();output.push(resultFigure(proposalAsset('continual learning/LLM/c794f9c6-35bd-4abe-9580-c5c49bd79791.png'),'Continual pretraining: FTN, a basic transformer, and end-to-end test-time training','Parameter-matched, two-task sequential training: retention and transfer in language models.'));continue;
    }
    if(line==='[Line chart results for each kind of maze, here]'){
      flush();output.push('<div class="result-grid">'+[
        ['fixed','Fixed recurring environments'],['random','Randomized environments'],
        ['continual','Large, continual environments'],['noisy','Environments with noise']
      ].map(([id,title])=>resultFigure('/assets/preview/maze-'+id+'.png',title+': exploration coverage comparisons','',title)).join('')+
        '</div><p class="result-source">Original paper results: median coverage across 32 runs per condition; shaded regions show the full range. “Both” is full M2E; OCP is the Go-Explore-style return-to-frontier comparator; FCP is the feedback-conditioned policy.</p>');continue;
    }
    if(line==='[M2E vs Dreamer vs P2E charts from Globo].'){
      flush();output.push(resultFigure('/assets/preview/globo-seed-19.png','M2E, Dreamer and Plan2Explore spatial coverage','At 223,000 observations: M2E visits 1,102 distinct ground cells, Dreamer 185, and Plan2Explore 410.')+resultFigure('/assets/preview/globo-seed-101.png','M2E, Dreamer and Plan2Explore spatial coverage in the second world','At 446,000 observations: M2E visits 1,556 distinct ground cells, Dreamer 1,015, and Plan2Explore 651.'));continue;
    }
    if(line==='[Results figures comparing against baselines from the paper]'){
      flush();output.push('<div class="result-grid">'+resultFigure('/assets/preview/ftn-accuracy.png','FTN and baseline retained classification accuracy','Retained accuracy on shared-shuffle MNIST.')+resultFigure('/assets/preview/ftn-headstart.png','FTN and baseline new-task starting loss','New-task starting loss, before weight training. Lower is better.')+'</div><p class="result-source">Means over five seeds, redrawn from the FTN manuscript’s comparison table.</p>'+
        '<h4>Retaining earlier solutions and preparing for new ones</h4>'+
        '<p>These networks learn the same 16 classification tasks in sequence. After each task, they are tested on all 16. The ordinary network loses earlier solutions as it learns new ones. FTN keeps those solutions accessible and becomes better at tasks it has not yet trained on, by assembling components learned from previous tasks.</p>'+
        '<div class="result-grid">'+
        resultFigure(proposalAsset('results/ftn-unmasked.png'),'Ordinary network accuracy across 16 sequential classification tasks','Earlier solutions are overwritten.','Ordinary sequential learning')+
        resultFigure(proposalAsset('results/ftn-assemblies.png'),'FTN accuracy across the same 16 sequential classification tasks','Earlier solutions are retained; performance on untrained tasks improves.','Functional Task Networks')+
        '</div><p class="result-source">Original manuscript figures. White means higher accuracy. Below the diagonal: performance on tasks already learned. Above the diagonal: performance on tasks not yet trained.</p>'+
        '<h4>New learning improves earlier abilities</h4>'+
        '<p>Backward transfer asks whether learning later tasks helps or harms earlier ones. In a six-task language-model experiment, FTN-LoRA improves earlier-task accuracy by an average of 19 percentage points. Ordinary sequential LoRA loses 47 points, and LoRA with replay loses 9. The network can use later experience to improve what it learned before.</p>'+
        resultFigure('/assets/preview/ftn-backward-transfer.png','Backward transfer: sequential LoRA minus 47, LoRA with replay minus 9, and FTN-LoRA plus 19 percentage points','Six-task Qwen2.5 classification stream; means over five seeds, redrawn from the FTN manuscript. Positive values mean earlier abilities improved after learning later tasks.'));continue;
    }
    if(line==='[misc/baselines_table.png, caption: FTN was designed around missing capabilities for AGI, not for incremental improvements]'){
      flush();output.push(resultFigure(proposalAsset('misc/baselines_table.png'),'Comparison of continual-learning capabilities','FTN was designed around missing capabilities for AGI, not for incremental improvements'));continue;
    }
    if(line==='[FTN-M2E clip here]'){
      flush();output.push(`<figure class="proposal-film"><video controls playsinline preload="metadata" aria-label="FTN–M2E exploration and neural assemblies" src="${proposalAsset('exploration/globo/ftn/obs (5).mp4')}"></video><figcaption>FTN–M2E exploring Globo, with its frontier goal and active neural assembly.</figcaption></figure>`);continue;
    }
    const mediaMarkers={'[globo video]':['globo-m2e','Exploring the world of Globo','',true],
      '[maze video]':['maze-m2e','Blind Maze Exploration','',true],
      '[ftn classification video]':proposalReports.neuroscience.media};
    if(mediaMarkers[line]){
      flush();output.push(proposalMedia(...mediaMarkers[line]));
      if(line==='[globo video]')output.push('<figure class="proposal-comparison"><a href="/assets/preview/globo-ftn-seed19.png?v=labels2" target="_blank" rel="noopener"><img src="/assets/preview/globo-ftn-seed19.png?v=labels2" alt="Cumulative spatial discoveries for FTN+Metalearn2Explore, Metalearn2Explore, Plan2Explore, and Dreamer versus observations" loading="lazy"></a><figcaption>Cumulative spatial discovery in Globo, at matched observation counts. The chart ends at FTN+Metalearn2Explore’s observation count when this snapshot was captured.</figcaption></figure>');
      continue;
    }
    if(line==='[link to page with plans, timelines, budget etc]'){flush();output.push('<div class="proposal-inline-link">'+proposalLink('funding')+'</div>');continue;}
    if(line.startsWith('Explore -> discover ->')){flush();output.push(proposalLoop());continue;}
    if(line==='[FTN gifs]'){flush();output.push(proposalMedia(...proposalReports.neuroscience.media));continue;}
    if(line.startsWith('# ')){flush();output.push('<h2>'+proposalEscape(line.slice(2))+'</h2>');continue;}
    if(line.startsWith('## ')){flush();section=line.slice(3);output.push('<h3>'+proposalEscape(section)+'</h3>');continue;}
    if(line.startsWith('### ')){flush();section=line.slice(4);output.push('<h4>'+proposalEscape(section)+'</h4>');continue;}
    const match=line.match(/^\[link: (.+)\]$/);
    if(match){flush();const link=draftLink(match[1]);if(link)output.push('<div class="proposal-inline-link">'+link+'</div>');continue;}
    if(/^\d+\. /.test(line))flush();
    paragraph.push(line);
  }
  flush();return output.join('');
}
function renderProposalMarkdown(text){
  // Trusted, locally authored Markdown. Raw HTML supports videos and figure grids.
  return marked.parse(text,{gfm:true,breaks:false});
}
pages['/theory'] = () => `<div class="wrap"><article class="proposal-paper">${revisedProposalText ? renderProposalMarkdown(revisedProposalText) : `<p role="status">${revisedProposalError?'The proposal could not load. Please reload.':'Loading the proposal…'}</p>`}</article></div>`;
pages['/']=pages['/theory'];
pages['/experimental-control']=()=>{
  const section=revisedProposalText?.split('## Safety and experimental control\n')[1]?.split('\n## ')[0];
  const body=section?.replace('[Experimental boundaries, monitoring, and safety plan ↗](#/experimental-control)','').replace('[Browse results ↗](#/projects)','');
  return `<div class="wrap"><article class="proposal-paper"><h1>Experimental boundaries, monitoring, and safety plan</h1>${body?renderProposalMarkdown(body):'<p>See the proposal for the current experimental-control discussion.</p>'}<a href="#/">Back to proposal ↗</a></article></div>`;
};

// Keep older bookmarked entry points useful without requiring live model APIs.
for(const [old,id] of Object.entries({'/innovations/exploration':'m2e','/innovations/continual-learning':'ftn','/innovations/planning':'hierarchy','/how-it-works/exploration':'m2e','/how-it-works/learning':'ftn','/demos/globo':'agents','/demos/robotics':'hierarchy','/demos/images':'ftn','/demos/llms':'ftn','/resources':'funding'}))pages[old]=pages['/notes/'+id];
pages['/demos']=pages['/innovations'];
function highlightProposalNotes(){
  if(location.hash.startsWith('#/notes/')){
    const link=document.querySelector('#nav a[href="#/projects"]');
    link.classList.add('active');link.setAttribute('aria-current','page');
  }
}
window.addEventListener('hashchange',highlightProposalNotes);
route();
highlightProposalNotes();
fetch('/source/theory').then(response=>{if(!response.ok)throw Error('Proposal unavailable');return response.text()}).then(text=>{revisedProposalText=text;if(['','#/','#/theory','#/experimental-control'].includes(location.hash))route();}).catch(()=>{revisedProposalError=true;if(['','#/','#/theory','#/experimental-control'].includes(location.hash))route();});
