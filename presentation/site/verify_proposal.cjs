// Run with Node on a pod. The editable proposal is standard Markdown plus HTML media.
const fs=require('fs'),vm=require('vm');
const ctx=vm.createContext({pages:{}});
vm.runInContext(fs.readFileSync(__dirname+'/marked.js','utf8'),ctx);
vm.runInContext(fs.readFileSync(__dirname+'/proposal.js','utf8').split('// Keep older bookmarked')[0],ctx);
vm.runInContext(fs.readFileSync(__dirname+'/projects.js','utf8').split('const projectFeatures=')[0],ctx);
ctx.md=fs.readFileSync(__dirname+'/../proposal.md','utf8');
const html=vm.runInContext('renderProposalMarkdown(md)',ctx);
if((html.match(/<h1>/g)||[]).length!==1)throw Error('Expected one page title');
if(html.includes('[link:')||html.includes('[Chart:'))throw Error('Unresolved legacy placeholder');
if(!html.includes('<strong>missing capabilities</strong>'))throw Error('Markdown bold did not render');
for(const media of ['ftn-backward-transfer.png','ftn-unmasked.png','ftn-assemblies.png','maze-m2e.mp4','ftn-classification.mp4']){
 if(!html.includes(media))throw Error('Missing '+media);
}
vm.runInContext('revisedProposalText=md',ctx);
const page=vm.runInContext('pages["/theory"]()',ctx);
if((page.match(/<h1>/g)||[]).length!==1)throw Error('Page adds a duplicate title');
const safety=vm.runInContext('pages["/experimental-control"]()',ctx);
if(!safety.includes('Current experiments take place in simulated worlds'))throw Error('Safety excerpt missing');
console.log(JSON.stringify({videos:(html.match(/<video /g)||[]).length,images:(html.match(/<img /g)||[]).length,passed:true}));
