Intelligence from Experience — research proposal (October 2026)

CURRENT HOSTING — LOCAL LAPTOP (October 14, 2026)
See LOCAL_HOSTING.md. Run:
    bash presentation/site/rehost_local.sh
from the repository root. Proposal: http://127.0.0.1:8080/
Monitor: http://127.0.0.1:8765/continuous/
The current proposal is draft_v3.txt. Website files and generated media are local;
live monitor artifacts and all training remain on globo-dreamer. Base is no longer
needed for hosting. The notes below describe the earlier pod-hosted setup.

CURRENT PRESENTATION
The overview is in proposal.js. The full proposal is rendered directly from
../draft_v2.txt, with all nine [link:] markers connected to supporting pages.
Projects replaces the Research notes tab: projects.js defines eight projects
with a vertical thumbnail/headline list and dedicated pages. Six use the edited
video montages, ChatFTN embeds the live app, and diffusion includes a retained-
domain sample preview. Funding fields remain explicit draft placeholders.

Source is edited locally. Hosting and browser verification run on the base pod:
    /home/dev/proposal_source/presentation/site/
Runtime files, dependencies, short clips, and browser screenshots stay outside
sync in /home/dev/proposal_state/.

Current browser address: http://127.0.0.1:8080/
To reconnect from the laptop (infrastructure only, no local application):
    actl pod exec base -n obelisk --port 8080:8080
Leave that shell open while viewing the site. The static presentation is served
on base; the ChatFTN project additionally uses GPU 0 on the ftn-chat pod.

Pod server command:
    /home/dev/proposal_state/venv/bin/python /home/dev/proposal_source/presentation/site/server.py --port 8080 --chat http://10.42.20.231:7860
Server log: /home/dev/proposal_state/server.log

CHATFTN APPLET
On ftn-chat, in /home/dev/workspace:
    CUDA_VISIBLE_DEVICES=0 FTN_SERVER_NAME=0.0.0.0 FTN_ROOT_PATH=/chat FTN_WEBSITE_MODE=1 .venv/bin/python -u app.py --ckpt-dir /home/dev/ftn_chat_state/checkpoints_bayes_qa_ic --no-thinking
The bank, posterior, refs_usage.pt, and three saved input routes are loaded from
that directory. The source checkpoint is not modified. Website mode applies
the site's dark styling, hides advanced controls, disables thinking, and enables
repaired input routing by default. Outside website mode, the app retains its
usual controls and defaults. App log and PID are stored in
/home/dev/ftn_chat_state/site_chat.log and site_chat.pid. The base server proxies
/chat/ including streaming and queue POSTs. If ftn-chat is recreated, obtain its
new pod IP and restart the presentation server with the updated --chat address.
No external public Gradio share link is used.

Pod preparation and browser checks:
    /home/dev/proposal_state/venv/bin/python /home/dev/proposal_source/presentation/site/prepare_media.py
    PLAYWRIGHT_BROWSERS_PATH=/home/dev/proposal_state/browsers /home/dev/proposal_state/venv/bin/python /home/dev/proposal_source/presentation/site/check_site.py

LEGACY SITE REFERENCE
The notes below describe the previous live-demo integration. Its source remains
available, but the revised navigation uses brief, self-contained proposal pages
and does not call the live chat or experiment-monitor APIs.

No frontend packages, build step, or external fonts are required. New website
source lives entirely here. The server reads the SiT and robotics artifacts
from their existing repository locations; it does not copy or modify them.
Only selected artifact extensions and source references are exposed.

LIVE CONNECTIONS
The server expects these private, loopback-only forwarded services:
    Chat:  http://127.0.0.1:17860 → ftn-chat pod port 7860
    Globo: http://127.0.0.1:17865 → globo-dreamer pod port 8765
Override them with server.py --chat URL --globo URL if needed.

Keep each forwarding shell open (separate terminals):
    actl pod exec ftn-chat -n obelisk --port 17860:7860
    actl pod exec globo-dreamer -n obelisk --port 17865:8765
These commands must open a shell: actl skips port forwarding for one-shot
commands supplied after --. No public Gradio tunnel is used.

If chat is not already running, in the ftn-chat pod shell:
    cd /home/dev/workspace
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u app.py --checkpoints /home/dev/ftn_chat_state/checkpoints_bayes_qa_ic --no-thinking
Check GPU availability before restarting. The initial site session started
this service on the idle GPU 1; log: /home/dev/ftn_chat_state/site_chat.log.
The existing Globo report server is reused without changing its experiments.

The homepage presents the main narrative followed by a compact three-area
innovation directory. The live chat is on the Language models project page,
with expanded controls and no automatic prompt. It uses the input-conditioned
QA checkpoint, with thinking off and direct input routing on by default.
Existing training covers base_ftn, explorer, and rl_mt10; website-specific FAQ
training is planned.

CONTENT AND UPKEEP
index.html: navigation, shell, metadata.
app.js: original detail pages and live Gradio API integration.
innovations.js: revised three-area navigation and proposal media galleries.
assets/posters/: still frames extracted from the supplied videos.
style.css: responsive black / blue / gray presentation.
server.py: localhost server, artifact mapping, live-service proxy.
report.css / report.js: report styling, adaptive iframe height, media behavior.
site outline.txt: original user outline, preserved.

SiT: the complete existing memory_budget/artifacts/report.html is served with
matching styling. All original sections and relative artifact links remain.
Globo: /globo/continuous/ proxies the existing continuous live report; use its
Refresh report button for current values without interrupting video playback
on a timer. It requires the tunnel to remain available.
Robotics: uses artifacts/48x48 and existing misc GIFs. The outline's
train_mt10_segments.py does not exist in this checkout; content follows
train_mt10_transitions_bc_par.py, viz_mt10_segments.py, and
train_eval_mt10_plan.py. Unverified held-out splits are not asserted.
The outline's exploration/ is explorer/ in this repository.
Resources & team is intentionally empty, as requested.

The website has no persistent conversation storage. It sends live prompts to
the existing research app, whose global model/mask state is shared. This is a
local research demo, not a multi-user inference service.

REVISED OUTLINE (October 10)
The front page retains the main narrative and links to the full proposal.
Innovations has three pages: Exploration; Continual Learning: FTN; and
Representation and Planning. The text below OLD OUTLINE is retained as detail
pages, rather than driving primary navigation.

The principal galleries read presentation/proposal assets/ in place, including
files ignored by Git. Keep that folder available when moving the site. Local
MP4 and WebM responses support byte ranges for seeking. Video poster frames
are stored under site/assets/posters/. Videos play only on request.
The supplied SiT directory is empty; its existing repository report and samples
are retained. Anonymous recordings have neutral captions rather than inferred
training splits or task-success claims. Crafter is an additional collapsed
gallery. Resources & team remains reachable at #/resources but is not in the
new primary navigation.
