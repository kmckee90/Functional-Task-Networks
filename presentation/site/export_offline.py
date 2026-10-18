"""Export rendered proposal/project pages on the pod, then validate file:// use.

Expose the live site to the pod through an SSH reverse tunnel on port 18081.
All generated output stays outside source sync until explicitly downloaded.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import urllib.parse
import urllib.request
import zipfile

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

BASE = os.environ.get('PROPOSAL_EXPORT_BASE', 'http://127.0.0.1:18081')
ONLINE = 'https://klmckee-lenovo-thinkbook-15p-1.taila1eba.ts.net'
OUT = Path(os.environ.get('PROPOSAL_EXPORT_OUT', '/home/dev/globo_dreamer_state/proposal_githubio'))
OUT.mkdir(exist_ok=True)
(OUT / 'media').mkdir(exist_ok=True)
assets = {}


def bundle(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme and parsed.netloc != urllib.parse.urlsplit(BASE).netloc:
        return url
    if parsed.path.startswith(('/globo/', '/chat/', '/reports/')):
        return ONLINE + parsed.path + ('?' + parsed.query if parsed.query else '')
    path = urllib.parse.unquote(parsed.path)
    key = parsed.path
    if key not in assets:
        name = re.sub(r'[^A-Za-z0-9._-]', '_', Path(path).name)
        dest = 'media/' + hashlib.sha256(key.encode()).hexdigest()[:12] + '-' + name
        request_url = BASE + urllib.parse.quote(path, safe='/')
        with urllib.request.urlopen(request_url, timeout=120) as response:
            with (OUT / dest).open('wb') as target:
                shutil.copyfileobj(response, target)
        assets[key] = dest
    return assets[key] + ('#' + parsed.fragment if parsed.fragment else '')


def filename(route):
    return 'index.html' if route == '/' else route.strip('/').replace('/', '-') + '.html'


with sync_playwright() as p:
    browser = p.chromium.launch(args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    page.goto(BASE, wait_until='networkidle')
    page.wait_for_function('typeof revisedProposalText !== "undefined" && revisedProposalText !== null')
    projects = page.evaluate('researchProjects.filter(p=>p.id!=="continual-exploration").map(p=>p.id)')
    routes = ['/', '/projects'] + ['/projects/' + x for x in projects] + ['/research-plan', '/experimental-control']
    aliases = {'/theory': '/', '/notes/agents': '/projects/globo',
               '/notes/m2e': '/projects/maze', '/notes/ftn': '/projects/classification'}
    for route in routes:
        page.evaluate('(r)=>{location.hash="#"+r;route()}', route)
        if route == '/projects/globo':
            page.wait_for_function('document.querySelector("#globo-evidence")?.querySelector("img,svg")')
        page.wait_for_timeout(150)
        soup = BeautifulSoup(page.content(), 'html.parser')
        for script in soup.find_all('script'):
            script.decompose()
        for base in soup.find_all('base'):
            base.decompose()
        for element in soup.find_all(True):
            for attr in list(element.attrs):
                if attr.startswith('on'):
                    del element[attr]
            for attr in ('src', 'poster', 'href'):
                url = element.get(attr)
                if not url or url.startswith(('data:', 'mailto:', 'javascript:')):
                    continue
                if url.startswith('#/'):
                    target = aliases.get(url[1:], url[1:])
                    element[attr] = filename(target) if target in routes else ONLINE + '/' + url
                elif url.startswith('#'):
                    continue
                elif url.startswith('/') or url.startswith(BASE):
                    element[attr] = bundle(url)
            if element.has_attr('style') and 'url(' in element['style']:
                element['style'] = re.sub(r'url\(["\']?(/[^)"\']+)["\']?\)',
                    lambda m: 'url("' + bundle(m[1]) + '")', element['style'])
        script = soup.new_tag('script', src='offline.js', defer=True)
        soup.head.append(script)
        (OUT / filename(route)).write_text(str(soup), encoding='utf-8')
        print('Rendered', route, flush=True)
    # Local files need no fetch(), router, applet, CDN, or live API.
    (OUT / 'offline.js').write_text('''
const toggle=document.querySelector('#menu-toggle'),nav=document.querySelector('#nav');
toggle?.addEventListener('click',()=>{const open=toggle.getAttribute('aria-expanded')!=='true';toggle.setAttribute('aria-expanded',String(open));nav.classList.toggle('open',open)});
const dialog=document.querySelector('#lightbox');
document.querySelector('#lightbox .close')?.addEventListener('click',()=>dialog.close());
let gallery=[],index=0;
function showImage(){
 const item=gallery[index];dialog.querySelector('img').src=item.href;
 dialog.querySelector('img').alt=item.querySelector('img').alt;
 dialog.querySelector('p').textContent=item.querySelector('img').alt;
 dialog.querySelector('.image-count').textContent=`${index+1} / ${gallery.length}`;
 dialog.querySelector('.lightbox-navigation').hidden=gallery.length<2;
}
function move(delta){index=(index+delta+gallery.length)%gallery.length;showImage()}
document.addEventListener('click',e=>{
 const link=e.target.closest('.annotation-thumbnails a');
 if(!link||e.ctrlKey||e.metaKey||e.shiftKey||e.altKey)return;
 e.preventDefault();gallery=[...link.closest('.annotation-thumbnails').querySelectorAll('a')];
 index=gallery.indexOf(link);showImage();dialog.showModal();
});
dialog.querySelector('.previous').onclick=()=>move(-1);
dialog.querySelector('.next').onclick=()=>move(1);
dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close()});
dialog.addEventListener('keydown',e=>{
 if(gallery.length>1&&(e.key==='ArrowLeft'||e.key==='ArrowRight')){e.preventDefault();move(e.key==='ArrowLeft'?-1:1)}
});
''')
    # Make CSS asset references relative to the copied stylesheet itself.
    for dest in list(assets.values()):
        if dest.endswith('.css'):
            path = OUT / dest
            css = path.read_text()
            css = re.sub(r'url\(["\']?(/[^)"\']+)["\']?\)',
                         lambda m: 'url("../' + bundle(m[1]) + '")', css)
            path.write_text(css)
    errors, requests = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.route('http://**/*', lambda route: (requests.append(route.request.url), route.abort()))
    page.route('https://**/*', lambda route: (requests.append(route.request.url), route.abort()))
    videos = set()
    for route in routes:
        page.goto((OUT / filename(route)).as_uri(), wait_until='load')
        assert page.locator('main h1').count() == 1, route
        assert 'Loading the' not in page.locator('main').inner_text(), route
        for image in page.locator('img').all():
            if not image.get_attribute('src'):
                continue
            image.evaluate('(img)=>{img.loading="eager"}')
            image.evaluate('(img)=>img.decode()')
        for link in page.locator('a[href]').all():
            href = link.get_attribute('href')
            if not href.startswith(('http:', 'https:', '#', 'mailto:')):
                assert (OUT / urllib.parse.unquote(href.split('#')[0])).is_file(), (route, href)
        for video in page.locator('video').all():
            src = video.get_attribute('src')
            if src in videos:
                continue
            video.evaluate('''v=>new Promise((resolve,reject)=>{
              v.onloadedmetadata=()=>resolve(v.duration);v.onerror=()=>reject(Error('Video failed: '+v.src));
              v.preload='metadata';v.load();setTimeout(()=>reject(Error('Video timeout')),30000);
            })''')
            videos.add(src)
        if route in ('/', '/projects', '/projects/globo', '/projects/diffusion'):
            page.screenshot(path=str(OUT.parent / ('offline-check-' + filename(route) + '.png')), full_page=True)
        if route == '/':
            page.locator('.annotation-thumbnails a').first.click()
            assert page.locator('#lightbox').evaluate('(d)=>d.open')
            page.locator('#lightbox .next').click()
            assert page.locator('.image-count').inner_text() == '2 / 5'
            page.keyboard.press('ArrowLeft')
            assert page.locator('.image-count').inner_text() == '1 / 5'
            page.keyboard.press('ArrowLeft')
            assert page.locator('.image-count').inner_text() == '5 / 5'
            page.locator('#lightbox img').evaluate('(i)=>i.decode()')
            page.keyboard.press('Escape')
            assert not page.locator('#lightbox').evaluate('(d)=>d.open')
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), ('mobile overflow', route)
        page.set_viewport_size({'width': 1440, 'height': 1000})
        print('Offline verified', route, flush=True)
    assert not errors, errors
    assert not requests, requests
    browser.close()

(OUT / 'README.txt').write_text('''NASCENT — OFFLINE PROPOSAL

Open index.html in a modern web browser. No server, installation, or internet
connection is needed for the proposal, project pages, images, graphs, videos,
and bundled PDFs. Keep the entire folder together when moving or sharing it.

Live monitor links and external references still point to their original
online destinations and require internet/access to those services.

This is a static snapshot of the website. Source files and training runs
are not included. Browser checks verified every page, image, and unique video
using file:// with HTTP and HTTPS requests blocked.
''')
(OUT / 'manifest.json').write_text(json.dumps(dict(routes=routes, assets=assets,
    pages=len(routes), videos=len(videos), external_requests=requests), indent=2))
archive = OUT.with_suffix('.zip')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            z.write(path, path.relative_to(OUT.parent))
print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size,
                     pages=len(routes), assets=len(assets), videos=len(videos))), flush=True)
