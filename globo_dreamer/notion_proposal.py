"""Prepare a Notion-native copy of the offline proposal; run only on the pod."""
import json,re,sys,urllib.request,uuid
from pathlib import Path
from bs4 import BeautifulSoup,Comment,NavigableString

ROOT=Path('/home/dev/globo_dreamer_state/notion_proposal')
SITE=Path('/home/dev/globo_dreamer_state/export_gallery_20261016/proposal_offline')
ROOT.mkdir(exist_ok=True)
assets={}
def asset(src):
    if not src.startswith('media/'):return src
    if src not in assets:
        n=len(assets);p=SITE/src
        assets[src]={'key':f'ASSET{n:03}', 'path':str(p),'filename':p.name}
    return '@@'+assets[src]['key']+'@@'
def clean(s):return re.sub(r'\s+',' ',s).strip()
def inline(n):
    if isinstance(n,Comment):return ''
    if isinstance(n,NavigableString):return re.sub(r'\s+',' ',str(n))
    v=''.join(inline(c) for c in n.children)
    if n.name in ('b','strong'):return '**'+v+'**'
    if n.name in ('i','em'):return '*'+v+'*'
    if n.name=='br':return '<br>'
    if n.name=='a':
        h=n.get('href','')
        if n.find('img'):return block(n.find('img')).strip()
        if h.endswith('.html'):return f'<mention-page url="@@PAGE:{h}@@">{v}</mention-page>'
        return '['+v+']('+asset(h)+')' if h else v
    return v
def children(n):return ''.join(block(c) for c in n.children)
def columns(items):
    out='<columns>\n'
    for s in items:
        out+='\t<column>\n'+''.join('\t\t'+l+'\n' for l in s.strip().splitlines())+'\t</column>\n'
    return out+'</columns>\n\n'
def block(n):
    if isinstance(n,Comment):return ''
    if isinstance(n,NavigableString):return str(n) if str(n).strip() else ''
    name=n.name;cls=set(n.get('class',[]))
    if name in ('script','style','button','iframe') or cls&{'breadcrumb','project-back','project-view'}:return ''
    if name=='h1':return ''
    if name in ('h2','h3','h4','h5','h6'):return '#'*min(int(name[1]),4)+' '+inline(n).strip()+'\n\n'
    if name=='img':return '!['+n.get('alt','').replace(']','')+']('+asset(n['src'])+')\n\n'
    if name=='video':return '<video src="'+asset(n['src'])+'">'+n.get('aria-label','')+'</video>\n\n'
    if name in ('p','figcaption'):return inline(n).strip()+'\n\n'
    if 'annotation-thumbnails' in cls:
        return columns([block(a.find('img'))+a.find('span').get_text() for a in n.find_all('a',recursive=False)])
    if cls&{'result-grid','exploration-videos','ftn-videos','globo-map-grid','globo-summary-maps','llm-domain-grid'}:
        items=[block(c) for c in n.find_all(recursive=False)]
        return columns(items) if 1<len(items)<=5 else ''.join(items)
    if 'research-timeline' in cls:
        return ''.join('### '+li.select_one('.timeline-date').get_text(' ',strip=True)+'\n\n'+children(li.select_one('.timeline-content')) for li in n.find_all('li',recursive=False))
    if name in ('ul','ol'):
        out=''
        for li in n.find_all('li',recursive=False):
            text=''.join(inline(c) for c in li.children if getattr(c,'name',None) not in ('ul','ol')).strip()
            out+=('- ' if name=='ul' else '1. ')+text+'\n'
            for sub in li.find_all(['ul','ol'],recursive=False):out+=''.join('\t'+l+'\n' for l in block(sub).strip().splitlines())
        return out+'\n'
    if name=='table':
        out='<table header-row="true">\n'
        for tr in n.find_all('tr'):
            cells=[]
            for td in tr.find_all(['td','th'],recursive=False):
                if td.find('li'):s='<br>'.join(inline(li).strip() for li in td.find_all('li'))
                else:s=inline(td).strip()
                cells.append('\t\t<td>'+s+'</td>\n')
            out+='\t<tr>\n'+''.join(cells)+'\t</tr>\n'
        return out+'</table>\n\n'
    if name=='hr':return '---\n\n'
    if name=='a':return inline(n)+'\n\n'
    if name=='figure':return children(n)
    if cls&{'timeline-date','sit-domain-label'}:return '**'+n.get_text(' ',strip=True)+'**\n\n'
    if name in ('div','section','article','main'):return children(n)
    return inline(n)

def prepare():
    files=['index.html','projects.html','projects-globo.html','projects-robotics.html','projects-diffusion.html','projects-chatftn.html','projects-classification.html','projects-maze.html','projects-locomotion.html','projects-planning.html']
    pages=[]
    for f in files:
        soup=BeautifulSoup((SITE/f).read_text(),'html.parser')
        main=soup.select_one('main')
        pages.append({'file':f,'title':main.find('h1').get_text(),'content':children(main)})
    # SVG charts become PNGs so they display consistently in Notion.
    from playwright.sync_api import sync_playwright
    import shutil
    with sync_playwright() as p:
        browser=p.chromium.launch(args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1600,'height':1200},device_scale_factor=1)
        for a in assets.values():
            src=Path(a['path'])
            if src.suffix=='.svg':
                page.goto(src.as_uri());el=page.locator('svg')
                target=ROOT/(src.stem+'.png');el.screenshot(path=str(target))
                a['path']=str(target);a['filename']=target.name
        browser.close()
    bundle={'pages':pages,'assets':list(assets.values())}
    (ROOT/'bundle.json').write_text(json.dumps(bundle))
    print(json.dumps({'pages':[(p['file'],len(p['content'])) for p in pages],'assets':len(assets),'large':[(a['key'],a['path']) for a in assets.values() if Path(a['path']).stat().st_size>20*1024**2]}))

def upload():
    data=json.load(sys.stdin);p=Path(data['path']);u=data['upload']
    boundary='----Notion'+uuid.uuid4().hex
    body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{p.name}"\r\nContent-Type: {u.get("content_type","application/octet-stream")}\r\n\r\n').encode()+p.read_bytes()+f'\r\n--{boundary}--\r\n'.encode()
    headers={**u['upload_headers'],'Content-Type':'multipart/form-data; boundary='+boundary}
    req=urllib.request.Request(u['upload_url'],data=body,headers=headers,method='POST')
    with urllib.request.urlopen(req,timeout=180) as r:response=json.load(r)
    (ROOT/(data['key']+'-upload.json')).write_text(json.dumps(response))
    print(json.dumps(response))

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='upload':upload()
