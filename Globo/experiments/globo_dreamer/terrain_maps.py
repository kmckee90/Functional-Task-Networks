"""Comparable seed-wide terrain rasters and evaluation-only visitation overlays."""
import math
from PIL import Image, ImageDraw
from .phase2 import SHARED,read,atomic

COLORS=(('water',(60,118,161)),('snow',(218,227,228)),('ice',(158,205,224)),
 ('sand',(191,171,114)),('grass',(91,130,70)),('dirt',(131,104,75)),('stone',(119,124,126)),
 ('rock',(119,124,126)),('leaves',(53,98,53)),('tree',(100,87,57)),('wood',(136,110,69)),
 ('lava',(205,85,43)),('desert',(186,150,102)))

def bounds(states,previous=None):
    points=[]
    for s in states:
        points.extend((int(k.split(',')[0])*4,int(k.split(',')[1])*4) for k in s.get('counts',{}))
        if s.get('position'):points.append((s['position'][0],s['position'][2]))
    if not points:return previous
    xs,zs=zip(*points)
    b=[math.floor((min(xs)-32)/64)*64,math.ceil((max(xs)+36)/64)*64,
       math.floor((min(zs)-32)/64)*64,math.ceil((max(zs)+36)/64)*64]
    if previous:b=[min(b[0],previous[0]),max(b[1],previous[1]),min(b[2],previous[2]),max(b[3],previous[3])]
    return b

def raster(seed,b):
    x0,x1,z0,z1=b
    # Keep output resolution bounded even on very long runs; cached source tiles retain 2m pixels.
    scale=min(4.,640/max(x1-x0,z1-z0));w=max(1,round((x1-x0)*scale));h=max(1,round((z1-z0)*scale))
    im=Image.new('RGB',(w,h),'#303642');done=0;tiles=[]
    for tz in range(z0//64,z1//64):
        for tx in range(x0//64,x1//64):
            tiles.append([tx,tz]);tile=read(SHARED/'terrain'/str(seed)/f'tile_{tx}_{tz}.json')
            if not tile:continue
            done+=1;small=Image.new('RGB',(32,32));pix=[]
            for height,kind in tile['pixels']:
                if kind==0:pix.append((48,54,66));continue
                name=tile['names'][int(kind)-1];base=next((color for token,color in COLORS if token in name),(113,127,79))
                shade=max(.7,min(1.15,.92+height*.002))
                pix.append(tuple(round(min(255,c*shade)) for c in base))
            small.putdata(pix)
            # North (+z) at top, east (+x) right.
            small=small.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            left=round((tx*64-x0)*scale);top=round((z1-(tz+1)*64)*scale)
            right=round(((tx+1)*64-x0)*scale);bottom=round((z1-tz*64)*scale)
            im.paste(small.resize((max(1,right-left),max(1,bottom-top)),Image.Resampling.NEAREST),(left,top))
    return im,tiles,done,scale

def publish(seed,trials):
    states=[read(SHARED/'runs'/t['name']/'live/visitation.json') for t in trials]
    out=SHARED/'monitor/maps';out.mkdir(parents=True,exist_ok=True)
    old=read(out/f'seed{seed}.json');b=bounds(states,old.get('bounds'))
    if b is None:
        x,y,z=map(float,trials[0]['spawn'].split(','));b=bounds([dict(position=[x,y,z])])
    terrain,tiles,done,scale=raster(seed,b)
    visited={(int(k.split(',')[0])//16,int(k.split(',')[1])//16) for s in states for k in s.get('counts',{})}
    tiles.sort(key=lambda t:(tuple(t) not in visited,abs(t[0])+abs(t[1])))
    atomic(SHARED/'terrain'/str(seed)/'request.json',dict(tiles=tiles))
    peak=max([1]+[v for s in states for v in s.get('counts',{}).values()]);x0,x1,z0,z1=b
    def xy(x,z):return ((x-x0)*scale,(z1-z)*scale)
    for t,s in zip(trials,states):
        layer=Image.new('RGBA',terrain.size);draw=ImageDraw.Draw(layer)
        for key,count in s.get('counts',{}).items():
            x,z=(int(v)*4 for v in key.split(','));a=math.log1p(count)/math.log1p(peak)
            p=xy(x,z+4);q=xy(x+4,z)
            draw.rectangle((*p,*q),fill=(255,round(220-170*a),40,round(80+150*a)))
        im=Image.alpha_composite(terrain.convert('RGBA'),layer);draw=ImageDraw.Draw(im)
        last=None
        for _,x,z,life in s.get('recent',[]):
            p=xy(x,z)
            if last and last[0]==life:draw.line((last[1],p),fill='#f9f7df',width=1)
            last=(life,p)
        if s.get('spawn'):
            x,_,z=s['spawn'];px,py=xy(x,z);draw.rectangle((px-3,py-3,px+3,py+3),outline='#70cfff',width=2)
        if s.get('position'):
            x,_,z=s['position'];px,py=xy(x,z);yaw=s.get('yaw',0)
            draw.ellipse((px-4,py-4,px+4,py+4),fill='white',outline='black')
            draw.line((px,py,px-12*math.sin(yaw),py-12*math.cos(yaw)),fill='white',width=2)
        canvas=Image.new('RGB',(im.width+80,im.height+65),'#171d27');canvas.paste(im,(50,20));d=ImageDraw.Draw(canvas)
        d.text((50,3),f'North (+Z) up - seed {seed}',fill='white')
        for i in range(3):
            x=x0+(x1-x0)*i/2;z=z1-(z1-z0)*i/2
            d.text((max(0,50+i*im.width/2-15),im.height+23),f'{x:g}',fill='white')
            d.text((2,17+i*im.height/2),f'{z:g}',fill='white')
        d.text((50,im.height+43),'X (metres)     Z (metres) on left',fill='white')
        path=out/f'{t["name"]}.png';tmp=path.with_suffix('.tmp.png');canvas.save(tmp);tmp.replace(path)
    info=dict(bounds=b,cell_size=4,terrain_resolution=2,max_visits=peak,tiles_ready=done,tiles_requested=len(tiles),
        orientation='north (+z) up',steps={t['name']:s.get('last_step') for t,s in zip(trials,states)})
    atomic(out/f'seed{seed}.json',info);return info
