"""Build the incremental GitHub Pages gallery on the pod."""
from pathlib import Path
from urllib.parse import unquote
import shutil
import av
from bs4 import BeautifulSoup

root=Path('/home/dev/globo_dreamer_state/found_footage')
out=root/'gallery_build'
out.mkdir(exist_ok=True)
(out/'media').mkdir(exist_ok=True)
(out/'thumbnails').mkdir(exist_ok=True)
source=Path(__file__).with_name('clip-gallery.js').read_text()
markup=source.split('()=>`',1)[1].rsplit('`;',1)[0]
gallery=BeautifulSoup(markup,'html.parser')
clip_count=len(gallery.select('video'))
times={'cow fly':15,'defend':12,'plants':12,'place_block':14,'zoom':9}
for video in gallery.select('video'):
    name=Path(unquote(video['src'])).name
    src=root/'input'/name
    with av.open(str(src)) as c:
        stream=c.streams.video[0]
        target=int(times.get(src.stem,4)*float(stream.average_rate))
        for i,frame in enumerate(c.decode(video=0)):
            if i>=target:
                break
        poster=Path(unquote(video['poster'])).name
        frame.to_image().save(out/'thumbnails'/poster,quality=90)
    shutil.copyfile(src,out/'media'/name)
    shutil.copyfile(out/'thumbnails'/poster,out/'media'/poster)
    video['src']='media/'+name.replace(' ','%20')
    video['poster']='media/'+poster
gallery.select_one('a')['href']='index.html'
index=BeautifulSoup((root/'deployed-index.html').read_text(),'html.parser')
if not index.select_one('a[href="clip-gallery.html"]'):
    link=index.new_tag('a',href='clip-gallery.html',style='font-size:24px;line-height:1.4');link.string='More agent videos'
    p=index.new_tag('p');p.append(link)
    index.select_one('.annotation-thumbnails').insert_after(p)
(out/'index.html').write_text(str(index))
index.title.string='Clip Gallery — Nascent AGI'
main=index.select_one('main');main.clear();main.append(gallery)
(out/'clip-gallery.html').write_text(str(index))
assert len(index.select('main video'))==clip_count
assert not any('montage' in v['src'] or 'found-footage' in v['src'] for v in index.select('main video'))
print(f'Built gallery: {len(index.select("main video"))} clips, posters, and proposal link.')
