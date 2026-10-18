function notifySize(){parent.postMessage({type:'report-height',height:document.documentElement.scrollHeight},location.origin)}
new ResizeObserver(notifySize).observe(document.body);
window.addEventListener('load',notifySize);
document.querySelectorAll('img').forEach(img=>{img.loading='lazy';img.addEventListener('click',()=>parent.postMessage({type:'report-image',src:img.src,alt:img.alt||'Experiment artifact'},location.origin))});
document.querySelectorAll('video').forEach(video=>{video.preload='metadata';video.setAttribute('playsinline','')});
document.querySelectorAll('a').forEach(a=>{a.target='_blank';a.rel='noopener'});
