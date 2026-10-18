
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
