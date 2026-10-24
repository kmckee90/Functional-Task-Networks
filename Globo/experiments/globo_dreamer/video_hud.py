"""Display-only, frame-aligned bodily state; never added to policy pixels."""
import numpy as np
from PIL import ImageDraw, ImageFont

HEIGHT=322
FONT=ImageFont.load_default(size=11)

def values(record):
    slots=list(record.get('slots') or [])[:4]
    slots += [float('nan')]*(4-len(slots))
    return np.asarray([record.get(k,float('nan')) for k in
        ('hp','hunger','thirst','energy','temperature','breath','wield')]+slots,dtype=np.float32)

def draw(canvas, state):
    d=ImageDraw.Draw(canvas)
    d.rectangle((0,280,canvas.width,HEIGHT),fill='#10151d')
    if state is None:
        d.text((6,283),'HUD unavailable for this frame',font=FONT,fill='#e5eaf0')
        return
    def n(i):return f'{state[i]:g}' if np.isfinite(state[i]) else '?'
    d.text((6,281),f'HP {n(0)}   Hunger {n(1)}   Thirst {n(2)}',font=FONT,fill='#e5eaf0')
    d.text((6,294),f'Energy {n(3)}   Temp {n(4)} C   Air {n(5)}',font=FONT,fill='#e5eaf0')
    d.text((6,307),'Slots:',font=FONT,fill='#e5eaf0')
    for i in range(4):
        x=43+i*72
        selected=state[6]==i+1
        if selected:d.rounded_rectangle((x-2,306,x+64,320),radius=2,fill='#d9b341')
        d.text((x,307),f'{i+1}: {n(7+i)}',font=FONT,fill='#10151d' if selected else '#e5eaf0')
