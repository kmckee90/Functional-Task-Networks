"""Pod-only still extraction from the approved-for-review annotated montage."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from annotated_montage_preview import ROOT,frames,text_box,arrow,RED,FONT

OUT=ROOT/'annotated_frames'
VIDEO=ROOT/'nascent-annotated-preview-v3.mp4'

def at(seconds):
    index=round(seconds*24)
    return next(frames(VIDEO,index,index+1))[1].to_image()

def main():
    OUT.mkdir(exist_ok=True)
    base=next(frames(ROOT/'ftn-m2e-montage-smooth-climbing-preview.mp4',198,199))[1].to_image()
    im=Image.new('RGB',(1620,900),'#111820');im.paste(base,(20,100));d=ImageDraw.Draw(im)
    d.text((20,24),'Observation, exploration goal, and active neural state',
           font=ImageFont.truetype(FONT,34),fill='white')
    for box in [(23,170,657,809),(661,240,977,400),(661,489,977,809)]:d.rectangle(box,outline=RED,width=5)
    text_box(im,(1040,145),['256×256 frames downsample','to RGB 64×64 for input.'],27)
    # Route the observation arrow above the frame, keeping both goal and mask clear.
    d.line([(1028,160),(1005,160),(1005,85),(340,85),(340,150)],fill=RED,width=5)
    arrow(im,(340,150),(340,170))
    text_box(im,(1040,345),['A compressed, unusual observation','in memory as the goal state'],27)
    arrow(im,(1027,365),(979,325))
    text_box(im,(1040,575),["Agent's active brain state (FTN):",'1 white pixel = 1 active neuron.','Purple: Recent trace of activity.'],27)
    arrow(im,(1027,630),(979,630))
    im.save(OUT/'01_observation_goal_and_ftn.png')
    at(129).save(OUT/'02_punched_wolf.png')
    prior,next_state=at(183.5),at(186.5)
    pair=Image.new('RGB',(2080,800),'#111820');pair.paste(prior,(20,60));pair.paste(next_state,(1100,60))
    d=ImageDraw.Draw(pair);font=ImageFont.truetype(FONT,28)
    d.text((20,16),'Prior goal and neural state',font=font,fill='white')
    d.text((1100,16),'Next goal and neural state',font=font,fill='white')
    # Short, thick transition arrow between the unmodified annotated panels.
    d.polygon([(995,399),(1047,399),(1047,379),(1085,415),(1047,451),(1047,431),(995,431)],fill=RED)
    pair.save(OUT/'03_goal_and_brain_state_change.png')
    at(232).save(OUT/'04_jump_off_wolf.png')
    at(255).save(OUT/'05_collecting_nuts.png')
    (OUT/'frames.json').write_text(json.dumps(dict(source=VIDEO.name,
        images={'01_observation_goal_and_ftn.png':'First wandering frame with combined annotations',
                '02_punched_wolf.png':129,'03_goal_and_brain_state_change.png':[183.5,186.5],
                '04_jump_off_wolf.png':232,'05_collecting_nuts.png':255}),indent=2))
    print('Saved five PNGs to',OUT,flush=True)

if __name__=='__main__':main()
