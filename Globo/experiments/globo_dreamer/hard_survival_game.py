"""Patch only a newly copied engine game; never mutate shared source assets."""
import json
import re
from pathlib import Path

def apply(game):
    game=Path(game)
    animal=game/'mods/animals/init.lua';text=animal.read_text()
    start=text.index('-- Cap mobkit health (also the regeneration ceiling)')
    end=text.index('\n---\n-- Food Web',start)
    animal.write_text(text[:start]+text[end:])
    health=game/'mods/health/init.lua';text=health.read_text()
    for name,rate in [('thirst','t_rate'),('hunger','hun_rate')]:
        old=f'{name}1 = {name} + {rate}'
        assert text.count(old)==1
        text=text.replace(old,f'{name}1 = {name} + ({rate} < 0 and 4 * {rate} or {rate})')
        # HUD feedback reports the actual effective periodic loss rate.
        old=f'meta:set_int("{name}_rate", {rate})'
        assert text.count(old)==1
        text=text.replace(old,f'meta:set_int("{name}_rate", {rate} < 0 and 4 * {rate} or {rate})')
    start=text.index('-- bonus/malus from thirst');end=text.index('if temperature',start)
    section=text[start:end]
    section,n=re.subn(r'h_rate = h_rate - (\d+)',lambda m:f'h_rate = h_rate - {4*int(m[1])}',section)
    assert n==6,n
    text=text[:start]+section+text[end:];health.write_text(text)
    actions=game/'mods/health/on_actions.lua';text=actions.read_text()
    text,n=re.subn(r'(thirst|hunger) = \1 - (\d+)',lambda m:f'{m[1]} = {m[1]} - {4*int(m[2])}',text)
    assert n==6,n
    actions.write_text(text)
    effects=game/'mods/health/health_effects.lua';text=effects.read_text()
    for name,amount in [('thirst','rant'),('hunger','ranh')]:
        old=f'{name} = {name} - {amount}'
        assert text.count(old)==1
        text=text.replace(old,old+' * 4')
    effects.write_text(text)
    (game/'survival_variant.json').write_text(json.dumps(dict(
        animal_hp='original species definitions, cap removed',
        food_water_depletion_multiplier=4,resource_shortage_damage_multiplier=4,
        resource_gains_multiplier=1,other_damage_multiplier=1),indent=2))
