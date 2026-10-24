"""Fixed, bounded encodings of the numeric HUD, inventory and status effects."""
import numpy as np

EFFECTS = ('Food Poisoning', 'Fungal Infection', 'Dust Fever', 'Drunk', 'Hangover',
           'Intestinal Parasites', 'Tiku High', 'Neurotoxicity', 'Hepatotoxicity',
           'Photosensitivity', 'Meta-Stim')
NAMES = ('health', 'hunger', 'thirst', 'energy', 'breath', 'internal_temperature',
         'external_temperature', 'energy_capacity', 'score', 'heal_rate',
         'hunger_rate', 'thirst_rate', 'recovery_rate') + tuple(
         f'slot_{i}_quantity' for i in range(1,10)) + tuple(
         f'slot_{i}_selected' for i in range(1,10)) + EFFECTS + ('other_effect',)

def encode(s):
    def ratio(key, maximum): return float(s.get(key,0))/max(1,float(maximum))
    def signed(key, scale): return .5+.5*np.tanh(float(s.get(key,0))/scale)
    slots=list(s.get('slots') or []);caps=list(s.get('slot_max') or [])
    effects={e[0] for e in (s.get('effects') or []) if e}
    v=[ratio('hp',s.get('hp_max',1000)),ratio('hunger',1000),ratio('thirst',1000),
       ratio('energy',s.get('energy_max',1000)),ratio('breath',s.get('breath_max',10)),
       (float(s.get('temperature',37))-20)/30,
       (float(s.get('temperature_external',20))+50)/150,
       ratio('energy_max',10000),max(0,float(s.get('score',0)))/(1000+max(0,float(s.get('score',0)))),
       *[signed(k,100) for k in ('heal_rate','hunger_rate','thirst_rate','recovery_rate')],
       *[(slots[i]/max(1,caps[i])) if i<len(slots) and i<len(caps) else 0 for i in range(9)],
       *[float(s.get('wield')==i) for i in range(1,10)],
       *[float(e in effects) for e in EFFECTS],float(bool(effects-set(EFFECTS)))]
    return np.nan_to_num(np.clip(v,0,1),nan=0,posinf=1,neginf=0).astype(np.float32)
