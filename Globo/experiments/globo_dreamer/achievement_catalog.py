"""Evaluation event names shared by logging and the monitor."""
LABELS = {
    'kill_wolf': 'Kill a wolf', 'kill_cow': 'Kill a cow',
    'dig_block': 'Dig a block', 'place_block': 'Place a block',
    'lay_egg': 'Lay an egg', 'drink_water': 'Drink water',
    'egg_mature': 'Egg survives to reproductive score',
    'harvest_potatoes': 'Harvest mature potatoes', 'harvest_papaya': 'Harvest papaya',
    'eat_thorn_fruit': 'Eat thorn fruit', 'eat_dawn_cactus': 'Eat dawn cactus fruit',
    'heal_goldcap': 'Recover health with goldcap', 'coffee_energy': 'Restore energy with coffee',
    'sleep': 'Sleep successfully', 'rest_energy': 'Recover energy while asleep',
    'rest_health': 'Recover health while asleep', 'kill_other': 'Kill another animal species',
    'eat_food': 'Consume food', 'new_food': 'Discover a food type',
    'new_resource': 'Collect a new resource type', 'new_species': 'Hunt a new species',
    'climb_5m': 'Climb 5 m above life start', 'climb_10m': 'Climb 10 m above life start',
    'climb_20m': 'Climb 20 m above life start',
    'cross_water': 'Travel 5 m through water and exit alive',
    'new_biome': 'Reach a new biome',
    'survive_predator': 'Survive 30 s after a predator hit without another hit',
}
KINDS = tuple(LABELS)
