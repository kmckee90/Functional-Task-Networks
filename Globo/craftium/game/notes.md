TODO:
- Gym HUD (Each thing should have its own float)

- Boom griffin- You can find its nest by watching where it flys back to. Flies down and eats an occasional cow. Has bombs in its nest

- Red Ivy: Will grow up alongside anything that is near it. You can climb up it like some of the canes. Will grow up trees or cliffs.
- Night Lily: pretty flower
- Glowfly: come out randomly at night. They fly around. Are attracted to Night Lilies. Emit light. They will fly away and disappear during the day
- locust swarm
- prairie dogs, build tunnels underground
- alpine stalker: quick jumpy thing that doesn't take much damage. eats eggs and players. lives in the high country
- something that throws you high in the air
- worm that lives in a tunnel. will come out to hunt. dies if it can't get back in its tunnel
- fountain that clearly has water in it but very dangerous somehow. so the agent must figure out how to nuetralize the danger. burn it or something

- fix animal movement
- creature that migrates
- make there be a minimal amount of light around the player at night to simulate feeling around

- spiders lay webs that slow creatures down. The webs are very flammable
- something you eat that makes your vision white out
- night drakness should depend on cloud level

hide useful or dangerous things in the cana fields

some way to kill pumas and wolves
punch node it blows everything in front of you away

Something that shoots. Anything within x spaces it will shoot at.
Snagon
- something that gives a lot of water but makes you really hungry, that hurts you, that saps your energy, etc

Puffbush: you don't get hurt if you fall in this.
Something that grabs a section of blocks and throws them.  They come down as blocks and hurt what's around.
some animal that shoots fire. maybe only when hurt

Leaves should grow back

- something blows up and creates a ball of blocks
- make thing level randomness( things do differnet amounts of damage, give variable amounts of food, water, etc)
- system level randomness. (dryer, no cows, etc)

- small ponds

- make the fire puffin emit light (not straight forward since it is an entity)
- make puffin seek ponds


- make lava ox heat a larger area

- make breeding
    - make pregnant a timer rather than a bool
- temperature should affect you
    - hides that keep you warm
    - eggs need to be a certain temperature to hatch

- fix tangler to be more intersting

- make seasons shorter?

- tell you the direction that damage came from
- dropped items should burn in fire
- flowing water should push you
- player should be no different than other animals to the mobs
- plants spread
- make animals sleep
- animals run from fire
- adjust the exile energy system to have a max energy and regain your current energy when you are still


CHECK:
- wolves are often getting killed by cows
- behaviors seem wonky. Do they need more idle time
- make sure animals can be in equilibrium
- is it better to just have the infinite map?
- what does armor do?
- do animals get hurt by thorns?



BUGS:
- mobs now able to punch through walls
- sneachens aren't eating?
- wolf energy not decreasing
- health effects are being cleared instantly
- will drift at last speed once at energy 0
2024-01-03 07:50:40: ERROR[Server]: suspiciously large amount of objects detected: 264 in (-22,0,-20); removing all of them.



- Temperature
    you start to die if not kept in a certain temperature range. It is colder at night. Certain parts of the map are colder. Some days are colder than others. If you are too hot you will lose energy faster and get thirsty faster.
    Energy expenditure raises your core temp
    The air has an ambient temp
    Your core temp goes toward this temperature over time


ANNOYING MINETEST QUIRKS:
- animals must be created as nodes (ie eggs or whatever) not as just entities if you want them as decorations
- on_construct is only called if that last field of the decoration row is 1




IDEAS:
- currant in water
- eat something that makes you glow
- bouyancy
    - need to be able to create some kind of raft
- Chaos Hawk
- badger
- Ogre
- Frost mephit
- Sand Slug
- Block that moves through the air at a particular elevation until it hits something.  Anything on top of it will ride along. Spawns randomly
- 
- tar
Motion detector
Apple Tree: grows slowly: drops apples occasionally. Has a chance of dropping wood when broken.
Fountain: Will spray water over the nearby blocks.
Slug: takes up a block. Will move in a direction pushed. Picks up the first block in its path and becomes this when it comes to a stop.
Willow wisp
Throw rocks. Do a small amount of damage
- migrations?
What would require more planning?




Later:
- Wind that affects the direction that fire spreads
- Do we want to be able to dig clay or cut trees?
- need a way to throw things away
- potatoes only spread under the right conditions
- Kill grib_weed if it is too cold for too long
- sun berry: plant that glows when it is in dry soil.
- Each block type has a chance of becoming an item or just disappearing when you break it.




Changes to Minetest
----
Allow you to send other commands
Allow you to generate the whole map and reject those that don't meet some criteria
The way lava and water flow
Allow other color lights


Notes
----
What can we take from existing mods?


Stats
-----
Seeds:
5:  -4