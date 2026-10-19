globo = {}

local default_path = minetest.get_modpath("main")

dofile(default_path .. "/settingstypes.lua") -- must be loaded first so constants are available.

dofile(default_path .. "/utils.lua")
dofile(default_path .. "/tools.lua")
dofile(default_path .. "/player.lua")
dofile(default_path .. "/nodes.lua")
dofile(default_path .. "/biomes.lua")
dofile(default_path .. "/decorations.lua")
dofile(default_path .. "/mapgen.lua")
dofile(default_path .. "/commands.lua")
dofile(default_path .. "/inferno.lua")
dofile(default_path .. "/conveyor.lua")
dofile(default_path .. "/tangler.lua")
dofile(default_path .. "/effects.lua")
-- dofile(default_path.."/sounds.lua")
