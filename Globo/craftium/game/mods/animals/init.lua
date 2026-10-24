---@diagnostic disable-next-line: lowercase-global
animals = {}

-- Internationalization
animals.S = minetest.get_translator("animals")

local path = minetest.get_modpath(minetest.get_current_modname())

dofile(path .. "/utils.lua")
dofile(path .. "/animal_data.lua")
dofile(path .. "/crafts.lua")
dofile(path .. "/api_capture.lua")
dofile(path .. "/api.lua")

dofile(path .. "/impethu.lua")
dofile(path .. "/kubwakubwa.lua")
dofile(path .. "/darkasthaan.lua")

dofile(path .. "/pegasun.lua")
dofile(path .. "/sneachan.lua")

dofile(path .. "/badger.lua")
dofile(path .. "/chaoshawk.lua")
dofile(path .. "/cow.lua")
dofile(path .. "/frostmephit.lua")
dofile(path .. "/giantmoth.lua")
dofile(path .. "/lavaox.lua")
dofile(path .. "/mongoose.lua")
dofile(path .. "/ogre.lua")
dofile(path .. "/sandslug.lua")
dofile(path .. "/snagon.lua")
dofile(path .. "/skipfungus.lua")
dofile(path .. "/wolf.lua")
dofile(path .. "/puma.lua")
dofile(path .. "/gazelle.lua")
dofile(path .. "/firepuffin.lua")
-- dofile(path.."/test.lua")

-- Cap mobkit health (also the regeneration ceiling) for every animal species.
-- Reapply after activation because saved entities can restore older HP values.
for name, def in pairs(minetest.registered_entities) do
    if name:sub(1, 8) == "animals:" and type(def.max_hp) == "number" then
        def.max_hp = math.min(def.max_hp, 20)
        local activate = def.on_activate
        def.on_activate = function(self, ...)
            if activate then activate(self, ...) end
            self.max_hp = math.min(self.max_hp, 20)
            self.hp = math.min(self.hp or self.max_hp, self.max_hp)
        end
    end
end

---
-- Food Web

--[[
The aim is for mobs to be permanent populations, rather than spawning "ex nihilo".
Therefore most are small animals with small ranges. Have actual food webs that keep them alive.

Ocean:
"plankton (water)" -> gundu -> sarkamos


Caves:
"invisibly small stuff" -> impethu -> kubwakubwa/darkasthaan-> darkasthaan



Land:
plants/dirt/sneachan -> pegasun














]]
