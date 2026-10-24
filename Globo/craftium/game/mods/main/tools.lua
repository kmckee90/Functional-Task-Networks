-- https://github.com/minetest/minetest/blob/master/doc/lua_api.md#items
-- The hand
minetest.register_item(
    ":", {
        type = "none",
        wield_image = "wieldhand.png",
        wield_scale = {x = 1, y = 1, z = 2.5},
        liquids_pointable = true,
        range = 2.3,
        tool_capabilities = {
            full_punch_interval = 0.9,
            max_drop_level = 1,
            groupcaps = {
                choppy = {times = {[3] = minimal.hand_chop}, uses = 0, maxlevel = 1},
                crumbly = {times = {[3] = minimal.hand_crum}, uses = 0, maxlevel = 1},
                snappy = {times = {[3] = minimal.hand_snap}, uses = 0, maxlevel = 1},
                oddly_breakable_by_hand = {
                    times = {
                        [1] = minimal.hand_crum * minimal.t_scale1,
                        [2] = minimal.hand_crum * minimal.t_scale2,
                        [3] = minimal.hand_crum,
                    },
                    uses = 0,
                },
            },
            damage_groups = {fleshy = 5},
        },
    }
)

--[[
minetest.register_item(":", {
	type = "none",
	wield_image = "wieldhand.png",
	wield_scale = {x = 1, y = 1, z = 2.5},
	tool_capabilities = {
		full_punch_interval = 0.9,
		max_drop_level = 0,
	},
	})
]] --

minetest.register_tool(
    "main:rock", {
        description = "Rock",
        inventory_image = "default_stone.png",
        tool_capabilities = {
            full_punch_interval = 1.0,
            max_drop_level = 0,
            groupcaps = {choppy = {times = {[2] = 2.00, [3] = 1.00}, uses = 20, maxlevel = 1}},
        },
    }
)

minetest.register_tool(
    "main:tool_bones", {
        description = "Bone Tool",
        inventory_image = "basenodes_tool_bones.png",
        tool_capabilities = {
            full_punch_interval = 1.0,
            max_drop_level = 1,
            groupcaps = {
                crumbly = {times = {[1] = 2.00, [2] = 1.00}, uses = 20, maxlevel = 1},
                snappy = {times = {[2] = 3.00, [3] = 2.40}, uses = 20, maxlevel = 1},
            },
            damage_groups = {fleshy = 100},
        },
    }
)

minetest.register_tool(
    "main:lay_egg_disabled", {
        description = "Not enough hunger or energy to lay an egg.",
        inventory_image = "lay_egg_disabled.png",
        on_use = function(itemstack, user)
            globo.chat_send_player(user:get_player_name(), "Not enough hunger or energy to lay an egg.")
            return itemstack
        end,
    }
)

minetest.register_craftitem(
    "main:lay_egg", {
        description = "Lay an egg",
        inventory_image = "lay_egg.png",

        on_use = function(itemstack, user, pointed_thing)
            globo.lay_egg(user)
            -- Consuming the item is handled in `globo.lay_egg_item_update`.
            return itemstack
        end,
    }
)

minetest.register_tool(
    "main:sleep", {
        description = "Go to sleep",
        inventory_image = "sleep.png",

        on_use = function(itemstack, user, pointed_thing)
            globo.player_sleep(user)
            return itemstack
        end,
    }
)

-- Match standalone play: either use button activates abilities, even in air.
for _, name in ipairs({"main:lay_egg", "main:lay_egg_disabled", "main:sleep"}) do
    local use = minetest.registered_items[name].on_use
    minetest.override_item(name, {on_place = use, on_secondary_use = use})
end
