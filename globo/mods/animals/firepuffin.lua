----------------------------------------------------------------------
-- FirePuffin
--[[
slowly moving orb that warms the area around it moves in a direction till off the map. It blasts the ground every min or so destroying most plant life and creating embers. Stays x blocks off the ground
]] ---------------------------------------------------------------------
local get_gametime = std.minetest.get_gametime

local baseOx = {
    physical = false,
    collide_with_objects = true,
    collisionbox = {-0.45, -0.01, -0.45, 0.45, 1.39, 0.45},
    visual = "sprite",
    textures = {"firepuffin.png"},
    visual_size = {x = 10, y = 10},
    timeout = 0,
    dtime = 0,

    light_source = 8,
    on_step = function(self, dtime, moveresult)
        self.dtime = self.dtime + dtime
        -- warm air around it
        local pos = self.object:get_pos()
        climate.air_temp_source(pos, 20, 90, 1.0, 1)

        if self.dtime > 1 then
            self.dtime = 0

            local current_velocity = self.object:get_velocity()

            local ground = globo.get_pos_ground(pos)
            local changeVelocity = false
            if ground then
                if ground.y + 11 < pos.y then
                    current_velocity.y = -.5
                    changeVelocity = true
                elseif ground.y + 9 > pos.y then
                    current_velocity.y = .5
                    changeVelocity = true
                end
            end
            if current_velocity.x == 0 and current_velocity.z == 0 then
                current_velocity.x = math.sin(get_gametime()) * .2
                current_velocity.z = math.cos(get_gametime()) * .2
                changeVelocity = true
            end
            if changeVelocity then
                self.object:set_velocity(current_velocity)
            end

            if animals.random_float(0, 1) < 0.01 then
                local p = minetest.find_node_near(
                    pos, 1, {"air", "climate:air_temp", "climate:air_temp_visible"}
                )
                -- minetest.spawn_falling_node(p, "main:ember")
                minetest.set_node(p, {name = "main:ember"})
                minetest.check_for_falling(p)
            end

            -- puffin blast the ground
            if animals.random_float(0, 1) < 0.02 then
                local ground_pos = globo.get_pos_ground(pos)
                if ground_pos then
                    local ground_node = minetest.get_node(ground_pos)
                    globo.create_explosion(ground_pos, 5, 200)
                end
            end
        end

    end,
    on_activate = mobkit.actfunc,
    get_staticdata = mobkit.statfunc, -- optional mobkit props
    -- or used by built in behaviors
    -- physics = [function user defined] 		-- optional, overrides built in physics
    animation = {
        walk = {range = {x = 71, y = 90}, speed = 24, loop = true},
        fast = {range = {x = 91, y = 110}, speed = 24, loop = true},
        stand = {
            {range = {x = 1, y = 30}, speed = 28, loop = true},
            {range = {x = 31, y = 70}, speed = 32, loop = true},
        },
    }, -- on actions
    drops = {{name = "animals:carcass_vert_large", chance = 1, min = 2, max = 4}},
    on_punch = function(self, puncher, time_from_last_punch, tool_capabilities, dir)
        -- minimal.log("ox punched. hp:"..self.hp)
        animals.on_punch(self, tool_capabilities, puncher, 55, 0.05)
        return true
    end,
    on_rightclick = function(self, clicker)
    end,
    groups = {falling_node = 0},
}

minetest.register_entity("animals:firepuffin", baseOx)

minetest.register_node(
    "animals:firepuffin_spawn", {
        tiles = {"animals_gundu_eggs.png"},
        stack_max = minimal.stack_max_medium,
        drawtype = "nodebox",
        paramtype = "light",
        node_box = {type = "fixed", fixed = {-0.125, -0.5, -0.125, 0.125, -0.125, 0.125}},
        groups = {
            snappy = 3,
            falling_node = 1,
            dig_immediate = 3,
            flammable = 1,
            temp_pass = 1,
            edible = 1,
        },
        on_construct = function(pos)
            minetest.get_node_timer(pos):start(1)
        end,
        on_timer = function(pos, elapsed)
            minetest.set_node(pos, {name = "air"})
            pos.y = pos.y + 10
            minetest.add_entity(pos, "animals:firepuffin")
        end,
    }
)

