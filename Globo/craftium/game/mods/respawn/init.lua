minetest.register_on_dieplayer(
    function(player, reason)
        minetest.after(
            1, function(name)
                player:respawn()
                local name = player:get_player_name()
                minetest.close_formspec(name, "")
            end, player
        )
    end
)
