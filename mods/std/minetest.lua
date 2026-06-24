local minetest_get_gametime = minetest.get_gametime

local minetest = {}

-- `minetest.get_gametime()` returns nil before the first server step. This implementation returns 0 instead.
function minetest.get_gametime()
    local value = minetest_get_gametime()
    return value ~= nil and value or 0
end

return minetest
