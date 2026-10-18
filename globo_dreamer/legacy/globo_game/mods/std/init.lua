---@diagnostic disable-next-line: lowercase-global
std = {}

local path = minetest.get_modpath("std")

std.minetest = dofile(path .. "/minetest.lua")
