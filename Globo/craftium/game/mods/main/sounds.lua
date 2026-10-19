-- Override default player sounds
function default.node_sound_defaults(table)
    table = table or {}
    table.footstep = nil
    table.dug = nil
    table.place = nil
    table.jump = nil
    return table
end

-- Override default item sounds
function default.node_sound_stone_defaults(table)
    table = table or {}
    table.footstep = nil
    table.dug = nil
    table.place = nil
    return table
end

-- Add more sound override functions for other node groups if needed
