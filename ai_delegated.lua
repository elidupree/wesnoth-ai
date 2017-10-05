--! ai_delegated.lua ==============================================================
local helper = wesnoth.require("lua/helper.lua")

local builtin_ai_stuff = ai --...

local pack = function(...) return {...} end
local wins = 0
local total_fitness = 0
local games = 0
local survivor_games = 0


scenario_ended = function()
  games = games + 1

  local units = wesnoth.get_units ({})
  local win = false
  local sides_defeated = {}
  for I, unit in ipairs (units) do
    if unit.canrecruit and unit.hitpoints <= 0 then
      if wesnoth.is_enemy (unit.side,builtin_ai_stuff.side) then
        -- we won the game! Amazing!
        win = true
        wins = wins + 1
        --current_organism.fitness.wins = current_organism.fitness.wins + 1
      end
      sides_defeated [unit.side] = true
    end
  end
  --error_message(inspect(organisms,{depth=2}))
  
  error_message ("Game over. Wins: ".. wins.."/"..games, true)
end

local do_move_by_type = {
  Move = function(move) builtin_ai_stuff.move(move.src_x, move.src_y, move.dst_x, move.dst_y) end,
  Attack = function(move) 
    if move.src_x ~= move.dst_x or move.src_y ~= move.dst_y then
      builtin_ai_stuff.move(move.src_x, move.src_y, move.dst_x, move.dst_y)
    end
    -- TODO: handle invalidation by events
    builtin_ai_stuff.attack(move.dst_x, move.dst_y, move.attack_x, move.attack_y, move.weapon and move.weapon+1)
  end,
  Recruit = function(move) builtin_ai_stuff.recruit(move.unit_type, move.dst_x, move.dst_y) end,
  EndTurn = function() return true end,
}


local calculate_and_do_one_move = function()
  dump_all_to_rust()
  local best_move = receive_from_rust() --choose_move()
  local movetype, value = best_move, {}
  if type(best_move) == "table" then
    movetype, value  = next (best_move)
  end
  --error_message (movetype, true)
  --error_message (movetype..inspect(value), true)
  return do_move_by_type [movetype] (value)
end


local our_ai = { }
function our_ai:evaluation()
  local units = wesnoth.get_units ({})
  for I, unit in ipairs (units) do
    --wesnoth.message (inspect({unit.moves, unit.attacks_left }))
    if unit.side == wesnoth.current.side and (unit.moves >0 or unit.attacks_left >0) then
      return 10000
    end
  end
  return nil
end
function our_ai:execution()
  for hack=1,100 do
    if calculate_and_do_one_move() then break end
  end
  
  local units = wesnoth.get_units ({})
  for I, unit in ipairs (units) do
    if unit.side == wesnoth.current.side then
      builtin_ai_stuff.stopunit_all(unit)
    end
  end
  hidden_gold = wesnoth.sides [wesnoth.current.side].gold
  wesnoth.sides [wesnoth.current.side].gold = 0
end

return our_ai
--! ==============================================================

