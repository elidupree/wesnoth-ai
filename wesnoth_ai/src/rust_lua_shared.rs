use std::collections::{HashMap, HashSet};

use fake_wesnoth;
use super::*;

//Avoid the built in tanh, to guarantee my consistency with Lua
pub fn hyperbolic_tangent (value: f64)->f64 {
  if value >  18.0 {return  1.0;}
  if value < -18.0 {return -1.0;}
  let term = (value*2.0).exp();
  return (term - 1.0)/(term + 1.0)
}

pub fn multiply_into (input: &[f64], output: &mut [f64], matrix: & Matrix) {
  assert_eq!(input.len(), matrix.input_size);
  assert_eq!(output.len(), matrix.output_size);
  for input_index in 0..matrix.input_size {
    for output_index in 0..matrix.output_size {
      output [output_index] += input [input_index]*matrix.weights [output_index + input_index*matrix.output_size]
    }
  }
}


pub fn next_memory (organism: & Organism, memory: & Memory, input: &NeuralInput)->Memory {
  let mut result = Memory {layers: Vec::new()};
  for layer in 0..organism.layer_sizes.len() {
    let layer_weights = &organism.weights_by_input.get (& input.input_type).unwrap()[layer];
    let mut next_layer = layer_weights.bias.clone();
    multiply_into (&memory.layers [layer], &mut next_layer, & layer_weights.hidden_matrix);
    multiply_into (if layer == 0 {&input.vector} else {& result.layers [layer - 1]}, &mut next_layer, & layer_weights.input_matrix);
    for item in next_layer.iter_mut() {
      *item = hyperbolic_tangent (*item);
    }
    result.layers.push (next_layer);
  }
  result
}

pub fn initial_memory (organism: & Organism)->Memory {
  let mut result = Memory {layers: Vec::new()};
  for layer in 0..organism.layer_sizes.len() {
    let next_layer = vec![0.0; organism.layer_sizes [layer]];
    result.layers.push (next_layer);
  }
  result 
}

pub fn evaluate_move (organism: & Organism, memory: & Memory, input: & NeuralInput)->f64 {
  if input.input_type == "end_turn" {return 0.0;}
  let mut output = vec![0.0];
  multiply_into (&next_memory (organism, memory, input).layers.last().unwrap(), &mut output, & organism.output_weights);
  output [0]
}

thread_local! {
  pub static INPUTS: HashMap <String, usize> = {
    let mut result = HashMap::new();
    result.insert ("turn_started".to_string(), 4);
    result.insert ("unit_added".to_string(), UNIT_SIZE);
    result.insert ("unit_removed".to_string(), UNIT_SIZE);
    
    result.insert ("move".to_string(), LOCATION_SIZE + UNIT_SIZE);
    result.insert ("attack".to_string(), LOCATION_SIZE*2 + UNIT_SIZE*2 + 8);
    result.insert ("recruit".to_string(), UNIT_SIZE);
    result
  }
}

pub fn neural_bool (value: bool)->f64 {if value {1.0} else {0.0}}

const LOCATION_SIZE: usize = 6;
pub fn neural_location (state: & fake_wesnoth::State,x: i32,y: i32)->Vec<f64> {
  let terrain = & state.get (x, y).terrain;
  let info = state.map.config.terrain_info.get (terrain).unwrap();
  vec![
    x as f64, y as f64,
    neural_bool (info. keep), neural_bool (info.castle), neural_bool (info.village),
    info.healing as f64,
  ]
}

const UNIT_SIZE: usize = 23;
pub fn neural_unit (state: & fake_wesnoth::State, unit: & fake_wesnoth::Unit)->Vec<f64> {
  let terrain = & state.get (unit.x, unit.y).terrain;
  vec![
    unit.x as f64, unit.y as f64,
    unit.moves as f64, unit.attacks_left as f64,
    unit.hitpoints as f64, (unit.max_experience - unit.experience) as f64,
    neural_bool (!state.is_enemy (state.current_side, unit.side)), neural_bool (unit.canrecruit),
    unit.max_hitpoints as f64, unit.max_moves as f64,
    neural_bool (unit.slowed), neural_bool (unit.poisoned), neural_bool (unit.not_living),
    unit.alignment as f64,
    neural_bool (unit.zone_of_control),
    unit.resistance.get ("blade").unwrap().clone() as f64,
    unit.resistance.get ("pierce").unwrap().clone() as f64,
    unit.resistance.get ("impact").unwrap().clone() as f64,
    unit.resistance.get ("fire").unwrap().clone() as f64,
    unit.resistance.get ("cold").unwrap().clone() as f64,
    unit.resistance.get ("arcane").unwrap().clone() as f64,
    unit.defense.get (terrain).unwrap().clone() as f64,
    unit.movement_costs.get (terrain).unwrap().clone() as f64,
    
  ]
}

pub fn neural_turn_started (state: & fake_wesnoth::State)->Vec<f64> {
  let my_side = &state.sides [state.current_side];
  let enemy = &state.sides [(state.current_side+1)&1];
  vec![my_side.gold as f64, 4.0, enemy.gold as f64, 4.0]
}

pub fn neural_wesnoth_move (state: &fake_wesnoth::State, input: & fake_wesnoth::Move)->NeuralInput {
  match input {
    &fake_wesnoth::Move::Move {src_x, src_y, dst_x, dst_y, ..} => NeuralInput {
      input_type: "move".to_string(),
      vector: neural_location (state, dst_x, dst_y).into_iter().chain(neural_unit (state, &state.get (src_x, src_y).unit.as_ref().unwrap())).collect()
    },
    &fake_wesnoth::Move::Attack {src_x, src_y, dst_x, dst_y, attack_x, attack_y, weapon} => {
      let mut attacker = state.get (src_x, src_y).unit.clone().unwrap();
      attacker.x = src_x;
      attacker.y = src_y;
      let defender = state.get (attack_x, attack_y).unit.as_ref().unwrap();
      let stats = fake_wesnoth::simulate_and_analyze (state, &attacker, defender, weapon, usize::max_value() - 1);
      NeuralInput {
        input_type: "attack".to_string(),
        vector: 
          vec![
            stats.0.death_chance, stats.1.death_chance,
            attacker.hitpoints as f64 - stats.0.average_hitpoints, defender.hitpoints as f64 - stats.1.average_hitpoints,
            0.0, 0.0, 0.0, 0.0
          ].into_iter()
          .chain(neural_location (state, dst_x, dst_y))
          .chain(neural_location (state, attack_x, attack_y))
          .chain(neural_unit (state, &state.get (src_x, src_y).unit.as_ref().unwrap()))
          .chain(neural_unit (state, &state.get (attack_x, attack_y).unit.as_ref().unwrap()))
          .collect()
      }

    }
    &fake_wesnoth::Move::Recruit {dst_x, dst_y, ref unit_type} => {
      let mut example = state.map.config.unit_type_examples.get (unit_type).unwrap().clone();
      example.side = state.current_side;
      example.x = dst_x;
      example.y = dst_y;
      
      NeuralInput {
        input_type: "recruit".to_string(),
        vector: neural_unit (state, & example),
      }
    },
    &fake_wesnoth::Move::EndTurn => unreachable!(),
  }
}

pub fn recruit_hexes (state: & fake_wesnoth::State, unit: & fake_wesnoth::Unit)->Vec<[i32; 2]> {
  let mut discovered = HashSet::new();
  let mut frontier = Vec::new();
  if state.map.config.terrain_info.get (&state.get (unit.x, unit.y).terrain).unwrap().keep {
    frontier.push ([unit.x, unit.y]);
  }
  while let Some (location) = frontier.pop() {
    for adjacent in fake_wesnoth::adjacent_locations (& state.map, location) {
      if state.map.config.terrain_info.get (&state.get (adjacent [0], adjacent [1]).terrain).unwrap().castle && !discovered.contains (& adjacent) {
        frontier.push (adjacent);
      }
      discovered.insert (location);
    }
  }
  discovered.into_iter().filter (| location | state.get (location [0], location [1]). unit.is_none()).collect()
}

pub fn possible_unit_moves(state: & fake_wesnoth::State, unit: & fake_wesnoth::Unit)->Vec<fake_wesnoth::Move> {
  if unit.side != state.current_side {return Vec::new();}
  
  let mut results = vec![];

  for location in fake_wesnoth::find_reach (state, unit) {
    let unit_there = state.get (location.0 [0], location.0 [1]).unit.as_ref();
    if unit_there.is_none() {
      results.push (fake_wesnoth::Move::Move {
        src_x: unit.x, src_y: unit.y,
        dst_x: location.0 [0], dst_y: location.0 [1],
        moves_left: location.1
      });
    }
    if unit.attacks_left > 0 && unit_there.map_or (true, | other | unit.x == other.x && unit.y == other.y) {
      for adjacent in fake_wesnoth::adjacent_locations (& state.map, location.0) {
        if let Some (neighbor) = state.get (adjacent [0], adjacent [1]).unit.as_ref() {
          if state.is_enemy (unit.side, neighbor.side) {
            for index in 0..unit.attacks.len() {
              results.push (fake_wesnoth::Move::Attack {
                src_x: unit.x, src_y: unit.y,
                dst_x: location.0 [0], dst_y: location.0 [1],
                attack_x: adjacent [0], attack_y: adjacent [1],
                weapon: index,
              });
            }
          }
        }
      }
    }
  }
  if unit.canrecruit {
    for location in recruit_hexes (state, unit) {
      for recruit in state.sides [unit.side].recruits.iter() {
        if state.sides [unit.side].gold >= state.map.config.unit_type_examples.get (recruit).unwrap().cost {
          results.push (fake_wesnoth::Move::Recruit {
            dst_x: location [0], dst_y: location [1],
            unit_type: recruit.clone(),
          });
        }
      }
    }
  }
  
  results
}

pub fn calculate_moves (state: &mut fake_wesnoth::State) {
  let mut added_moves: HashMap <usize,Vec<(fake_wesnoth::Move, f64)>> = HashMap::new();
  for (index, location) in state.locations.iter().enumerate() {
    if let Some (unit) = location.unit.as_ref() {
      if location.unit_moves.is_none() && unit.side == state.current_side {
        added_moves.insert (index, possible_unit_moves (state, unit).into_iter().map (| action | {
          let evaluation = evaluate_move (& state.sides [state.current_side].player, & state.sides [state.current_side].memory, &neural_wesnoth_move (state, &action));
          (action, evaluation)
        }).collect());
      }
    }
  }
  for (index, moves) in added_moves {
    state.locations [index].unit_moves = Some (moves)
  }
}

pub fn invalidate_moves (state: &mut fake_wesnoth::State, origin: [i32; 2], extra_turns: i32) {
  for location in state.locations.iter_mut() {
    if location.unit.as_ref().map_or (true, | unit | fake_wesnoth::distance_between ([unit.x, unit.y], origin) <= unit.moves + 1 + extra_turns*unit.max_moves) {
      location.unit_moves = None;
    }
  }
}

pub fn collect_moves (state: &mut fake_wesnoth::State)->Vec<(fake_wesnoth::Move, f64)> {
  calculate_moves (state);
  
  let mut results = vec![(fake_wesnoth::Move::EndTurn, 0.0)];
  for location in state.locations.iter() {
    if let Some (moves) = location.unit_moves.as_ref() {
      results.extend (moves.into_iter().cloned());
    }
  }
  
  results
}
pub fn choose_move (state: &mut fake_wesnoth::State)->fake_wesnoth::Move {
  let mut moves = collect_moves (state);
  moves.sort_by (|a, b| a.1.partial_cmp(&b.1).unwrap());
  //printlnerr!("Moves: {:?}", moves);
  moves.iter().rev().next().unwrap().0.clone()
}
