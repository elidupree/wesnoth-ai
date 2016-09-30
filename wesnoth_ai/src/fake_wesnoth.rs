use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use rand::{self, Rng};
use super::*;
use super::rust_lua_shared::*;


#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Attack {
  pub damage: i32,
  pub number: i32,
  pub damage_type: String,
  pub range: String,
  // TODO: specials
}
#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Side {
  pub gold: i32,
  pub enemies: HashSet <usize>,
  pub recruits: Vec<String>,
  pub player: Arc <Organism>,
  pub memory: Memory,
}


#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Unit {
  pub x: i32,
  pub y: i32,
  pub side: usize,
  pub alignment: i32,
  pub attacks_left: i32,
  pub canrecruit: bool,
  pub cost: i32,
  pub experience: i32,
  pub hitpoints: i32,
  pub level: i32,
  pub max_experience: i32,
  pub max_hitpoints: i32,
  pub max_moves: i32,
  pub moves: i32,
  pub resting: bool,
  pub slowed: bool,
  pub poisoned: bool,
  pub not_living: bool,
  pub zone_of_control: bool,
  pub defense: HashMap<String, i32>,
  pub movement_costs: HashMap <String, i32>,
  pub resistance: HashMap <String, i32>,
  pub attacks: Vec<Attack>,
  // TODO: abilities
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Location {
  pub terrain: String,
  pub village_owner: usize,
  pub unit: Option <Box <Unit>>,
  pub unit_moves: Option <Vec<(Move, f64)>>,
}
#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct TerrainInfo{
  pub keep: bool,
  pub castle: bool,
  pub village: bool,
  pub healing: i32,
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Faction {
  pub recruits: Vec<String>,
  pub leaders: Vec<String>,
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Config {
  pub unit_type_examples: HashMap <String, Unit>,
  pub terrain_info: HashMap <String, TerrainInfo>,
  pub factions: Vec<Faction>,
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Map {
  pub config: Arc <Config>,
  pub width: i32,
  pub height: i32,
  pub locations: Vec<Location>,
  pub starting_locations: Vec<[i32; 2]>,
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct State {
  pub map: Arc <Map>,
  pub current_side: usize,
  pub locations: Vec<Location>,
  pub sides: Vec<Side>,
  pub time_of_day: i32,
  pub turn: i32,
  pub max_turns: i32,
  pub scores: Option <Vec<f64>>,
}
impl State {
  pub fn get (&self, x: i32,y: i32)->&Location {& self.locations [((x-1)+(y-1)*self.map.width) as usize]}
  pub fn get_mut (&mut self, x: i32,y: i32)->&mut Location {&mut self.locations [((x-1)+(y-1)*self.map.width) as usize]}
  pub fn is_enemy (&self, side: usize, other: usize)->bool {self.sides [side].enemies.contains (& other)}
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub enum Move {
  Move {
    src_x: i32, src_y: i32, dst_x: i32, dst_y: i32, moves_left: i32,
  },
  Attack {
    src_x: i32, src_y: i32, dst_x: i32, dst_y: i32, attack_x: i32, attack_y: i32, weapon: usize,
  },
  Recruit {
    dst_x: i32, dst_y: i32, unit_type: String,
  },
  EndTurn,
}

pub fn apply_move (state: &mut State, input: & Move)->Vec<NeuralInput> {
  let mut results = Vec::new();
  match input {
    &Move::Move {src_x, src_y, dst_x, dst_y, moves_left} => {
      let mut unit = state.get_mut (src_x, src_y).unit.take().unwrap();
      results.push (NeuralInput {input_type: "unit_removed".to_string(), vector: neural_unit (state, &unit)});
      unit.x = dst_x;
      unit.y = dst_y;
      unit.moves = moves_left;
      unit.resting = false;
      results.push (NeuralInput {input_type: "unit_added".to_string(), vector: neural_unit (state, &unit)});
      if state.map.config.terrain_info.get (&state.get (dst_x, dst_y).terrain).unwrap().village {
        state.get_mut (dst_x, dst_y).village_owner = unit.side;
      }
      state.get_mut (dst_x, dst_y).unit = Some (unit);
      invalidate_moves (state, [src_x, src_y], 0);
      invalidate_moves (state, [dst_x, dst_y], 0);
    },
    &Move::Attack {src_x, src_y, dst_x, dst_y, attack_x, attack_y, weapon} => {
      //printlnerr!("Attack: {:?}", input);
      if src_x != dst_x || src_y != dst_y {
        results.extend (apply_move (state, &Move::Move {
          src_x: src_x, src_y: src_y, dst_x: dst_x, dst_y: dst_y, moves_left: 0
        }));
      }
      let mut attacker = state.get_mut (dst_x, dst_y).unit.take().unwrap();
      let defender = state.get_mut (attack_x, attack_y).unit.take().unwrap();
      attacker.moves = 0;
      attacker.attacks_left -= 1;
      let (new_attacker, new_defender) = combat_results (state, &attacker, &defender, weapon);
      
      results.push (NeuralInput {input_type: "unit_removed".to_string(), vector: neural_unit (state, & attacker)});
      if let Some (unit) = new_attacker.as_ref() {
        results.push (NeuralInput {input_type: "unit_added".to_string(), vector: neural_unit (state, unit)});
      }
      results.push (NeuralInput {input_type: "unit_removed".to_string(), vector: neural_unit (state, & defender)});
      if let Some (unit) = new_defender.as_ref() {
        results.push (NeuralInput {input_type: "unit_added".to_string(), vector: neural_unit (state, unit)});
      }
      
      let check_game_over = (attacker.canrecruit && new_attacker.is_none()) || (defender.canrecruit && new_defender.is_none());
      
      state.get_mut (dst_x, dst_y).unit = new_attacker;
      state.get_mut (attack_x, attack_y).unit = new_defender;
      
      if check_game_over {
        let mut remaining_leaders = Vec::new();
        for location in state.locations.iter() {
          if let Some (unit) = location.unit.as_ref() {
            if unit.canrecruit {
              remaining_leaders.push (unit);
            }
          }
        }
        assert! (remaining_leaders.len() >0);
        // TODO: allow allies and stuff
        if remaining_leaders.len() == 1 {
          printlnerr!("Victory...");
          state.scores = Some (vec![-1.0; state.sides.len()]);
          state.scores.as_mut().unwrap()[remaining_leaders [0].side] = 1.0;
        }
      }
      invalidate_moves (state, [dst_x, dst_y], 1);
      invalidate_moves (state, [attack_x, attack_y], 1);
    }
    &Move::Recruit {dst_x, dst_y, ref unit_type} => {
      let mut unit = Box::new (state.map.config.unit_type_examples.get (unit_type).unwrap().clone());
      unit.side = state.current_side;
      unit.x = dst_x;
      unit.y = dst_y;
      unit.moves = 0;
      unit.attacks_left = 0;
      results.push (NeuralInput {input_type: "unit_added".to_string(), vector: neural_unit (state, &unit)});
      state.get_mut (dst_x, dst_y).unit = Some (unit);
      invalidate_moves (state, [dst_x, dst_y], 0);
    },
    & Move::EndTurn => {
      state.current_side += 1;
      if state.current_side >= state.sides.len() {
        state.turn += 1;
        state.current_side = 0;
      }
      let mut added_units = Vec::new();
      for (index, location) in state.locations.iter_mut().enumerate() {
        if let Some (unit) = location.unit.as_mut() {
          if unit.side == state.current_side {
            let terrain_healing = state.map.config.terrain_info.get (&location.terrain).unwrap().healing;
            let healing = if unit.poisoned && terrain_healing >0 {0} else if unit.poisoned {-8} else {terrain_healing} + if unit.resting {2} else {0};
            unit.resting = true;
            unit.moves = unit.max_moves;
            unit.attacks_left = 1;
            if healing > 0 && unit.hitpoints < unit.max_hitpoints {
              unit.hitpoints = ::std::cmp::min (unit.max_hitpoints, unit.hitpoints + healing);
            }
            if healing < 0 && unit.hitpoints > 1{
              unit.hitpoints = ::std::cmp::max (1, unit.hitpoints + healing);
            }
            
          }
          added_units.push (index);
          
        }
      }
      results.push (NeuralInput {input_type: "turn_started".to_string(), vector: neural_turn_started (state)});
      for index in added_units {
        let unit = state.locations[index].unit.as_ref().unwrap();
        results.push (NeuralInput {input_type: "unit_added".to_string(), vector: neural_unit (state, unit)});
      }
      if state.turn >state.max_turns {
        // timeout = everybody loses
        state.scores = Some (vec![-1.0; state.sides.len()]);
      }
      for location in state.locations.iter_mut() {
        location.unit_moves = None;
      }
    },
  }
  
  for side in state.sides.iter_mut() {
    for input in results.iter() {
      side.memory = next_memory (& side.player, & side.memory, input);
      //printlnerr!("Processed {:?}", input);
    }
  }
  
  results
}

pub fn choose_defender_weapon (state: & State, attacker: & Unit, defender: & Unit, weapon: usize)->usize {
  let attacker_attack = &attacker.attacks [weapon];
  // pretty similar rules to Wesnoth, but is not important to get them exactly the same.
  let mut best_index = usize::max_value();
  let mut best_score = -100000000000.0;
  for (index, attack) in defender.attacks.iter().enumerate() {
    if attack.range == attacker_attack.range {
      let stats = simulate_and_analyze (state, attacker, defender, weapon, index);
      let score = ((stats.0.death_chance - stats.1.death_chance) *100000.0) +(attacker.hitpoints as f64 - stats.0.average_hitpoints) - (defender.hitpoints as f64 - stats.1.average_hitpoints);
      if score >best_score {
        best_score = score;
        best_index = index;
      }
    }
  }
  best_index
}

// TODO: remove duplicate code between this and simulate_combat
pub fn combat_results (state: & State, attacker: & Unit, defender: & Unit, weapon: usize)->(Option <Box <Unit>>, Option <Box <Unit>>) {
  
  struct Combatant {
    unit: Box <Unit>,
    swings_left: i32,
    damage: i32,
    chance: i32,
  }
  let make_combatant = |unit: &Unit, attack: Option <& Attack>, other: & Unit|->Combatant {
    Combatant {
      unit: Box::new (unit.clone()),
      swings_left: attack.map_or (0, | attack | attack.number),
      // TODO: correct rounding direction
      damage: attack.map_or (0, | attack | attack.damage * other.resistance.get (&attack.damage_type).cloned().unwrap_or (100) / 100),
      chance: other.defense.get (&state.get (other.x, other.y).terrain).unwrap().clone(),
    }
  };
  
  fn swing (swinger: &mut Combatant, victim: &mut Combatant)->bool {
    swinger.swings_left -= 1;
    if rand::thread_rng().gen_range (0, 100) >= swinger.chance {return true;}
    victim.unit.hitpoints -= swinger.damage;
    return victim.unit.hitpoints >0;
  }
  
  let attacker_attack = &attacker.attacks [weapon];
  // TODO: actual selection of best defender attack
  let defender_attack = defender.attacks.get (choose_defender_weapon (state, attacker, defender, weapon));
  
  let mut ac = make_combatant (attacker, Some (attacker_attack), defender);
  let mut dc = make_combatant (defender, defender_attack, attacker);
  
  while ac.swings_left > 0 || dc.swings_left > 0 {
    if ac.swings_left > 0 {
      if !swing (&mut ac, &mut dc) { break; }
    }
    if dc.swings_left > 0 {
      if !swing (&mut dc, &mut ac) { break; }
    }
  }
  
  (
    if ac.unit.hitpoints >0 {Some (ac.unit)} else {None},
    if dc.unit.hitpoints >0 {Some (dc.unit)} else {None},
  )
}

#[derive (Clone, PartialEq, Eq, Hash)]
pub struct CombatantState {
  pub hitpoints: i32,
  // TODO: slow, etc.
}
pub struct CombatStats {
  pub possibilities: HashMap<CombatantState, f64>,
}
// TODO: remove duplicate code between this and combat_results
pub fn simulate_combat (state: & State, attacker: & Unit, defender: & Unit, attacker_weapon: usize, defender_weapon: usize)->(CombatStats, CombatStats) {
  struct Combatant {
    stats: CombatStats,
    swings_left: i32,
    damage: i32,
    chance: i32,
  }
  let make_combatant = |unit: &Unit, attack: Option <& Attack>, other: & Unit|->Combatant {
    Combatant {
      stats: CombatStats {possibilities: ::std::iter::once ((CombatantState {hitpoints: unit.hitpoints}, 1.0)).collect()},
      swings_left: attack.map_or (0, | attack | attack.number),
      // TODO: correct rounding direction
      damage: attack.map_or (0, | attack | attack.damage * other.resistance.get (&attack.damage_type).cloned().unwrap_or (100) / 100),
      chance: other.defense.get (&state.get (other.x, other.y).terrain).unwrap().clone(),
    }
  };
  
  fn swing (swinger: &mut Combatant, victim: &mut Combatant) {
    swinger.swings_left -= 1;
    
    for (possibility, chance) in ::std::mem::replace (&mut victim.stats.possibilities, HashMap::new()) {
      if possibility.hitpoints <= 0 {
        (*victim.stats.possibilities.entry (possibility).or_insert (0.0)) += chance;
      }
      else {
        let mut hit_possibility = possibility.clone();
        hit_possibility.hitpoints -= swinger.damage;
        if hit_possibility.hitpoints < 0 {hit_possibility.hitpoints = 0;}
        (*victim.stats.possibilities.entry (possibility).or_insert (0.0)) += chance*((100 - swinger.chance) as f64/100.0);
        (*victim.stats.possibilities.entry (hit_possibility).or_insert (0.0)) += chance*(swinger.chance as f64/100.0);
      }
    }
  }
  
  let attacker_attack = &attacker.attacks [attacker_weapon];
  let defender_weapon = if defender_weapon == usize::max_value() - 1 { choose_defender_weapon (state, attacker, defender, attacker_weapon) } else {defender_weapon};
  let defender_attack = defender.attacks.get (defender_weapon);
  
  let mut ac = make_combatant (attacker, Some (attacker_attack), defender);
  let mut dc = make_combatant (defender, defender_attack, attacker);
  
  while ac.swings_left > 0 || dc.swings_left > 0 {
    if ac.swings_left > 0 {
      swing (&mut ac, &mut dc);
    }
    if dc.swings_left > 0 {
      swing (&mut dc, &mut ac);
    }
  }
  
  (
    ac.stats,
    dc.stats,
  )
}
pub struct AnalyzedStats {
  pub average_hitpoints: f64,
  pub death_chance: f64,
}
pub fn analyze_stats (stats: &CombatStats)->AnalyzedStats {
  let mut result = AnalyzedStats {
    average_hitpoints: 0.0,
    death_chance: 0.0,
  };
  for (possibility, chance) in stats.possibilities.iter() {
    result.average_hitpoints += possibility.hitpoints as f64*chance;
    if possibility.hitpoints <= 0 {result.death_chance += *chance;}
  }
  result
}
pub fn simulate_and_analyze (state: & State, attacker: & Unit, defender: & Unit, attacker_weapon: usize, defender_weapon: usize)->(AnalyzedStats, AnalyzedStats) {
  let stats = simulate_combat (state, attacker, defender, attacker_weapon, defender_weapon);
  (analyze_stats (&stats.0), analyze_stats (&stats.1))
}

pub fn adjacent_locations (map: & Map, coordinates: [i32; 2])->Vec<[i32; 2]> {
  vec![
    [coordinates [0], coordinates [1] + 1],
    [coordinates [0], coordinates [1] - 1],
    [coordinates [0]-1, coordinates [1] + (coordinates [0]&1)],
    [coordinates [0]-1, coordinates [1] - 1 + (coordinates [0]&1)],
    [coordinates [0]+1, coordinates [1] + (coordinates [0]&1)],
    [coordinates [0]+1, coordinates [1] - 1 + (coordinates [0]&1)],
  ].into_iter().filter (|&[x,y]| x >= 1 && y >= 1 && x <= map.width && y <= map.height).collect()
}

pub fn distance_between (first: [i32; 2], second: [i32; 2])->i32 {
  let horizontal = (first [0] - second [0]).abs();
  let vertical = (first [1] - second [1]).abs() + 
    if (first [1] <second [1] && (first [0] & 1) == 1 && (second [0] & 1) == 0) ||
       (first [1] >second [1] && (first [0] & 1) == 0 && (second [0] & 1) == 1) {1} else {0};
  return ::std::cmp::max (horizontal, vertical + horizontal/2);
}

pub fn find_reach (state: & State, unit: & Unit)->Vec<([i32; 2], i32)> {
  let mut frontiers = vec![HashSet::new(); (unit.moves + 1) as usize];
  let mut results = Vec::new();
  frontiers [unit.moves as usize].insert ([unit.x, unit.y]);
  for moves_left in (0..(unit.moves + 1)).rev() {
    for location in ::std::mem::replace (&mut frontiers [moves_left as usize], HashSet::new()) {
      for adjacent in adjacent_locations (& state.map, location) {
        let mut remaining = moves_left - unit.movement_costs.get (&state.get (adjacent [0], adjacent [1]).terrain).unwrap();
        if remaining >= 0 {
          if remaining >0 {
            let stuff = state.get (location [0], location [1]);
            let info = state.map.config.terrain_info.get (&stuff.terrain).unwrap();
            if info.village && stuff.village_owner != unit.side {
              remaining = 0;
            }
            for double_adjacent in adjacent_locations (& state.map, adjacent) {
              if state.get (double_adjacent [0], double_adjacent [1]).unit.as_ref().map_or (false, | neighbor | neighbor.zone_of_control && state.is_enemy (unit.side, neighbor.side)) {
                remaining = 0;
                break;
              }
            }
          }
          frontiers [remaining as usize].insert (adjacent);
        }
      }    
      results.push ((location, moves_left));
    }
  }
  results
}
