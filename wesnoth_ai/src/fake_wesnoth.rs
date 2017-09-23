use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use rand::{self, Rng};


pub trait Player {
  fn move_completed (&mut self, _state: & State, _previous: & Unit, _current: & Unit) {}
  fn attack_completed (&mut self, _state: & State, _attacker: & Unit, _defender: & Unit, _new_attacker: Option <&Unit>, _new_defender: Option <&Unit>) {}
  fn recruit_completed (&mut self, _state: & State, _unit: & Unit) {}
  fn turn_started (&mut self, _state: & State) {}
  fn choose_move (&mut self, state: & State)->Move;
}
     

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
  pub fn get_terrain_info (&self, x: i32,y: i32)->&TerrainInfo {self.map.config.terrain_info.get (&self.get(x,y).terrain).unwrap()}
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

pub fn apply_move (state: &mut State, players: &mut Vec<Box <Player>>, input: & Move) {
  match input {
    &Move::Move {src_x, src_y, dst_x, dst_y, moves_left} => {
      let mut unit = state.get_mut (src_x, src_y).unit.take().unwrap();
      let old_unit = unit.clone();
      unit.x = dst_x;
      unit.y = dst_y;
      unit.moves = moves_left;
      unit.resting = false;
      if state.get_terrain_info(dst_x, dst_y).village {
        state.get_mut (dst_x, dst_y).village_owner = unit.side + 1;
      }
      state.get_mut (dst_x, dst_y).unit = Some (unit.clone());
      for player in players.iter_mut() {
        player.move_completed (state, &old_unit, &unit);
      }
    },
    &Move::Attack {src_x, src_y, dst_x, dst_y, attack_x, attack_y, weapon} => {
      //printlnerr!("Attack: {:?}", input);
      if src_x != dst_x || src_y != dst_y {
        apply_move (state, players, &Move::Move {
          src_x: src_x, src_y: src_y, dst_x: dst_x, dst_y: dst_y, moves_left: 0
        });
      }
      let mut attacker = state.get_mut (dst_x, dst_y).unit.take().unwrap();
      let defender = state.get_mut (attack_x, attack_y).unit.take().unwrap();
      attacker.moves = 0;
      attacker.attacks_left -= 1;
      let (new_attacker, new_defender) = combat_results (state, &attacker, &defender, weapon);
      
      let check_game_over = (attacker.canrecruit && new_attacker.is_none()) || (defender.canrecruit && new_defender.is_none());
      
      state.get_mut (dst_x, dst_y).unit = new_attacker.clone();
      state.get_mut (attack_x, attack_y).unit = new_defender.clone();
      
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
          //printlnerr!("Victory...");
          state.scores = Some (vec![-1.0; state.sides.len()]);
          state.scores.as_mut().unwrap()[remaining_leaders [0].side] = 1.0;
        }
      }
      for player in players.iter_mut() {
        player.attack_completed (state, & attacker, & defender, new_attacker.as_ref().map (| unit | &**unit), new_defender.as_ref().map (| unit | &**unit));
      }
    }
    &Move::Recruit {dst_x, dst_y, ref unit_type} => {
      let mut unit = Box::new (state.map.config.unit_type_examples.get (unit_type).unwrap().clone());
      unit.side = state.current_side;
      unit.x = dst_x;
      unit.y = dst_y;
      unit.moves = 0;
      unit.attacks_left = 0;
      state.sides [state.current_side].gold -= unit.cost;
      state.get_mut (dst_x, dst_y).unit = Some (unit.clone());
      for player in players.iter_mut() {
        player.recruit_completed (state, &unit);
      }
    },
    & Move::EndTurn => {
      state.current_side += 1;
      if state.current_side >= state.sides.len() {
        state.turn += 1;
        state.current_side = 0;
        state.time_of_day = (state.time_of_day + 1) % 6;
      }
      state.sides [state.current_side].gold += total_income (state, state.current_side);
      for location in state.locations.iter_mut() {
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
        }
      }
      if state.turn >state.max_turns {
        // timeout = everybody loses
        state.scores = Some (vec![-1.0; state.sides.len()]);
      }
      for player in players.iter_mut() {
        player.turn_started (state);
      }
    },
  }
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

pub fn lawful_bonus (state: & State)->i32 {
  if state.time_of_day == 1 || state.time_of_day == 2 { 25 }
  else if state.time_of_day == 4 || state.time_of_day == 5 { 25 }
  else { 0 }
}
pub fn alignment_multiplier (state: & State, unit: & Unit)->i32 {
  100 + unit.alignment*lawful_bonus (state)
}

// TODO: remove duplicate code between this and simulate_combat
pub fn combat_results (state: & State, attacker: & Unit, defender: & Unit, weapon: usize)->(Option <Box <Unit>>, Option <Box <Unit>>) {
  #[derive (Debug)]
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
      damage: attack.map_or (0, | attack | attack.damage * other.resistance.get (&attack.damage_type).cloned().unwrap_or (100)*alignment_multiplier (state, unit) / 10000),
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
  ac.unit.resting = false;
  dc.unit.resting = false;
  
  while ac.swings_left > 0 || dc.swings_left > 0 {
    if ac.swings_left > 0 {
      if !swing (&mut ac, &mut dc) { break; }
    }
    if dc.swings_left > 0 {
      if !swing (&mut dc, &mut ac) { break; }
    }
  }
  //printlnerr!("{:?}, {:?}, {:?}, {:?}, ", attacker, defender, ac, dc);
  
  (
    if ac.unit.hitpoints >0 {Some (ac.unit)} else {None},
    if dc.unit.hitpoints >0 {Some (dc.unit)} else {None},
  )
}

#[derive (Clone, PartialEq, Eq, Hash, Debug)]
pub struct CombatantState {
  pub hitpoints: i32,
  // TODO: slow, etc.
}
#[derive (Clone, Debug)]
pub struct CombatStats {
  pub possibilities: HashMap<CombatantState, f64>,
}
// TODO: remove duplicate code between this and combat_results
pub fn simulate_combat (state: & State, attacker: & Unit, defender: & Unit, attacker_weapon: usize, defender_weapon: usize)->(CombatStats, CombatStats) {
  #[derive (Debug)]
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
      damage: attack.map_or (0, | attack | attack.damage * other.resistance.get (&attack.damage_type).cloned().unwrap_or (100)*alignment_multiplier (state, unit) / 10000),
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
  //printlnerr!("{:?}, {:?}, {:?}, {:?}, ", attacker, defender, ac, dc);
  
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
    [coordinates [0]-1, coordinates [1] - (coordinates [0]&1)],
    [coordinates [0]-1, coordinates [1] + 1 - (coordinates [0]&1)],
    [coordinates [0]+1, coordinates [1] - (coordinates [0]&1)],
    [coordinates [0]+1, coordinates [1] + 1 - (coordinates [0]&1)],
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
  let mut frontiers = vec![Vec::new(); (unit.moves + 1) as usize];
  let mut discovered = HashSet::new();
  let mut results = Vec::new();
  frontiers [unit.moves as usize].push ([unit.x, unit.y]);
  for moves_left in (0..(unit.moves + 1)).rev() {
    for location in ::std::mem::replace (&mut frontiers [moves_left as usize], Vec::new()) {
      for adjacent in adjacent_locations (& state.map, location) {
        let stuff = state.get (adjacent [0], adjacent [1]);
        let mut remaining = moves_left - unit.movement_costs.get (& stuff.terrain).unwrap();
        if remaining >= 0 && !discovered.contains (&adjacent) && stuff.unit.as_ref().map_or (true, | neighbor | !state.is_enemy (unit.side, neighbor.side)) {
          if remaining >0 {
            for double_adjacent in adjacent_locations (& state.map, adjacent) {
              if state.get (double_adjacent [0], double_adjacent [1]).unit.as_ref().map_or (false, | neighbor | neighbor.zone_of_control && state.is_enemy (unit.side, neighbor.side)) {
                remaining = 0;
                break;
              }
            }
          }
          discovered.insert (adjacent);
          frontiers [remaining as usize].push (adjacent);
        }
      }
      let stuff = state.get (location [0], location [1]);
      let info = state.map.config.terrain_info.get (&stuff.terrain).unwrap();
      let capture = info.village && stuff.village_owner != unit.side + 1;
      results.push ((location, if capture {0} else {moves_left}));
    }
  }
  results
}

pub fn total_income (state: & State, side: usize)->i32 {
  let mut villages = 0;
  let mut upkeep = 0;
  for location in state.locations.iter() {
    if state.map.config.terrain_info.get (&location.terrain).unwrap().village && location.village_owner == side + 1 {
      villages += 1;
    }
    if let Some (unit) = location.unit.as_ref() {
      if unit.side == side && unit.canrecruit == false {
        upkeep += unit.level;
      }
    }
  }
  villages*2 - ::std::cmp::max (0, upkeep - villages)
}
