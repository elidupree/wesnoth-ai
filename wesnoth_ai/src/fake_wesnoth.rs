use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::cmp::{min, max};
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
  pub damage_type: usize,
  pub range: String,
  // TODO: specials
  pub charge: bool,
  pub magical: bool,
  pub marksman: bool,
  pub poison: bool,
  pub slow: bool,
  //pub backstab: bool,
}
#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Side {
  pub gold: i32,
  pub enemies: Vec<bool>,
  pub recruits: Vec<String>,
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct UnitType {
  pub alignment: i32,
  pub cost: i32,
  pub level: i32,
  pub max_experience: i32,
  pub max_hitpoints: i32,
  pub max_moves: i32,
  pub not_living: bool,
  pub zone_of_control: bool,
  pub defense: Vec<i32>,
  pub movement_costs: Vec<i32>,
  pub resistance: [i32; 6],
  pub attacks: Vec<Attack>,
  // TODO: abilities
  pub skirmisher: bool,
  pub regeneration: i32,
  //pub leadership: bool,
  pub fearless: bool,
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Unit {
  pub x: i32,
  pub y: i32,
  pub id: usize,
  pub side: usize,
  pub attacks_left: i32,
  pub canrecruit: bool,
  pub experience: i32,
  pub hitpoints: i32,
  pub moves: i32,
  pub resting: bool,
  pub slowed: bool,
  pub poisoned: bool,
  pub unit_type: Arc<UnitType>,
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Location {
  pub terrain: usize,
  pub village_owner: Option<usize>,
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
  pub terrain_info: Vec<TerrainInfo>,
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
  pub next_id: usize,
  pub scores: Option <Vec<f64>>,
}
impl State {
  pub fn get (&self, x: i32,y: i32)->&Location {& self.locations [((x-1)+(y-1)*self.map.width) as usize]}
  pub fn get_mut (&mut self, x: i32,y: i32)->&mut Location {&mut self.locations [((x-1)+(y-1)*self.map.width) as usize]}
  pub fn geta (&self, coordinates: [i32; 2])->&Location {self.get(coordinates[0], coordinates[1])}
  pub fn geta_mut (&mut self, coordinates: [i32; 2])->&mut Location {self.get_mut(coordinates[0], coordinates[1])}
  pub fn get_terrain_info (&self, x: i32,y: i32)->&TerrainInfo {self.map.config.terrain_info.get (self.get(x,y).terrain).unwrap()}
  pub fn is_enemy (&self, side: usize, other: usize)->bool {self.sides [side].enemies[other]}
}

#[derive (Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
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
        state.get_mut (dst_x, dst_y).village_owner = Some(unit.side);
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
      unit.id = state.next_id;
      state.next_id += 1;
      unit.moves = 0;
      unit.attacks_left = 0;
      state.sides [state.current_side].gold -= unit.unit_type.cost;
      state.get_mut (dst_x, dst_y).unit = Some (unit.clone());
      for player in players.iter_mut() {
        player.recruit_completed (state, &unit);
      }
    },
    & Move::EndTurn => {
      let previous_side = state.current_side;
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
            let terrain_healing = state.map.config.terrain_info.get (location.terrain).unwrap().healing;
            let regular_healing = max (terrain_healing, unit.unit_type.regeneration);
            let healing = if unit.poisoned && regular_healing >0 {0} else if unit.poisoned {-8} else {terrain_healing} + if unit.resting {2} else {0};
            if regular_healing > 0 {unit.poisoned = false;}
            unit.resting = true;
            unit.moves = unit.unit_type.max_moves;
            unit.attacks_left = 1;
            if healing > 0 && unit.hitpoints < unit.unit_type.max_hitpoints {
              unit.hitpoints = ::std::cmp::min (unit.unit_type.max_hitpoints, unit.hitpoints + healing);
            }
            if healing < 0 && unit.hitpoints > 1{
              unit.hitpoints = ::std::cmp::max (1, unit.hitpoints + healing);
            }
          }
          if unit.side == previous_side {
            unit.slowed = false;
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

pub fn defender_weapon_score (stats: & CombatStats)->f64 {
  ((stats.combatants [0].death_chance - stats.combatants [1].death_chance) *100000.0)
    +(stats.combatants [1].original_hitpoints as f64 - stats.combatants [1].average_hitpoints)
    -(stats.combatants [0].original_hitpoints as f64 - stats.combatants [0].average_hitpoints)
}

pub fn choose_defender_weapon (state: & State, attacker: & Unit, defender: & Unit, weapon: usize)->usize {
  let attacker_attack = &attacker.unit_type.attacks [weapon];
  let matching_attacks = || defender.unit_type.attacks.iter().enumerate().filter (| &(_, attack) | attack.range == attacker_attack.range);
  if matching_attacks().count() == 1 {return matching_attacks().next().unwrap().0;}
  // pretty similar rules to Wesnoth, but is not important to get them exactly the same.
  let mut best_index = usize::max_value();
  let mut best_score = -100000000000.0;
  for (index, attack) in matching_attacks() {
    if attack.range == attacker_attack.range {
      let stats = simulate_combat (state, attacker, defender, weapon, index);
      let score = defender_weapon_score (&stats);
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
  else if state.time_of_day == 4 || state.time_of_day == 5 { -25 }
  else { 0 }
}
pub fn alignment_multiplier (state: & State, unit: & Unit)->i32 {
  let bonus = unit.unit_type.alignment*lawful_bonus (state);
  if unit.unit_type.fearless && bonus < 0 {return 100;}
  100 + bonus
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct CombatantInfo {
  swings_left: i32,
  pub swings: i32,
  pub damage: i32,
  pub slow_damage: i32,
  pub chance: i32,
  pub slow: bool,
  pub poison: bool,
}

fn round_damage (original: i32, numerator: i32, denominator: i32)->i32 {
  let damage_scaled = original * numerator;
  let damage_rounded_to_nearest = (damage_scaled + (denominator>>1))/denominator;
  if damage_rounded_to_nearest < 1 {
    1
  } else if (damage_rounded_to_nearest < original) && ((damage_scaled % denominator) == (denominator>>1)) {
    damage_rounded_to_nearest + 1
  } else {
    damage_rounded_to_nearest
  }
}

fn combatant_info (state: & State, unit: & Unit, opponent: & Unit, attack: Option <& Attack>, opponent_attack: Option<& Attack>, is_attacker: bool)->CombatantInfo {
  match attack {
    None => CombatantInfo {
      swings_left: 0,
      swings: 0,
      damage: 0,
      slow_damage: 0,
      chance: 0,
      slow: false, poison: false,
    },
    Some (attack) => {
      let attacker_attack = if is_attacker {Some(attack)} else {opponent_attack};
      let mut damage = attack.damage;
      if attacker_attack.map_or (false, | a | a.charge) {damage *= 2;}
      let multiplier = opponent.unit_type.resistance.get (attack.damage_type).cloned().unwrap_or (100)*alignment_multiplier (state, unit);
      let chance = if attack.magical {70} else {
        let chance = opponent.unit_type.defense.get (state.get (opponent.x, opponent.y).terrain).unwrap().clone();
        if chance <60 && attack.marksman {60} else {chance}
      };
      CombatantInfo {
        swings_left: attack.number,
        swings: attack.number,
        damage: round_damage (attack.damage, multiplier, 10000),
        slow_damage: round_damage (attack.damage, multiplier, 10000),
        chance: chance,
        slow: attack. slow, poison: attack.poison,
      }
    },
  }
}

fn make_combatants <R, Maker: Fn (& Unit, & Unit, Option <& Attack>, Option <& Attack>, CombatantInfo)->R> (state: & State, attacker: & Unit, defender: & Unit, attacker_weapon: usize, defender_weapon: usize, make_combatant: Maker) -> (R,R) {
  let attacker_attack = &attacker.unit_type.attacks [attacker_weapon];
  let defender_weapon = if defender_weapon == CHOOSE_WEAPON { choose_defender_weapon (state, attacker, defender, attacker_weapon) } else {defender_weapon};
  let defender_attack = defender.unit_type.attacks.get (defender_weapon);
  
  (
    make_combatant (attacker, defender, Some (attacker_attack), defender_attack, combatant_info (state, attacker, defender, Some (attacker_attack), defender_attack, true)),
    make_combatant (defender, attacker, defender_attack, Some (attacker_attack), combatant_info (state, defender, attacker, defender_attack, Some (attacker_attack), false)),
  )
}

pub const CHOOSE_WEAPON: usize = ::std::usize::MAX - 1;

pub fn combat_results (state: & State, attacker: & Unit, defender: & Unit, weapon: usize)->(Option <Box <Unit>>, Option <Box <Unit>>) {
  #[derive (Debug)]
  struct Combatant {
    unit: Box <Unit>,
    info: CombatantInfo,
  }
  
  fn swing (swinger: &mut Combatant, victim: &mut Combatant)->bool {
    swinger.info.swings_left -= 1;
    if rand::thread_rng().gen_range (0, 100) >= swinger.info.chance {return true;}
    
    victim.unit.hitpoints -= if swinger.unit.slowed {swinger.info.slow_damage} else {swinger.info.damage};
    if swinger.info.slow {victim.unit.slowed = true;}
    if swinger.info.poison {victim.unit.poisoned = true;}
    
    return victim.unit.hitpoints >0;
  }
  fn finish (me: &mut Combatant, other: &mut Combatant) {
    if me.unit.hitpoints > 0 {
      let enemy_level = other.unit.unit_type.level;
      let experience = if other.unit.hitpoints > 0 {enemy_level} else if enemy_level == 0 {4} else {enemy_level*8};
      me.unit.experience += experience;
      if me.unit.experience >= me.unit.unit_type.max_experience {
        me.unit.experience -= me.unit.unit_type.max_experience;
        me.unit.hitpoints = me.unit.unit_type.max_hitpoints;
      }
    }
  }
  
  let (mut ac, mut dc) = make_combatants (state, attacker, defender, weapon, CHOOSE_WEAPON, |unit, other, attack, other_attack, info | {
    Combatant {
      unit: Box::new (unit.clone()),
      info: info,
    }
  });
  ac.unit.resting = false;
  dc.unit.resting = false;
  
  while ac.info.swings_left > 0 || dc.info.swings_left > 0 {
    if ac.info.swings_left > 0 {
      if !swing (&mut ac, &mut dc) { break; }
    }
    if dc.info.swings_left > 0 {
      if !swing (&mut dc, &mut ac) { break; }
    }
  }
  //printlnerr!("{:?}, {:?}, {:?}, {:?}, ", attacker, defender, ac, dc);
  
  finish (&mut ac, &mut dc);
  finish (&mut dc, &mut ac);
  (
    if ac.unit.hitpoints >0 {Some (ac.unit)} else {None},
    if dc.unit.hitpoints >0 {Some (dc.unit)} else {None},
  )
}

// TODO: slow, etc.

use smallvec::SmallVec;
use arrayvec::ArrayVec;
#[derive (Clone, Debug)]
pub struct CombatantStats {
  pub info: CombatantInfo,
  pub original_hitpoints: i32,
  pub hits_to_die: usize,
  pub average_hitpoints: f64,
  pub death_chance: f64,
}
#[derive (Clone, Debug)]
pub struct CombatPossibility {
  chance: f64,
}
#[derive (Clone, Debug)]
pub struct CombatStats {
  pub possibilities: SmallVec<[f64; 6*6]>,
  pub combatants: [CombatantStats; 2],
}

impl CombatStats {
  pub fn index_of (&self, hits: [usize; 2])->usize {
    hits[0] + hits[1]*(self.combatants[0].info.swings+1) as usize
  }
  pub fn possibility_chance (&self, hits: [usize; 2])->f64 {
    self.possibilities.get (self.index_of(hits)).cloned().unwrap_or (0.0)
  }
  pub fn possibility_chance_mut (&mut self, hits: [usize; 2])->&mut f64 {
    let idx = self.index_of (hits);
    self.possibilities.get_mut(idx).unwrap()
  }
}

pub fn simulate_combat (state: & State, attacker: & Unit, defender: & Unit, attacker_weapon: usize, defender_weapon: usize)->CombatStats {
  if defender_weapon == CHOOSE_WEAPON {
    let attacker_attack = &attacker.unit_type.attacks [attacker_weapon];
    let matching_attacks = || defender.unit_type.attacks.iter().enumerate().filter (| &(_, attack) | attack.range == attacker_attack.range);
    if matching_attacks().count() > 1 {
      return matching_attacks().map (|(index, _attack)| {
        let stats = simulate_combat (state, attacker, defender, attacker_weapon, index);
        let score = defender_weapon_score (&stats);
        (stats, score)
      }).max_by (| a,b | a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
    }
  }

  fn swing (stats: &mut CombatStats, swinger: usize, victim: usize) {
    let max_hits_by_swinger_so_far = (stats.combatants [swinger].info.swings - stats.combatants [swinger].info.swings_left) as usize;
    let max_hits_by_victim_so_far = (stats.combatants [victim].info.swings - stats.combatants [victim].info.swings_left) as usize;
    
    for hits_by_swinger in 0..(max_hits_by_swinger_so_far+1) {
      for hits_by_victim in 0..min(stats.combatants [swinger].hits_to_die, max_hits_by_victim_so_far+1) {
        let hits = if swinger == 0 {[hits_by_swinger, hits_by_victim]} else {[hits_by_victim, hits_by_swinger]};
        let next_hits = if swinger == 0 {[hits_by_swinger+1, hits_by_victim]} else {[hits_by_victim, hits_by_swinger+1]};
        
        let chance = stats.possibility_chance (hits);
        let chance_change = chance*(stats.combatants [swinger].info.chance as f64/100.0);
        *stats.possibility_chance_mut (hits) -= chance_change;
        *stats.possibility_chance_mut (next_hits) += chance_change;
      }
    }
    stats.combatants [swinger].info.swings_left -= 1;
  }
  
  let (ac, dc) = make_combatants (state, attacker, defender, attacker_weapon, defender_weapon, |unit, other, attack, other_attack, info | {
    CombatantStats {
      info: info,
      original_hitpoints: unit.hitpoints,
      hits_to_die: other_attack.map_or (1, | other_attack| ((unit.hitpoints+other_attack.damage-1) / other_attack.damage) as usize),
      average_hitpoints: 0.0,
      death_chance: 0.0,
    }
  });
  let number_of_possibilities = ((ac.info.swings+1)*(dc.info.swings+1)) as usize;
  let mut possibilities = SmallVec::with_capacity (number_of_possibilities);
  for _ in 0..number_of_possibilities {
    possibilities.push (0.0);
  }
  let mut stats = CombatStats {
    combatants: [ac, dc],
    possibilities: possibilities,
  };
  *stats.possibility_chance_mut ([0, 0]) = 1.0;
  
  while stats.combatants [0].info.swings_left > 0 || stats.combatants [1].info.swings_left > 0 {
    if stats.combatants [0].info.swings_left > 0 {
      swing (&mut stats, 0, 1);
    }
    if stats.combatants [1].info.swings_left > 0 {
      swing (&mut stats, 1, 0);
    }
  }
  
  for hits_by_attacker in 0..(stats.combatants [0].info.swings+1) as usize {
    for hits_by_defender in 0..(stats.combatants [1].info.swings+1) as usize {
      let hits = [hits_by_attacker, hits_by_defender];
      let chance = stats.possibility_chance (hits);
      for (unit, opponent) in [(0, 1), (1, 0)].iter().cloned() {
        let hits_by_opponent = if unit == 0 {hits_by_defender} else {hits_by_attacker};
        let hitpoints = max (0, stats.combatants [unit].original_hitpoints - stats.combatants [opponent].info.damage*hits_by_opponent as i32);
        stats.combatants [unit].average_hitpoints += hitpoints as f64*chance;
        if hitpoints <= 0 {stats.combatants [unit].death_chance += chance;}
      }
    }
  }

  
  stats
}

pub fn adjacent_locations (map: & Map, coordinates: [i32; 2])->ArrayVec<[[i32; 2]; 6]> {
  let mut result = ArrayVec::new();
  {
  let mut consider = | [x,y]:[i32;2] | {
    if x >= 1 && y >= 1 && x <= map.width && y <= map.height {
      result.push ([x,y]);
    }
  };
  consider ([coordinates [0], coordinates [1] + 1]);
  consider ([coordinates [0], coordinates [1] - 1]);
  consider ([coordinates [0]-1, coordinates [1] - (coordinates [0]&1)]);
  consider ([coordinates [0]-1, coordinates [1] + 1 - (coordinates [0]&1)]);
  consider ([coordinates [0]+1, coordinates [1] - (coordinates [0]&1)]);
  consider ([coordinates [0]+1, coordinates [1] + 1 - (coordinates [0]&1)]);
  }
  result
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
        let mut remaining = moves_left - unit.unit_type.movement_costs.get (stuff.terrain).unwrap();
        if remaining >= 0 && !discovered.contains (&adjacent) && stuff.unit.as_ref().map_or (true, | neighbor | !state.is_enemy (unit.side, neighbor.side)) {
          if remaining > 0 && !unit.unit_type.skirmisher {
            for double_adjacent in adjacent_locations (& state.map, adjacent) {
              if state.get (double_adjacent [0], double_adjacent [1]).unit.as_ref().map_or (false, | neighbor | neighbor.unit_type.zone_of_control && state.is_enemy (unit.side, neighbor.side)) {
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
      let info = state.map.config.terrain_info.get (stuff.terrain).unwrap();
      let capture = info.village && stuff.village_owner != Some(unit.side);
      results.push ((location, if capture {0} else {moves_left}));
    }
  }
  results
}

pub fn total_income (state: & State, side: usize)->i32 {
  let mut villages = 0;
  let mut upkeep = 0;
  for location in state.locations.iter() {
    if state.map.config.terrain_info.get (location.terrain).unwrap().village && location.village_owner == Some(side) {
      villages += 1;
    }
    if let Some (unit) = location.unit.as_ref() {
      if unit.side == side && unit.canrecruit == false {
        upkeep += unit.unit_type.level;
      }
    }
  }
  2 + villages*2 - ::std::cmp::max (0, upkeep - villages)
}
