use rand::{random};

use fake_wesnoth;
use rust_lua_shared::*;


#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Player {
  pub unit_moves: Vec<Option <Vec<(Move, f64)>>>,
}

use fake_wesnoth::{State, Unit, Move};

impl fake_wesnoth::Player for Player {
  fn move_completed (&mut self, state: & State, previous: & Unit, current: & Unit) {
    self.invalidate_moves (state, [previous.x, previous.y], 0);
    self.invalidate_moves (state, [current.x, current.y], 0)
  }
  fn attack_completed (&mut self, state: & State, attacker: & Unit, defender: & Unit, _: Option <& Unit>, _: Option <& Unit>) {
    self.invalidate_moves (state, [attacker.x, attacker.y], 0);
    self.invalidate_moves (state, [defender.x, defender.y], 0);
  }

  fn recruit_completed (&mut self, state: & State, unit: & Unit) {
    self.invalidate_moves (state, [unit.x, unit.y], 0);
  }

  fn turn_started (&mut self, _: & State) {
    for location in self.unit_moves.iter_mut() {
      *location = None;
    }
  }

  fn choose_move (&mut self, state: & State)->Move {
    let mut moves = self.collect_moves (state);
    moves.sort_by (|a, b| a.1.partial_cmp(&b.1).unwrap());
    //printlnerr!("Moves: {:?}", moves);
    moves.iter().rev().next().unwrap().0.clone()
  }
}

impl Player {
  pub fn new(map: & fake_wesnoth::Map)->Player {Player {unit_moves: vec![None; (map.width*map.height) as usize]}}

  pub fn calculate_moves (&mut self, state: &fake_wesnoth::State) {
    for (index, location) in state.locations.iter().enumerate() {
      if let Some (unit) = location.unit.as_ref() {
        if self.unit_moves [index].is_none() && unit.side == state.current_side {
          self.unit_moves [index] = Some (possible_unit_moves (state, unit).into_iter().map (| action | {
            let evaluation = evaluate_move (state, &action, true);
            (action, evaluation)
          }).collect());
        }
      }
    }
  }
  
  pub fn invalidate_moves (&mut self, state: &fake_wesnoth::State, origin: [i32; 2], extra_turns: i32) {
    for (index, location) in state.locations.iter().enumerate() {
      if location.unit.as_ref().map_or (true, | unit | fake_wesnoth::distance_between ([unit.x, unit.y], origin) <= unit.moves + 1 + extra_turns*unit.unit_type.max_moves) {
        self.unit_moves [index] = None;
      }
    }
  }
  
  pub fn collect_moves (&mut self, state: &fake_wesnoth::State)->Vec<(fake_wesnoth::Move, f64)> {
    self.calculate_moves (state);
  
    let mut results = vec![(fake_wesnoth::Move::EndTurn, 0.0)];
    for location in self.unit_moves.iter() {
      if let Some (moves) = location.as_ref() {
        results.extend (moves.into_iter().cloned());
      }
    }
  
    results
  }
}
  
  fn stats_badness (unit: & Unit, stats: & fake_wesnoth::CombatantStats)->f64 {
    (unit.hitpoints as f64 - stats.average_hitpoints)*(if unit.canrecruit {2.0} else {1.0})
    + stats.death_chance*(if unit.canrecruit {1000.0} else {50.0})
  }
  
  pub fn evaluate_unit_position (state: & State, unit: &Unit, x: i32, y: i32)->f64 {
    let mut result = 0.0;
    let location = state.get(x, y);
    let terrain_info = state.map.config.terrain_info.get (location.terrain).unwrap();
    let defense = *unit.unit_type.defense.get (location.terrain).unwrap() as f64;
    result -= defense / 10.0;
    if terrain_info.healing > 0 {
      let healing = ::std::cmp::min(terrain_info.healing, unit.unit_type.max_hitpoints - unit.hitpoints);
      result += healing as f64 - 0.8;
    }
    if unit.canrecruit {
      //result -= defense / 2.0;
      if !terrain_info.keep {result -= 6.0;}
    }
    result
  }
  
  pub fn evaluate_move (state: & State, input: & fake_wesnoth::Move, accurate: bool)->f64 {
    match input {
      &fake_wesnoth::Move::Move {src_x, src_y, dst_x, dst_y, moves_left} => {
        let unit = state.get (src_x, src_y).unit.as_ref().unwrap();
        let mut result =
          evaluate_unit_position (state, unit, dst_x, dst_y)
          - evaluate_unit_position (state, unit, src_x, src_y)
          + (random::<f64>() - moves_left as f64) / 100.0;
        let destination = state.get(dst_x, dst_y);
        let terrain_info = state.map.config.terrain_info.get (destination.terrain).unwrap();
        if terrain_info.village {
          if let Some(owner) = destination.village_owner.as_ref() {
            if state.is_enemy (unit.side, *owner) {
              result += 14.0;
            }
            else if *owner != unit.side {
              result -= 0.5;
            }
          }
          else {
            result += 10.0;
          }
        }
        
        result
      },
      &fake_wesnoth::Move::Attack {src_x, src_y, dst_x, dst_y, attack_x, attack_y, weapon} => {
        let mut attacker = state.get (src_x, src_y).unit.clone().unwrap();
        attacker.x = dst_x;
        attacker.y = dst_y;
        let defender = state.get (attack_x, attack_y).unit.as_ref().unwrap();
        let stats = if accurate {
          fake_wesnoth::simulate_combat (state, &attacker, defender, weapon, fake_wesnoth::CHOOSE_WEAPON)
        } else {
          fake_wesnoth::guess_combat (state, &attacker, defender, weapon, fake_wesnoth::CHOOSE_WEAPON)
        };
        evaluate_move (state, &fake_wesnoth::Move::Move {src_x, src_y, dst_x, dst_y, moves_left: 0}, accurate) + random::<f64>() + stats_badness (&defender, &stats.combatants [1]) - stats_badness (&attacker, &stats.combatants [0])
      }
      &fake_wesnoth::Move::Recruit {dst_x, dst_y, unit_type} => {
        let mut example = state.map.config.unit_type_examples.get (unit_type).unwrap().clone();
        example.side = state.current_side;
        example.x = dst_x;
        example.y = dst_y;
      
        random::<f64>()+100.0
      },
      &fake_wesnoth::Move::EndTurn => 0.0,
    }
  }

  
  pub fn evaluate_state (state: & State)->Vec<f64> {
    let mut ally_values = vec![0.0; state.sides.len()];
    let mut enemy_values = vec![0.0; state.sides.len()];
    for location in state.locations.iter() {
      if let Some(unit) = location.unit.as_ref() {
        let constant_value = unit.unit_type.cost as f64 + if unit.canrecruit {50.0} else {0.0};
        let value = constant_value*(1.0 + unit.hitpoints as f64/unit.unit_type.max_hitpoints as f64)/2.0;
        for index in 0..state.sides.len() {
          if state.is_enemy (unit.side, index) {
            enemy_values[index] += value;
          }
          else {
            ally_values[index] += value;
          }
        }
      }
      if let Some(owner) = location.village_owner {
        let value = 9.0;
        for index in 0..state.sides.len() {
          if state.is_enemy (owner, index) {
            enemy_values[index] += value;
          }
          else {
            ally_values[index] += value;
          }
        }
      }
    }
    (0..state.sides.len()).map(|side| {
      let ally = ally_values[side];
      let enemy = enemy_values[side];
      let diff = ally-enemy;
      let tot = ally+enemy;
      let ratio = diff/tot;
      ratio.abs().powf(1.0/3.0)*ratio.signum()
    }).collect()
  }

use std::collections::BinaryHeap;
use std::rc::Rc;
use std::cell::Cell;
use std::cmp::Ordering;
use smallvec::SmallVec;

pub fn play_turn_fast (state: &mut State, allow_combat: bool, stop_at_combat: bool)->Vec<Move> {
  #[derive (Clone)]
  struct Action {
    evaluation: f64,
    action: Move,
    source: [i32; 2],
    destination: Option<[i32; 2]>,
    valid: Cell<bool>,
  }
  struct ActionReference (Rc<Action>);
  #[derive (Clone)]
  struct LocationInfo {
    choices: Vec<Rc<Action>>,
    moves_attacking: Vec<Rc<Action>>,
    last_update: usize,
    distance_to_target: i32,
  }
  struct Info {
    locations: Vec<LocationInfo>,
    actions: BinaryHeap <ActionReference>,
    last_update: usize,
    allow_combat: bool,
  }
  
  impl Ord for ActionReference {
    fn cmp(&self, other: &Self) -> Ordering {
      self.0.evaluation.partial_cmp (&other.0.evaluation).unwrap()
    }
  }
  impl PartialEq for ActionReference {
    fn eq(&self, other: &Self) -> bool {
      self.0.evaluation == other.0.evaluation
    }
  }
  impl Eq for ActionReference {}
  impl PartialOrd for ActionReference {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
      Some(self.cmp(other))
    }
  }
  
  fn index (state: & State, x: i32,y: i32)->usize {((x-1)+(y-1)*state.map.width) as usize}
  let mut info = Info {
    locations: vec![LocationInfo {
      choices: Vec::new(),
      moves_attacking: Vec::new(),
      last_update: 0,
      distance_to_target: i32::max_value(),
    }; state.locations.len()],
    actions: BinaryHeap::new(),
    last_update: 0,
    allow_combat
  };
  
  fn evaluate (info: &Info, state: & State, action: & Move)-> f64 {
    let evaluation = evaluate_move (state, & action, false);
    match action {
      &fake_wesnoth::Move::Move {src_x, src_y, dst_x, dst_y, ..} => {
        if !state.get (src_x, src_y).unit.as_ref().unwrap().canrecruit {
          let distance_1 = info.locations [index (state, src_x, src_y)].distance_to_target;
          let distance_2 = info.locations [index (state, dst_x, dst_y)].distance_to_target;
          let yay = distance_1 - distance_2 - 2;
          //printlnerr!("{:?}", ((src_x, src_y), distance_1, ( dst_x, dst_y), distance_2));
          if yay > 0 {
            //printlnerr!("{:?}", ((src_x, src_y), distance_1, ( dst_x, dst_y), distance_2));
            return evaluation + 5.0+yay as f64
          }
        }
      },
      _=>(),
    }
    evaluation
  }
  fn generate_action (info: &mut Info, state: & State, unit: & Unit, action: Move) {
    let evaluation = evaluate (info, state, & action);
    if evaluation < 0.0 {return}
    let result = Rc::new (Action {
      evaluation, action: action.clone(), valid: Cell::new (true), source: [unit.x, unit.y],
      destination: match action {
        fake_wesnoth::Move::Move {dst_x, dst_y, ..}
        | fake_wesnoth::Move::Attack {dst_x, dst_y, ..}
        | fake_wesnoth::Move::Recruit {dst_x, dst_y, ..} => Some([dst_x, dst_y]),
        _=>None,
      }
    });
    info.actions.push (ActionReference (result.clone())) ;
    match action {
      fake_wesnoth::Move::Attack {attack_x, attack_y, ..} => {
        info.locations [index (state, attack_x, attack_y)].moves_attacking.push (result.clone());
      },
      _=>()
    }
    info.locations [index (state, unit.x, unit.y)].choices.push (result);
  }
  fn reevaluate_action (info: &mut Info, state: & State, action: & Action) {
    if !action.valid.get() { return }
    let mut new_action = (*action).clone();
    action.valid.set (false);
    new_action.evaluation = evaluate (info, state, & action.action);
    let new_action = Rc::new(new_action);
    info.actions.push (ActionReference (new_action.clone())) ;
    match action.action {
      fake_wesnoth::Move::Attack {attack_x, attack_y, ..} => {
        info.locations [index (state, attack_x, attack_y)].moves_attacking.push (new_action.clone());
      },
      _=>()
    }
    info.locations [index (state, action.source[0], action.source[1])].choices.push (new_action);
  }
  fn generate_reach (info: &mut Info, state: & State, unit: & Unit) {
    let reach = fake_wesnoth::find_reach (state, unit);
    for &(location, moves_left) in reach.list.iter() {
      if state.geta (location).unit.as_ref().map_or (false, | unit | unit.side != state.current_side || unit.moves == 0) {continue}
      
      if location != [unit.x, unit.y] {
        generate_action (info, state, unit, fake_wesnoth::Move::Move {
          src_x: unit.x, src_y: unit.y, dst_x: location [0], dst_y: location [1], moves_left: 0
        });
      }
      if info.allow_combat && unit.attacks_left >0 {
        for adjacent in fake_wesnoth::adjacent_locations (& state.map, location) {
          if let Some(neighbor) = state.geta (adjacent).unit.as_ref() {
            if state.is_enemy (unit.side, neighbor.side) {
              for index in 0..unit.unit_type.attacks.len() {
                generate_action (info, state, unit, fake_wesnoth::Move::Attack {
                  src_x: unit.x, src_y: unit.y, dst_x: location [0], dst_y: location [1],
                  attack_x: adjacent [0], attack_y: adjacent [1], weapon: index
                });
              }
            }
          }
        }
      }
    }
    if info.allow_combat && unit.canrecruit {
      for location in recruit_hexes (state, [unit.x, unit.y]) {
        for & recruit in state.sides [unit.side].recruits.iter() {
          if state.sides [unit.side].gold >= state.map.config.unit_type_examples [recruit].unit_type.cost {
            generate_action (info, state, unit, fake_wesnoth::Move::Recruit{
              dst_x: location [0], dst_y: location [1], unit_type: recruit
            });
          }
        }
      }
    }
  }
  
  let mut target_frontier:SmallVec <[[i32;2];32]> = SmallVec::new();
  for y in 1..(state.map.height+1) { for x in 1..(state.map.width+1) {
    let location = state.get (x,y);
    if let Some(unit) = location.unit.as_ref() {
      if unit.canrecruit && state.is_enemy (unit.side, state.current_side) {
        info.locations [index (state, x,y)].distance_to_target = 0;
        target_frontier.push ([unit.x, unit.y]);
      }
    }
    if state.map.config.terrain_info [location.terrain].village && location.village_owner.map_or (true, | owner | state.is_enemy (owner, state.current_side)) {
      info.locations [index (state, x,y)].distance_to_target = 0;
      target_frontier.push ([x,y]);
    }
  }}
  
  while !target_frontier.is_empty() {
    let mut next_frontier:SmallVec <[[i32;2];32]> = SmallVec::new();
    for location in target_frontier.iter() {
      let distance = info.locations [index (state, location [0], location [1])].distance_to_target;
      for adjacent in fake_wesnoth::adjacent_locations (& state.map, *location) {
        let other_distance = &mut info.locations [index (state, adjacent [0], adjacent [1])].distance_to_target;
        if *other_distance >distance + 1 {
          *other_distance = distance + 1;
          next_frontier.push (adjacent) ;
        }
      }
      //printlnerr!("{:?}", (location, distance));
    }
    
    target_frontier = next_frontier;
  }
  
  for y in 1..(state.map.height+1) { for x in 1..(state.map.width+1) {
    let location = state.get (x,y);
    if let Some(unit) = location.unit.as_ref() {
      if unit.side == state.current_side && (unit.moves > 0 || unit.attacks_left > 0) {
        generate_reach (&mut info, state, unit);
      }
    }
  }}
  
  fn update_info_after_move (info: &mut Info, state: & State, action: &Action) {
    
    match action.action {
      fake_wesnoth::Move::Move {src_x, src_y, dst_x, dst_y, moves_left} => {
        for other_action in info.locations [index (state, action.source [0], action.source [1])].choices.drain(..) {
          other_action.valid.set (false);
        }
      },
      fake_wesnoth::Move::Attack {src_x, src_y, dst_x, dst_y, attack_x, attack_y, weapon} => {
        for other_action in info.locations [index (state, action.source [0], action.source [1])].choices.drain(..) {
          other_action.valid.set (false);
        }
        if state.get (attack_x, attack_y).unit.is_none() {
          let modified: Vec<_> = info.locations [index(state, attack_x, attack_y)].moves_attacking.drain(..).filter (|m|m.valid.get()).map (| m| m.source).collect();
          info.last_update += 1;
          for source in modified {
            let index = index(state, source[0], source[1]);
            if info.locations [index].last_update < info.last_update {
              info.locations [index].last_update = info.last_update;
              for other_action in info.locations [index].choices.drain(..) {
                other_action.valid.set (false);
              }
              generate_reach (info, state, state.geta (source).unit.as_ref().unwrap());
            }
          }
        }
        else {
          for action in ::std::mem::replace(&mut info.locations [index(state, attack_x, attack_y)].moves_attacking, Vec::new()) {
            reevaluate_action (info, state, &action);
          }
        }
      },
      fake_wesnoth::Move::Recruit {dst_x, dst_y, ref unit_type} => {

      },
      fake_wesnoth::Move::EndTurn => {},
    }
  }
  
  let mut result = Vec::new() ;
  loop {
    let choice;
    let mut temporarily_invalid_choices: SmallVec<[ActionReference; 8]> = SmallVec::new() ;
    loop {
      let candidate = match info.actions.pop() {
        Some(a)=>a,
        _=>{
          result.push (fake_wesnoth::Move::EndTurn);
          fake_wesnoth::apply_move (state, &mut Vec::new(), & fake_wesnoth::Move::EndTurn) ;
          return result
        },
      };
      if !candidate.0.valid.get() {continue}
      if let Some(destination) = candidate.0.destination {
        if let Some(unit) = state.geta (destination).unit.as_ref() {
          if unit.side == state.current_side && unit.moves > 0 {
            temporarily_invalid_choices.push (candidate);
          }
          continue;
        }
      }
      if let fake_wesnoth::Move::Recruit {unit_type, ..} = candidate.0.action {
        if state.sides [state.current_side].gold < state.map.config.unit_type_examples [unit_type].unit_type.cost {
          continue;
        }
      }
      choice = candidate.0;
      break
    }
    
    result.push (choice.action.clone() );
    if stop_at_combat && match choice.action {fake_wesnoth::Move::Attack {..} => true, _=>false} {
      return result
    }
    
    info.actions.extend (temporarily_invalid_choices);
    fake_wesnoth::apply_move (state, &mut Vec::new(), & choice.action) ;
    if state.scores.is_some() {
      return result
    }
    update_info_after_move (&mut info, &*state, & choice) ;
  }
}
