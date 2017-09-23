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
            let evaluation = self.evaluate_move (state, &action);
            (action, evaluation)
          }).collect());
        }
      }
    }
  }
  
  pub fn invalidate_moves (&mut self, state: &fake_wesnoth::State, origin: [i32; 2], extra_turns: i32) {
    for (index, location) in state.locations.iter().enumerate() {
      if location.unit.as_ref().map_or (true, | unit | fake_wesnoth::distance_between ([unit.x, unit.y], origin) <= unit.moves + 1 + extra_turns*unit.max_moves) {
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
  
  fn stats_badness (&self, unit: & Unit, stats: & fake_wesnoth::AnalyzedStats)->f64 {
    (unit.hitpoints as f64 - stats.average_hitpoints)*(if unit.canrecruit {2.0} else {1.0})
    + stats.death_chance*(if unit.canrecruit {1000.0} else {50.0})
  }
  
  pub fn evaluate_move (&self, state: & State, input: & fake_wesnoth::Move)->f64 {
    match input {
      &fake_wesnoth::Move::Move {src_x, src_y, moves_left, ..} => {
        let unit = state.get (src_x, src_y).unit.as_ref().unwrap();
        if unit.canrecruit {return -1.0;}
        random::<f64>() - moves_left as f64
      },
      &fake_wesnoth::Move::Attack {src_x, src_y, dst_x, dst_y, attack_x, attack_y, weapon} => {
        let mut attacker = state.get (src_x, src_y).unit.clone().unwrap();
        if attacker.canrecruit && (src_x != dst_x || src_y != dst_y){return -1.0;}
        attacker.x = dst_x;
        attacker.y = dst_y;
        let defender = state.get (attack_x, attack_y).unit.as_ref().unwrap();
        let stats = fake_wesnoth::simulate_and_analyze (state, &attacker, defender, weapon, usize::max_value() - 1);
        random::<f64>() + self.stats_badness (&defender, &stats.1) - self.stats_badness (&attacker, &stats.0)
      }
      &fake_wesnoth::Move::Recruit {dst_x, dst_y, ref unit_type} => {
        let mut example = state.map.config.unit_type_examples.get (unit_type).unwrap().clone();
        example.side = state.current_side;
        example.x = dst_x;
        example.y = dst_y;
      
        random::<f64>()+100.0
      },
      &fake_wesnoth::Move::EndTurn => unreachable!(),
    }
  }
}

