use rand::{random};
use std::marker::PhantomData;

use fake_wesnoth;
use rust_lua_shared::*;


#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Player <LookaheadPlayer: Fn(&State, usize)->Box<fake_wesnoth::Player>> {
  make_player: LookaheadPlayer,
}

use fake_wesnoth::{State, Unit, Move};

impl<LookaheadPlayer: Fn(&State, usize)->Box<fake_wesnoth::Player>> fake_wesnoth::Player for Player<LookaheadPlayer> {
  fn choose_move (&mut self, state: & State)->Move {
    state.locations.iter()
      .filter_map (| location | location.unit.as_ref())
      .filter(|unit| unit.side == state.current_side)
      .flat_map (|unit| possible_unit_moves (state, unit).into_iter())
      .chain(::std::iter::once (fake_wesnoth::Move::EndTurn))
      .map (| action | {
        let evaluation = self.evaluate_move (state, &action);
        (action, evaluation)
      })
      .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
      .unwrap().0
  }
}

impl<LookaheadPlayer: Fn(&State, usize)->Box<fake_wesnoth::Player>> Player<LookaheadPlayer> {
  pub fn new(make_player: LookaheadPlayer)->Self where Self: Sized {Player{make_player: make_player}}
  pub fn evaluate_move (&self, state: & State, input: & fake_wesnoth::Move)->f64 {
    let mut total_score = 0f64;
    let starting_turn = state.turn;
    for _ in 0..100 {
      let mut playout_state = state.clone();
      let mut players: Vec<_> = playout_state.sides.iter().enumerate().map (| (index, _side) | (self.make_player)(&playout_state, index)).collect();
      fake_wesnoth::apply_move (&mut playout_state, &mut players, & input);
      while playout_state.scores.is_none() && playout_state.turn < starting_turn + 40 {
        let choice = players [playout_state.current_side].choose_move (& playout_state) ;
        fake_wesnoth::apply_move (&mut playout_state, &mut players, & choice);
      }
      if let Some(scores) = playout_state.scores {
        total_score += scores [state.current_side];
      }
    }
    println!("evaluated move {:?} with score {}", input, total_score);
    total_score
  }
}

