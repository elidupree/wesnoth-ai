use rand::{random};
use std::marker::PhantomData;

use fake_wesnoth;
use rust_lua_shared::*;


#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Player <LookaheadPlayer: Fn(&State, usize)->Box<fake_wesnoth::Player>> {
  make_player: LookaheadPlayer,
  root: Node
}

struct Node {
  state: Arc<State>,
  visits: i32,
  moves: Vec<ProposedMove>,
}

struct ProposedMove {
  action: Move,
  visits: i32,
  total_score: f64,
  determined_outcomes: Vec<Node>,
}

use fake_wesnoth::{State, Unit, Move};

impl<LookaheadPlayer: Fn(&State, usize)->Box<fake_wesnoth::Player>> fake_wesnoth::Player for Player<LookaheadPlayer> {
  fn choose_move (&mut self, state: & State)->Move {
    self.root = Note {
      state: Arc::new (state.clone()),
      visits: 0,
      moves: Vec::new(),
    };
    for _ in 0..100 {
      self.step ();
    }
    self.root.moves.iter()
      .max_by_key (|a| a.visits)
      .unwrap().action
  }
}

impl<LookaheadPlayer: Fn(&State, usize)->Box<fake_wesnoth::Player>> Player<LookaheadPlayer> {
  pub fn new(make_player: LookaheadPlayer)->Self where Self: Sized {Player{make_player: make_player}}
  /*pub fn evaluate_move (&self, state: & State, input: & fake_wesnoth::Move)->f64 {
    let mut total_score = 0f64;
    let starting_turn = state.turn;
    for _ in 0..100 {
      let mut playout_state = state.clone();
      let mut players: Vec<_> = playout_state.sides.iter().enumerate().map (| (index, _side) | (self.make_player)(&playout_state, index)).collect();
      fake_wesnoth::apply_move (&mut playout_state, &mut players, & input);
      while playout_state.scores.is_none() && playout_state.turn < starting_turn + 10 {
        let choice = players [playout_state.current_side].choose_move (& playout_state) ;
        fake_wesnoth::apply_move (&mut playout_state, &mut players, & choice);
      }
      if let Some(scores) = playout_state.scores {
        total_score += scores [state.current_side];
      }
    }
    println!("evaluated move {:?} with score {}", input, total_score);
    total_score
  }*/
  
  
  pub fn step (&mut self) {
    let mut node = &mut self.root;
    while node.visits > 0 {
      if node.moves.is_empty() { node.moves = node.state.locations.iter()
        .filter_map (| location | location.unit.as_ref())
        .filter(|unit| unit.side == state.current_side)
        .flat_map (|unit| possible_unit_moves (state, unit).into_iter())
        .chain(::std::iter::once (fake_wesnoth::Move::EndTurn))
        .map(| action | {
          ProposedMove {
            action: action, total_score: 0.0, visits: 0, determined_outcomes: Vec::new(),
          }
        });
      }
      let log_visits = 2.0*(node.visits as f64).ln();
      let priority = | proposed | proposed.total_score/proposed.visits as f64 + (log_visits/(proposed.visits as f64)).sqrt();
      let choice = node.moves.iter()
          .max_by (|a,b| priority(a).cmp(priority(b)))
          .unwrap()
      if choice.determined_outcomes.is_empty() || (match choice.action {Attack{..}=>true,_=>false} && choice.determined_outcomes.len()) {
        let mut state_after = node.state.clone();
        fake_wesnoth::apply_move (&mut state_after, None, & action);
        
        choice.determined_outcomes.push(Node {
          state: state_after, visits: 0, moves: Vec::new(),
        });
        node = choice.determined_outcomes.last_mut();
      }
      else {
        node = rand::thread_rng()::choose(choice.determined_outcomes).unwrap()
      }
    }
    
    let score = self.evaluate_state (&node.state);
    
  }
}

