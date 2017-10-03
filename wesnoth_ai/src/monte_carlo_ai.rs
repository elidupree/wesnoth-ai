use rand::{self, random, Rng};
use std::marker::PhantomData;
use std::sync::Arc;

use fake_wesnoth;
use rust_lua_shared::*;


#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Player <LookaheadPlayer: Fn(&State, usize)->Box<fake_wesnoth::Player>> {
  make_player: LookaheadPlayer,
}

#[derive (Clone, Serialize, Deserialize, Debug)]
struct Node {
  state: Arc<State>,
  visits: i32,
  total_score: f64,
  moves: Vec<ProposedMove>,
}

#[derive (Clone, Serialize, Deserialize, Debug)]
struct ProposedMove {
  action: Move,
  visits: i32,
  total_score: f64,
  determined_outcomes: Vec<Node>,
}

use fake_wesnoth::{State, Unit, Move};

impl<LookaheadPlayer: Fn(&State, usize)->Box<fake_wesnoth::Player>> fake_wesnoth::Player for Player<LookaheadPlayer> {
  fn choose_move (&mut self, state: & State)->Move {
    let mut root = Node {
      state: Arc::new (state.clone()),
      visits: 0,
      total_score: 0.0,
      moves: Vec::new(),
    };
    for _ in 0..100 {
      self.step_into_node (&mut root);
    }
    root.moves.iter()
      .max_by_key (|a| a.visits)
      .unwrap().action.clone()
  }
}

impl<LookaheadPlayer: Fn(&State, usize)->Box<fake_wesnoth::Player>> Player<LookaheadPlayer> {
  pub fn new(make_player: LookaheadPlayer)->Self where Self: Sized {Player{make_player: make_player}}
  pub fn evaluate_state (&self, state: & State)->f64 {
    let mut total_score = 0f64;
    let starting_turn = state.turn;
    
    let mut playout_state = state.clone();
    let mut players: Vec<_> = playout_state.sides.iter().enumerate().map (| (index, _side) | (self.make_player)(&playout_state, index)).collect();
    while playout_state.scores.is_none() && playout_state.turn < starting_turn + 10 {
      let choice = players [playout_state.current_side].choose_move (& playout_state) ;
      fake_wesnoth::apply_move (&mut playout_state, &mut players, & choice);
    }
    if let Some(scores) = playout_state.scores {
      total_score += scores [state.current_side];
    }
    total_score
  }
  
  fn step_into_node (&self, node: &mut Node)->f64 {
    let score = if node.visits == 0 {
      self.evaluate_state (&node.state)
    }
    else {
      if node.moves.is_empty() { node.moves = node.state.locations.iter()
        .filter_map (| location | location.unit.as_ref())
        .filter(|unit| unit.side == node.state.current_side)
        .flat_map (|unit| possible_unit_moves (&node.state, unit).into_iter())
        .chain(::std::iter::once (fake_wesnoth::Move::EndTurn))
        .map(| action | {
          ProposedMove {
            action: action, total_score: 0.0, visits: 0, determined_outcomes: Vec::new(),
          }
        }).collect();
      }
      let log_visits = 2.0*(node.visits as f64).ln();
      let priority = | proposed: &ProposedMove | proposed.total_score/proposed.visits as f64 + (log_visits/(proposed.visits as f64)).sqrt();
      let choice = node.moves.iter_mut()
          .max_by (|a,b| priority(a).partial_cmp(&priority(b)).unwrap())
          .unwrap();
      let next_node = if choice.determined_outcomes.is_empty() || (match choice.action {fake_wesnoth::Move::Attack{..}=>true,_=>false} && choice.visits > (1>>choice.determined_outcomes.len())) {
        let mut state_after = (*node.state).clone();
        fake_wesnoth::apply_move (&mut state_after, &mut Vec::new(), & choice.action);
        
        choice.determined_outcomes.push(Node {
          state: Arc::new(state_after), visits: 0, total_score: 0.0, moves: Vec::new(),
        });
        choice.determined_outcomes.last_mut().unwrap()
      }
      else {
        rand::thread_rng().choose_mut(&mut choice.determined_outcomes).unwrap()
      };

      let score = self.step_into_node (next_node);
      
      choice.total_score += score;
      choice.visits += 1;
      
      score
    };
    
    node.total_score += score;
    node.visits += 1;
    
    score
  }
}

