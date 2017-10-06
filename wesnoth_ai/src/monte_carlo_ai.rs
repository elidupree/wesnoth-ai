use rand::{self, random, Rng};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::cell::RefCell;

use fake_wesnoth;
use rust_lua_shared::*;


#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Player <LookaheadPlayer: Fn(&State, usize)->Box<fake_wesnoth::Player>> {
  make_player: LookaheadPlayer,
  pub last_root: Option <Node>,
}

pub trait DisplayableNode {
  fn visits(&self)->i32;
  fn state (&self)->Option<Arc<State>>;
  fn info_text (&self)->String;
  fn descendants (&self)->Vec<&DisplayableNode>;
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Node {
  pub state: Arc<State>,
  pub visits: i32,
  pub total_score: f64,
  pub moves: Vec<ProposedMove>,
  pub turn: Arc<TurnGlobals>,
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct RaveScore {
  pub visits: i32,
  pub total_score: f64,
}

#[derive (Serialize, Deserialize, Debug, Default)]
pub struct TurnGlobals {
  pub rave_scores: Mutex<HashMap<fake_wesnoth::Move, RaveScore>>,
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct TreeGlobals {
  pub starting_turn: i32,
  pub starting_side: usize,
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct ProposedMove {
  pub action: Move,
  pub visits: i32,
  pub total_score: f64,
  pub determined_outcomes: Vec<Node>,
}

impl DisplayableNode for Node {
  fn visits(&self)->i32 {self.visits}
  fn state (&self)->Option<Arc<State>> {Some(self.state.clone())}
  fn info_text (&self)->String {format!("{:.2}\n{}", self.total_score/self.visits as f64, self.visits)}
  fn descendants (&self)->Vec<&DisplayableNode> {
    self.moves.iter().map (| proposed | proposed as &DisplayableNode).collect()
  }
}
impl DisplayableNode for ProposedMove {
  fn visits(&self)->i32 {self.visits}
  fn state (&self)->Option<Arc<State>> {None}
  fn info_text (&self)->String {format!("{:.2}\n{}", self.total_score/self.visits as f64, self.visits)}
  fn descendants (&self)->Vec<&DisplayableNode> {
    self.determined_outcomes.iter().map (| outcome | outcome as &DisplayableNode).collect()
  }
}

use fake_wesnoth::{State, Unit, Move};

impl<LookaheadPlayer: Fn(&State, usize)->Box<fake_wesnoth::Player>> fake_wesnoth::Player for Player<LookaheadPlayer> {
  fn choose_move (&mut self, state: & State)->Move {
    let mut root = Node {
      state: Arc::new (state.clone()),
      visits: 0,
      total_score: 0.0,
      moves: Vec::new(),
      turn: Arc::default(),
    };
    let mut globals = TreeGlobals {
      starting_turn: state.turn,
      starting_side: state.current_side,
    };
    for _ in 0..7000 {
      self.step_into_node (&mut globals, &mut root);
    }
    
    let result = root.moves.iter()
      .max_by_key (|a| a.visits)
      .unwrap().action.clone();
    
    self.last_root = Some(root);
    result
  }
}

impl<LookaheadPlayer: Fn(&State, usize)->Box<fake_wesnoth::Player>> Player<LookaheadPlayer> {
  pub fn new(make_player: LookaheadPlayer)->Self where Self: Sized {Player{make_player: make_player, last_root: None}}
  /*pub fn evaluate_state (&self, state: & State)->Vec<f64> {
    let mut total_score = 0f64;
    let starting_turn = state.turn;
    let starting_side = state.current_side;
    
    let mut playout_state = state.clone();
    let mut players: Vec<_> = playout_state.sides.iter().enumerate().map (| (index, _side) | (self.make_player)(&playout_state, index)).collect();
    while playout_state.scores.is_none() && !(playout_state.current_side == state.current_side && playout_state.turn == starting_turn + 2) {
      let choice = players [playout_state.current_side].choose_move (& playout_state) ;
      fake_wesnoth::apply_move (&mut playout_state, &mut players, & choice);
    }
    if let Some(scores) = playout_state.scores {
      return scores;
    }
    ::naive_ai::evaluate_state(&playout_state) //vec![0.0; state.sides.len()]
  }*/
  
  fn step_into_node (&self, globals: &mut TreeGlobals, node: &mut Node)->Vec<f64> {
    let scores = if let Some(scores) = node.state.scores.clone() {
      scores
    }
    /*else if node.visits == 0 {
      self.evaluate_state (&node.state)
    }*/
    else if node.state.current_side == globals.starting_side && node.state.turn == globals.starting_turn + 3 {
      ::naive_ai::evaluate_state(&node.state)
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
      let c = 0.2; //2.0;
      let c_log_visits = c*((node.visits+1) as f64).ln();
      let priority_state = node.state.clone();
      let priority_turn = node.turn.clone();
      let priority = | proposed: &ProposedMove | {
        let rave_score = priority_turn.rave_scores.lock().unwrap().get(&proposed.action).cloned().unwrap_or(RaveScore { visits: 0, total_score: 0.0, });
        let naive_weight = 0.000001;
        let rave_weight = rave_score.visits as f64;
        let exact_weight = (proposed.visits*proposed.visits) as f64;
        let total_weight = naive_weight + rave_weight + exact_weight;
               
        let uncertainty_bonus = if rave_score.visits + proposed.visits == 0 { 100000.0 }
          else { (c_log_visits/(
            // Note: From each position, there are a lot of possible moves – let's say 100. This causes a problem: When exploring a NEW move, we would normally try every follow-up move before repeating any of them. But that potentially means 100 trials of bad follow-ups, making the initial move look bad and not get explored further. So we want to play mostly good follow-ups for a while before exploring others. For this reason, we allow the RAVE trials to reduce the uncertainty bonus. This isn't perfect (for instance, when a unit makes an incomplete move, it still has many possible useless follow-ups that are completely fresh), but it empirically helped a lot when I originally implemented it.
            // I think there's been a slight problem where the RAVE certainty bonus becomes arbitrarily high, permanently burying a good move that got unlucky at first. So we limit the RAVE bonus to a fixed maximum, allowing the bad follow-ups to EVENTUALLY get more exploration.
            (if rave_score.visits > 6 {6.0} else {rave_score.visits as f64}/2.0)
            + proposed.visits as f64
          )).sqrt() };
        
        let naive_score = ::naive_ai::evaluate_move (&priority_state, &proposed.action);
        if naive_score.abs() > 10000.0 { printlnerr!("Warning: unexpectedly high naive eval"); }
        let mut total_score = naive_score*(naive_weight/total_weight);
        if rave_score.visits > 0 {
          total_score += (rave_score.total_score/rave_score.visits as f64) * (rave_weight/total_weight);
        }
        if proposed.visits > 0 {
          total_score += (proposed.total_score/proposed.visits as f64) * (exact_weight/total_weight);
        }
        total_score + uncertainty_bonus
      };
      let choice = node.moves.iter_mut()
          .max_by (|a,b| priority(a).partial_cmp(&priority(b)).unwrap())
          .unwrap();
      let next_node = if choice.determined_outcomes.is_empty() || (match choice.action {fake_wesnoth::Move::Attack{..}=>true,_=>false} && choice.visits > (1<<choice.determined_outcomes.len())) {
        let mut state_after = (*node.state).clone();
        fake_wesnoth::apply_move (&mut state_after, &mut Vec::new(), & choice.action);
        
        choice.determined_outcomes.push(Node {
          state: Arc::new(state_after), visits: 0, total_score: 0.0, moves: Vec::new(), turn: match choice.action {fake_wesnoth::Move::EndTurn=>Arc::default(),_=>node.turn.clone()} ,
        });
        choice.determined_outcomes.last_mut().unwrap()
      }
      else {
        rand::thread_rng().choose_mut(&mut choice.determined_outcomes).unwrap()
      };

      let scores = self.step_into_node (globals, next_node);
      
      choice.total_score += scores[node.state.current_side];
      choice.visits += 1;
      if match choice.action {fake_wesnoth::Move::EndTurn=>false,_=>true} {
        let mut guard = node.turn.rave_scores.lock().unwrap();
        let rave_score = guard.entry (choice.action.clone()).or_insert (RaveScore { visits: 0, total_score: 0.0, });
        rave_score.total_score += scores[node.state.current_side];
        rave_score.visits += 1;
      }
      
      scores
    };
    
    node.total_score += scores[node.state.current_side];
    node.visits += 1;
    
    scores
  }
}

