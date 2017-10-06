use rand::{self, random, Rng};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::cell::RefCell;
use std::cmp::min;

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





/*struct Attack {
  src_x: i32, src_y: i32, dst_x: i32, dst_y: i32, attack_x: i32, attack_y: i32, weapon: usize,
}
struct Move {
  src_x: i32, src_y: i32, dst_x: i32, dst_y: i32,
}*/
#[derive (Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
struct SpaceClearingMoves {
  planned_moves: Vec<([i32; 2], [i32; 2])>,
  desired_moves: Vec<([i32; 2], [i32; 2])>,
  follow_up: Box<GenericNodeType>,
}

fn to_wesnoth_move (state_globals: &StateGlobals, locations: ([i32; 2], [i32; 2]))->fake_wesnoth::Move {
  fake_wesnoth::Move::Move {
    src_x: locations.0[0], src_y: locations.0[1],
    dst_x: locations.1[0], dst_y: locations.1[1],
    moves_left: state_globals.reaches[&locations.0][&locations.1]
  }
}

#[derive (Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum GenericNodeType {
  ChooseAttack,
  ChooseHowToClearSpace(SpaceClearingMoves),
  ExecuteAttack (fake_wesnoth::Move),
  FinishTurnLazily (Option<fake_wesnoth::Move>),
}
use self::GenericNodeType::{ChooseAttack, ChooseHowToClearSpace, ExecuteAttack, FinishTurnLazily};

#[derive (Serialize, Deserialize, Debug, Default)]
pub struct GenericTurnGlobals {
  pub rave_scores: Mutex<HashMap<GenericNodeType, RaveScore>>,
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct StateGlobals {
  pub reaches: HashMap<[i32;2], HashMap<[i32;2], i32>>
}
impl StateGlobals {
  pub fn new (state: & State)->StateGlobals {
    StateGlobals {
      reaches: state.locations.iter()
        .filter_map (| location | location.unit.as_ref())
        .map (| unit |
          (
            [unit.x, unit.y],
            fake_wesnoth::find_reach (state, unit).into_iter().collect()
          )
        ).collect()
    }
  }
}

pub struct GenericNode {
  pub state: Arc<State>,
  pub state_globals: Arc<StateGlobals>,
  pub turn: Arc<GenericTurnGlobals>,
  pub tree: Arc<TreeGlobals>,
  pub visits: i32,
  pub total_score: f64,
  pub choices: Vec<GenericNode>,
  pub node_type: GenericNodeType,
}

enum StepIntoResult {
  PlayedOutWithScores(Vec<f64>),
  TurnedOutToBeImpossible,
}
use self::StepIntoResult::{PlayedOutWithScores, TurnedOutToBeImpossible};

pub fn choose_moves (state: & State)->(GenericNode, Vec<Move>) {
  let mut root = GenericNode {
    state: Arc::new (state.clone()),
    turn: Arc::default(),
    tree: Arc::new(TreeGlobals {
      starting_turn: state.turn,
      starting_side: state.current_side,
    }),
    state_globals: Arc::new(StateGlobals::new (state)),
    visits: 0,
    total_score: 0.0,
    choices: Vec::new(),
    node_type: ChooseAttack, 
  };

  for _ in 0..7000 {
    root.step_into ();
  }
  
  let mut result = Vec::new();
  {
  let mut node = &root;
  loop {
    match node.node_type.clone() {
      ChooseHowToClearSpace(moves) => {
        result.clear();
        result.extend (moves.planned_moves.iter().map (| &locations | to_wesnoth_move (&node.state_globals, locations)));
        result.extend (moves.desired_moves.iter().map (| &locations | to_wesnoth_move (&node.state_globals, locations)));
      },
      ExecuteAttack (action) => {
        result.push (action);
        break;
      },
      FinishTurnLazily (Some(action)) => {
        result.push (action);
        if let Some (&fake_wesnoth::Move::EndTurn) = result.last() {
          break;
        }
      },
      _=>(),
    };
    node = node.choices.iter()
      .max_by_key (|a| a.visits)
      .unwrap();
  }
  }
    
  (root, result)
}

impl GenericNode {
  fn new_child(&self, node_type: GenericNodeType) -> GenericNode {
    GenericNode {
      state: self.state.clone(), turn: self.turn.clone(), tree: self.tree.clone(), state_globals: self.state_globals.clone(),
      visits: 0, total_score: 0.0,
      choices: Vec::new(),
      node_type: node_type,
    }
  }
  fn push_new_child(&mut self, node_type: GenericNodeType) {
    let new_child = self.new_child (node_type);
    self.choices.push (new_child);
  }
  
  fn set_state (&mut self, new_state: State) {
    self.state = Arc::new(new_state);
    self.state_globals = Arc::new(StateGlobals::new (&self.state));
  }
  
  fn init_attack_choices (&mut self) {
    let mut new_children = Vec::new();
    for unit in self.state.locations.iter()
        .filter_map (| location | location.unit.as_ref())
        .filter(|unit| unit.side == self.state.current_side && unit.attacks_left > 0) {
      for location in fake_wesnoth::find_reach (&self.state, unit) {
        let unit_there = self.state.get (location.0 [0], location.0 [1]).unit.as_ref();
        if unit_there.map_or (false, | other | other.side != self.state.current_side) { continue; }
        for adjacent in fake_wesnoth::adjacent_locations (& self.state.map, location.0) {
          if let Some (neighbor) = self.state.get (adjacent [0], adjacent [1]).unit.as_ref() {
            if !self.state.is_enemy (unit.side, neighbor.side) { continue; }
            for index in 0..unit.unit_type.attacks.len() {
              let attack = fake_wesnoth::Move::Attack {
                src_x: location.0 [0], src_y: location.0 [1],
                dst_x: location.0 [0], dst_y: location.0 [1],
                attack_x: adjacent [0], attack_y: adjacent [1],
                weapon: index,
              };
              new_children.push(self.new_child (
                GenericNodeType::ChooseHowToClearSpace(SpaceClearingMoves {
                  desired_moves: vec![([unit.x,unit.y],location.0)],
                  planned_moves: Vec::new(),
                  follow_up: Box::new(ExecuteAttack(attack)),
                })
              ));
            }
          }
        }
      }
    }
    self.choices.extend(new_children);
    self.push_new_child (GenericNodeType::FinishTurnLazily(None));
  }
  
  fn init_clearspace_choices (&mut self) {
    let mut movers = HashMap::new();
    let state_hack = self.state.clone();
    let state_globals = self.state_globals.clone();
    let get = | movers: &HashMap<[i32;2], Option<[i32;2]>>, location |->Option<[i32;2]> {
      movers.get (&location).cloned().unwrap_or_else (|| {
        state_hack.geta (location).unit.as_ref().map (| unit | [unit.x, unit.y])
      })
    };
    let info = match self.node_type {ChooseHowToClearSpace(ref info) => info.clone(), _=>unreachable!()};
    for (index, planned_move) in info.planned_moves.iter().enumerate() {
      let destination = get (&movers, planned_move.1);
      if destination.is_none() {
        let source = get (&movers, planned_move.0).unwrap();
        movers.insert (planned_move.0, None);
        movers.insert (planned_move.1, Some(source));
      }
      else {
        for adjacent in fake_wesnoth::adjacent_locations (& self.state.map, planned_move.1) {
          if self.state.geta (adjacent).unit.as_ref().map_or (false, | other | other.side != self.state.current_side) { continue; }
          
          // we may try moving the blocking unit (at destination) out of the way first
          let mut new_info = info.clone();
          let previous = new_info.planned_moves.iter().position (| k | k.1 == planned_move.1);
          // TODO: should we exclude changes that reduce the amount a unit is moving?
          let (blocker_original_location, insert_index) = match previous {
            None => (planned_move.1, index),
            Some (previous) => (new_info.planned_moves.remove (previous).0, min(index, previous)),
          };
          if let Some (moves_left) = state_globals.reaches [&blocker_original_location].get (&adjacent) {
            // all changes must increase the amount of movement, to avoid infinite loops
            let previous_moves_left = state_globals.reaches [&blocker_original_location][&planned_move.1];
            if *moves_left < previous_moves_left {
              new_info.planned_moves.insert (insert_index, (blocker_original_location, adjacent));
              self.push_new_child (GenericNodeType::ChooseHowToClearSpace(new_info));
            }
          }
          
          // we may also try continuing the movement of the blocked unit
          if let Some (moves_left) = state_globals.reaches [&planned_move.0].get(&adjacent) {
            let mut new_info = info.clone();
            // all changes must increase the amount of movement, to avoid infinite loops
            let previous_moves_left = state_globals.reaches [&planned_move.0][&planned_move.1];
            if *moves_left < previous_moves_left {
              new_info.planned_moves[index].1 = adjacent;
              self.push_new_child (GenericNodeType::ChooseHowToClearSpace(new_info));
            }
          }
        }
        return;
      }
    }
    for desired_move in info.desired_moves.iter() {
      if self.state.geta (desired_move.1).unit.is_some() {
        for adjacent in fake_wesnoth::adjacent_locations (& self.state.map, desired_move.1) {
          if self.state.geta (adjacent).unit.as_ref().map_or (false, | other | other.side != self.state.current_side) { continue; }
          
          let mut new_info = info.clone();
          if let Some (_moves_left) = state_globals.reaches [&desired_move.1].get(&adjacent) {
            new_info.planned_moves.insert (0, (desired_move.1, adjacent));
            self.push_new_child (GenericNodeType::ChooseHowToClearSpace(new_info));
          }
        }
        return;
      }
    }
    
    let mut new_child = self.new_child (*info.follow_up);
    let mut state_after = (*self.state).clone();
    for &locations in info.planned_moves.iter().chain (info.desired_moves.iter()) {
      fake_wesnoth::apply_move (&mut state_after, &mut Vec::new(), & to_wesnoth_move (&state_globals, locations));
    }
    new_child.set_state(state_after);
    self.choices.push (new_child);
  }

  fn update_choices (&mut self) {
    match self.node_type.clone() {
      ChooseAttack => {
        if self.choices.is_empty() {
          self.init_attack_choices();
        }
      },
      ChooseHowToClearSpace(_) => {
        if self.choices.is_empty() {
          self.init_clearspace_choices();
        }
      }
      ExecuteAttack(_) => (),
      FinishTurnLazily(previous_action) => {
        if self.choices.is_empty() {
          match previous_action {
            Some(fake_wesnoth::Move::EndTurn)=>{
              self.push_new_child (ChooseAttack);
            },
            _=> {
          
          let next_move = self.state.locations.iter()
            .filter_map (| location | location.unit.as_ref())
            .flat_map (| unit | {
              let coords = [unit.x, unit.y];
              let state_hack = self.state.clone();
              self.state_globals.reaches [&[unit.x, unit.y]].iter().filter_map(move | (destination, moves_left) | {
                if state_hack.geta(*destination).unit.is_none() {
                  Some(fake_wesnoth::Move::Move {
                    src_x: coords[0], src_y: coords[1],
                    dst_x: destination [0], dst_y: destination [1],
                    moves_left: *moves_left
                  })
                }
                else { None }
              })
            })
            .chain(::std::iter::once (fake_wesnoth::Move::EndTurn))
            .map (| action | (action.clone(), ::naive_ai::evaluate_move (& self.state, &action)))
            .max_by (|a,b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap().0;
          
          let mut state_after = (*self.state).clone();
          fake_wesnoth::apply_move (&mut state_after, &mut Vec::new(), & next_move);
          
          let mut new_child = self.new_child (FinishTurnLazily (Some (next_move)));
          new_child.set_state(state_after);
          self.choices.push (new_child);
            },
          };
        }
      }
    }
  }


  fn step_into (&mut self)->StepIntoResult {
    let scores = if let Some(scores) = self.state.scores.clone() {
      scores
    }
    else if self.state.current_side == self.tree.starting_side && self.state.turn == self.tree.starting_turn + 3 {
      ::naive_ai::evaluate_state(&self.state)
    }
    else {
      self.update_choices ();
      
      
      let c = 0.2; //2.0;
      let c_log_visits = c*((self.visits+1) as f64).ln();
      
      let priority_state = self.state.clone();
      let priority_turn = self.turn.clone();
      let priority_type = self.node_type.clone();
      let priority = | choice: &GenericNode| {
        let mut rave_score = RaveScore { visits: 0, total_score: 0.0, };
        let mut naive_score = 0.0;
        
        match (&priority_type, &choice.node_type) {
          (&ChooseAttack, &ChooseHowToClearSpace(ref info)) => {
            if let Some(scores) = priority_turn.rave_scores.lock().unwrap().get(& info.follow_up).cloned() {
              rave_score = scores;
            }
            match *info.follow_up {
              ExecuteAttack(fake_wesnoth::Move::Attack {
                src_x, src_y,
                dst_x, dst_y, attack_x, attack_y, weapon,
              }) => {
                naive_score = ::naive_ai::evaluate_move (&priority_state, &fake_wesnoth::Move::Attack {src_x: info.desired_moves [0].0[0], src_y: info.desired_moves [0].0[1], dst_x, dst_y, attack_x, attack_y, weapon, });
              },
              _ => (),
            }
          },
          _ => (),
        };
        let naive_weight = 0.000001;
        let rave_weight = rave_score.visits as f64;
        let exact_weight = (choice.visits*choice.visits) as f64;
        let total_weight = naive_weight + rave_weight + exact_weight;
               
        let uncertainty_bonus = if rave_score.visits + choice.visits == 0 { 100000.0 }
          else { (c_log_visits/(
            (if rave_score.visits > 6 {6.0} else {rave_score.visits as f64}/2.0)
            + choice.visits as f64
          )).sqrt() };
        
        if naive_score.abs() > 10000.0 { printlnerr!("Warning: unexpectedly high naive eval"); }
        let mut total_score = naive_score*(naive_weight/total_weight);
        if rave_score.visits > 0 {
          total_score += (rave_score.total_score/rave_score.visits as f64) * (rave_weight/total_weight);
        }
        if choice.visits > 0 {
          total_score += (choice.total_score/choice.visits as f64) * (exact_weight/total_weight);
        }
        total_score + uncertainty_bonus
      };
      let choice = match self.node_type {
        ExecuteAttack(ref attack) => {
          if self.choices.is_empty() || self.visits > (1<<self.choices.len()) {
            let mut state_after = (*self.state).clone();
            fake_wesnoth::apply_move (&mut state_after, &mut Vec::new(), attack);
            let mut new_child = self.new_child(ChooseAttack);
            new_child.set_state(state_after);
            self.choices.len() - 1
          }
          else {
            rand::thread_rng().gen_range(0, self.choices.len())
          }
        },
        _ => match (0.. self.choices.len())
          .max_by (|a,b| priority(&self.choices[*a]).partial_cmp(&priority(&self.choices[*b])).unwrap()) {
          None => return TurnedOutToBeImpossible,
          Some(k) => k,
        },
      };
      
      let scores = match self.choices [choice].step_into () {
        PlayedOutWithScores(scores) => scores,
        TurnedOutToBeImpossible => {
          match self.step_into() {
            PlayedOutWithScores(scores) => scores,
            TurnedOutToBeImpossible => return TurnedOutToBeImpossible,
          }
        },
      };
      
      scores
    };
    
    self.total_score += scores[self.state.current_side];
    self.visits += 1;
    
    if match self.node_type {ExecuteAttack(_)=>true,_=>false} {
      let mut guard = self.turn.rave_scores.lock().unwrap();
      let rave_score = guard.entry (self.node_type.clone()).or_insert (RaveScore { visits: 0, total_score: 0.0, });
      rave_score.total_score += scores[self.state.current_side];
      rave_score.visits += 1;
    }
      
    PlayedOutWithScores(scores)
  }

}
