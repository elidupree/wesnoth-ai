use rand::{self, random, Rng};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::collections::{HashMap, HashSet, BTreeMap};
use std::collections::Bound::{Excluded, Unbounded};
use std::cell::RefCell;
use std::cmp::min;
use std::any::Any;
use std::fmt::{self,Debug};
use std::iter::once;

use ordered_float::OrderedFloat;

use fake_wesnoth;
use naive_ai;
use rust_lua_shared::*;
use fake_wesnoth::{State, Unit, Move};


#[derive (Debug)]
pub struct GenericNode {
  pub state: Arc<State>,
  pub state_globals: Arc<StateGlobals>,
  pub tree: Arc<TreeGlobals>,
  pub visits: i32,
  pub total_score: f64,
  pub naive_score: f64,
  pub choices: Vec<GenericNode>,
  pub node_type: Box<GenericNodeType>,
}

pub trait GenericNodeType: Any + Send + Sync + Debug {
  fn export_moves (&self, node: &GenericNode) -> Vec<fake_wesnoth::Move> {Vec::new()}

  fn initialize_choices (&self, node: &GenericNode) -> (Vec<GenericNode>, Option <Box <GenericNodeType>>);
  
  fn focal_point (&self, node: &GenericNode) -> Option<[i32; 2]> {None}
  fn has_similarity_scores (&self, node: &GenericNode, directory: &SimilarMovesDirectory) -> bool {false}
  fn get_some_similar_moves (&self, node: &GenericNode, directory: &SimilarMovesDirectory) -> Vec<(SimilarMoveIndex, SimilarMoveData)> {Vec::new()}
  fn add_similarity_score (&self, node: &GenericNode, directory: &mut SimilarMovesDirectory, score: f64) {}
  fn make_choice_override (&self, node: &GenericNode) -> Option<(usize, Vec<GenericNode>)> {None}
}

pub trait DisplayableNode {
  fn visits(&self)->i32;
  fn state (&self)->Option<Arc<State>>;
  fn info_text (&self)->String;
  fn detail_text (&self)->String {String::new()}
  fn descendants (&self)->Vec<&DisplayableNode>;
}

impl DisplayableNode for GenericNode {
  fn visits(&self)->i32 {self.visits}
  fn state (&self)->Option<Arc<State>> {Some(self.state.clone())}
  fn info_text (&self)->String {format!("{:.2}\n{}", self.total_score/self.visits as f64, self.visits)}
  fn detail_text (&self)->String {format!("{:?}", self.node_type)}
  fn descendants (&self)->Vec<&DisplayableNode> {
    self.choices.iter().map (| choice | choice as &DisplayableNode).collect()
  }
}




#[derive (Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct SimilarMoveIndex {
  reference_point: OrderedFloat <f64>,
  random_distinguisher: OrderedFloat <f64>
}

#[derive (Clone, Debug)]
pub struct SimilarMoveData {
  total_score: f64,
  visits: f64,
  situation: SimilarMoveSituation,
}

#[derive (Clone, Debug)]
pub struct SimilarMoveSituation {
  state: Arc <State>,
}

fn hex_weight_for_similarity (state: &State, focal_point: Option<[i32; 2]>, location: [i32; 2])->f64 {
  // we want all of the weights to add up to around 1.0
  match focal_point {
    None => {
      1.0/(state.map.width*state.map.height) as f64
    },
    Some (focal_point) => {
      let distance = fake_wesnoth::distance_between (focal_point, location);
      //there are 6N tiles at distance N
      //so we want \sum_0^inf 6N*weight(N) to converge
      //how about weight(N) = 1/6N^3
      // then it converges to 1.645
      // also we need a special case for 0
      if distance == 0 {0.5}
      else { (distance as f64).powi(3) / (2.0*6.0*1.645)}
    },
  }
}
fn similar_move_index (data: & SimilarMoveSituation, focal_point: Option<[i32; 2]>)->SimilarMoveIndex {
  let mut result = 0.0;
  for y in 1..(data.state.map.height+1) { for x in 1..(data.state.map.width+1) {
    let weight = hex_weight_for_similarity (&data.state, focal_point, [x,y]);
    result += weight * hex_similarity_score_1d (data, [x,y]);
  }}
  SimilarMoveIndex {
    reference_point: OrderedFloat(result),
    random_distinguisher: OrderedFloat(rand::thread_rng().gen()),
  }
}
fn similarity_distance (first: & SimilarMoveSituation, second: & SimilarMoveSituation, focal_point: Option<[i32; 2]>)->f64 {
  let mut result = 0.0;
  for y in 1..(first.state.map.height+1) { for x in 1..(first.state.map.width+1) {
    let weight = hex_weight_for_similarity (&first.state, focal_point, [x,y]);
    result += weight * (hex_similarity_score_1d (first, [x,y]) - hex_similarity_score_1d (second, [x,y])).abs();
  }}
  result
}
fn hex_similarity_score_1d (data: & SimilarMoveSituation, location: [i32;2])->f64 {
  let location = data.state.geta (location);
  if let Some(unit) = location.unit.as_ref() {
    unit.hitpoints as f64/unit.unit_type.max_hitpoints as f64
  }
  else {
    0.0
  }
}
fn distance_weight (distance: f64) -> f64 {
  //printlnerr!("distance {:?} became {:?}", distance, 0.5f64.powf(distance.abs()*100.0));
  0.5f64.powf(distance.abs()*100.0)
}

#[derive (Clone, Debug, Default)]
pub struct SimilarMoves {
  data: BTreeMap<SimilarMoveIndex, SimilarMoveData>,
}

fn get_some_similar_moves (similar_moves: Option <& SimilarMoves>, index: SimilarMoveIndex, count: usize)->Vec<(SimilarMoveIndex, SimilarMoveData)> {
  match similar_moves {
    None => Vec::new(),
    Some(similar_moves) => {
      let mut result = Vec::new();
      let mut earlier_iter = similar_moves.data.range ((Unbounded, Excluded(index.clone()))).rev();
      let mut later_iter = similar_moves.data.range ((Excluded(index.clone()), Unbounded));
      let mut earlier_value = earlier_iter.next();
      let mut later_value = later_iter.next();
      loop {
        match (earlier_value, later_value) {
          (None, None) => break,
          (Some(earlier), None) => {
            result.push ((earlier.0.clone(), earlier.1.clone()));
            let len = result.len();
            result.extend (earlier_iter.map(|(k,v)| (k.clone(), v.clone())).take(count - len));
            break
          },
          (None, Some(later)) => {
            result.push ((later.0.clone(), later.1.clone()));
            let len = result.len();
            result.extend (later_iter.map(|(k,v)| (k.clone(), v.clone())).take(count - len));
            break
          },
          (Some(earlier), Some(later)) => {
            if index.reference_point.0 - earlier.0.reference_point.0 < later.0.reference_point.0 - index.reference_point.0 {
              result.push ((earlier.0.clone(), earlier.1.clone()));
              earlier_value = earlier_iter.next();
            }
            else {
              result.push ((later.0.clone(), later.1.clone()));
              later_value = later_iter.next();
            }
          }
        }
        if result.len() >= count { break }
      }
      result
    }
  }
}

fn add_similarity_score (similar_moves: &mut SimilarMoves, index: SimilarMoveIndex, situation: SimilarMoveSituation, score: f64) {
  let data = similar_moves.data.entry(index).or_insert(SimilarMoveData {
    total_score:0.0,
    visits:0.0,
    situation,
  });
  data.total_score += score;
  data.visits += 1.0;
}

#[derive (Clone, Debug, Default)]
pub struct SimilarMovesDirectory {
  pub attacks: HashMap<Attack, SimilarMoves>,
  pub moves: HashMap<([i32;2], [i32;2]), SimilarMoves>,
  pub recruit_types: HashMap <usize, SimilarMoves>,
  pub recruit_locations: HashMap <[i32; 2], SimilarMoves>,
  pub finish_turns: HashMap <(i32,usize), SimilarMoves>,
}


#[derive (Debug)]
pub struct TreeGlobals {
  similar_moves: Mutex <SimilarMovesDirectory>,
  pub starting_turn: i32,
  pub starting_side: usize,
}


#[derive (Clone, Debug, Default)]
pub struct StateGlobals {
  similarity_index: SimilarMoveIndex,
  pub reaches: Arc<HashMap<[i32;2], fake_wesnoth::Reach>>
}



#[derive (Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct Attack {
  unit_id: usize, dst_x: i32, dst_y: i32, attack_x: i32, attack_y: i32, weapon: usize,
}
#[derive (Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct Recruit {
  dst_x: i32, dst_y: i32, unit_type: usize,
}
/*struct Move {
  src_x: i32, src_y: i32, dst_x: i32, dst_y: i32,
}*/
#[derive (Clone)]
pub struct SpaceClearingMoves {
  planned_moves: Vec<([i32; 2], [i32; 2])>,
  state_globals: Arc<StateGlobals>,
}

impl SpaceClearingMoves {
  fn new(state_globals: Arc<StateGlobals>)->SpaceClearingMoves {
    SpaceClearingMoves {
      planned_moves: Vec::new(),
      state_globals
    }
  }
  fn to_wesnoth_moves (&self)->Vec<fake_wesnoth::Move> {
    self.planned_moves.iter().map (| &locations | to_wesnoth_move (&self.state_globals, locations)).collect()
  }
}
impl Debug for SpaceClearingMoves {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "SpaceClearingMoves {:?}", self.planned_moves)
  }
}

fn to_wesnoth_move (state_globals: &StateGlobals, locations: ([i32; 2], [i32; 2]))->fake_wesnoth::Move {
  fake_wesnoth::Move::Move {
    src_x: locations.0[0], src_y: locations.0[1],
    dst_x: locations.1[0], dst_y: locations.1[1],
    moves_left: state_globals.reaches[&locations.0].get (locations.1).unwrap()
  }
}

pub fn generate_reaches(state: & State)->HashMap<[i32;2], fake_wesnoth::Reach> {
  state.locations.iter()
        .filter_map (| location | location.unit.as_ref())
        .filter (| unit | unit.side == state.current_side)
        .map (| unit |
          (
            [unit.x, unit.y],
            fake_wesnoth::find_reach (state, unit)
          )
        ).collect()
}

pub fn update_reaches_after_move (reaches: &mut HashMap<[i32;2], fake_wesnoth::Reach>, state: & State, action: & fake_wesnoth::Move) {
  match action {
    &fake_wesnoth::Move::Move {src_x, src_y, dst_x, dst_y, moves_left} => {
      reaches.remove (&[src_x, src_y]);
      reaches.insert ([dst_x, dst_y], fake_wesnoth::find_reach (state, state.get(dst_x, dst_y).unit.as_ref().unwrap()));
    },
    &fake_wesnoth::Move::Attack {src_x, src_y, dst_x, dst_y, attack_x, attack_y, weapon} => {
      if state.get (attack_x, attack_y).unit.is_none() {
        ::std::mem::replace(reaches, generate_reaches(state));
      }
      else {
        reaches.remove (&[src_x, src_y]);
        if let Some(unit) = state.get(dst_x, dst_y).unit.as_ref() {
          reaches.insert ([dst_x, dst_y], fake_wesnoth::find_reach (state, unit));
        }
      }
    },
    &fake_wesnoth::Move::Recruit {dst_x, dst_y, ref unit_type} => {
      reaches.insert ([dst_x, dst_y], fake_wesnoth::find_reach (state, state.get(dst_x, dst_y).unit.as_ref().unwrap()));
    },
    &fake_wesnoth::Move::EndTurn => {
      ::std::mem::replace(reaches, generate_reaches(state));
    },
  };
}



#[derive (Serialize, Deserialize, Debug, Default)]
struct ChooseAttack;
#[derive (Debug)]
struct ExecuteAttack {
  attack: Attack,
  preparation: SpaceClearingMoves,
}

#[derive (Debug)]
struct ExecuteRecruit {
  recruit: Recruit,
  preparation: SpaceClearingMoves,
}
#[derive (Serialize, Deserialize, Debug, Default)]
struct FinishTurnLazily(Vec<fake_wesnoth::Move>);
#[derive (Clone)]
pub struct ChooseHowToClearSpace {
  plan: SpaceClearingMoves,
  desired_moves: Vec<([i32; 2], [i32; 2])>,
  desired_empty: Vec<[i32; 2]>,
  follow_up: Arc<Fn(SpaceClearingMoves)->Box<GenericNodeType>+Send+Sync>,
  steps: usize,
}
impl ChooseHowToClearSpace {
  fn finalize(&self)->SpaceClearingMoves {
    let mut result = self.plan.clone();
    result.planned_moves.extend (self.desired_moves.iter().filter (| &locations | locations.1 != locations.0));
    result
  }
}
impl Debug for ChooseHowToClearSpace {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "ChooseHowToClearSpace{:?}", (&self.plan, &self.desired_moves, &self.desired_empty, &self.steps))
  }
}

impl GenericNodeType for ChooseAttack {
  fn initialize_choices (&self, node: &GenericNode) -> (Vec<GenericNode>, Option <Box <GenericNodeType>>) {
    let mut new_children = Vec::new();
    for unit in node.state.locations.iter()
        .filter_map (| location | location.unit.as_ref())
        .filter(|unit| unit.side == node.state.current_side && unit.attacks_left > 0) {
      for location in node.state_globals.reaches [& [unit.x, unit.y]].list.iter() {
        if node.state.get (location.0 [0], location.0 [1]).unit.as_ref().map_or (false, | other | other.side != node.state.current_side) { continue; }
        
        if unit.canrecruit {
          for recruit_location in recruit_hexes (&node.state, location.0) {
            let unit_on_recruit_location = node.state.geta (recruit_location).unit.as_ref();
            if unit_on_recruit_location.map_or (false, | other | other.side != node.state.current_side) { continue; }
            
            for &recruit in node.state.sides [unit.side].recruits.iter() {
              if node.state.sides [unit.side].gold < node.state.map.config.unit_type_examples.get (recruit).unwrap().unit_type.cost { continue; }
              let action = Recruit {
                dst_x: recruit_location [0], dst_y: recruit_location [1],
                unit_type: recruit.clone(),
              };
              new_children.push(
                if location.0 == [unit.x,unit.y] && unit_on_recruit_location.is_none() {
                  node.new_child (ExecuteRecruit{recruit: action, preparation: SpaceClearingMoves::new(node.state_globals.clone())})
                }
                else {
                  node.new_child (ChooseHowToClearSpace{
                    desired_moves: vec![([unit.x,unit.y],location.0)],
                    desired_empty: vec![recruit_location],
                    plan: SpaceClearingMoves::new(node.state_globals.clone()),
                    follow_up: Arc::new(move |p| Box::new(ExecuteRecruit{recruit: action.clone(), preparation:p})),
                    steps: 0,
                  })
                }
              );
            }
          }
        }
        
        for adjacent in fake_wesnoth::adjacent_locations (& node.state.map, location.0) {
          if let Some (neighbor) = node.state.get (adjacent [0], adjacent [1]).unit.as_ref() {
            if !node.state.is_enemy (unit.side, neighbor.side) { continue; }
            for index in 0..unit.unit_type.attacks.len() {
              let attack = Attack {
                unit_id: unit.id,
                dst_x: location.0 [0], dst_y: location.0 [1],
                attack_x: adjacent [0], attack_y: adjacent [1],
                weapon: index,
              };
              let mut new_child =
                if location.0 == [unit.x,unit.y] {
                  node.new_child (ExecuteAttack{attack: attack, preparation: SpaceClearingMoves::new(node.state_globals.clone())})
                }
                else {
                  node.new_child (ChooseHowToClearSpace{
                    desired_moves: vec![([unit.x,unit.y],location.0)],
                    desired_empty: Vec::new(),
                    plan: SpaceClearingMoves::new(node.state_globals.clone()),
                    follow_up: Arc::new(move |p| Box::new(ExecuteAttack{attack: attack.clone(), preparation:p})),
                    steps: 0,
                  })
                };
              new_child.naive_score = ::naive_ai::evaluate_move (&node.state, &fake_wesnoth::Move::Attack {src_x: unit.x, src_y: unit.y, dst_x: location.0 [0], dst_y: location.0 [1], attack_x: adjacent [0], attack_y: adjacent [1], weapon: index, }, naive_ai::EvaluateMoveParameters { .. Default::default() });
              new_children.push(new_child);
            }
          }
        }
      }
    }
    new_children.push(node.new_child(FinishTurnLazily(Vec::new())));
    (new_children, None)
  }
}
impl GenericNodeType for ExecuteAttack {
  fn initialize_choices (&self, node: &GenericNode) -> (Vec<GenericNode>, Option <Box <GenericNodeType>>) {
    (Vec::new(), None)
  }
  fn export_moves (&self, node: &GenericNode) -> Vec<fake_wesnoth::Move> {
    let mut result = Vec::new();
    result.extend (self.preparation.to_wesnoth_moves());
    let Attack { unit_id, dst_x, dst_y, attack_x, attack_y, weapon, } = self.attack;
    result.push (fake_wesnoth::Move::Attack {
      src_x: dst_x, src_y: dst_y, dst_x, dst_y, attack_x, attack_y, weapon,
    });
    result
  }
  fn make_choice_override (&self, node: &GenericNode) -> Option<(usize, Vec<GenericNode>)> {
    let Attack {
                unit_id, dst_x, dst_y, attack_x, attack_y, weapon,
              } = self.attack;
    if node.choices.is_empty() || node.visits > (1<<node.choices.len()) {
      let mut new_child = node.new_child_unscored(ChooseAttack);
      new_child.do_moves_on_state(once (fake_wesnoth::Move::Attack {
        src_x: dst_x, src_y: dst_y, dst_x, dst_y, attack_x, attack_y, weapon,
      }));
      Some((node.choices.len(), vec![new_child]))
    }
    else {
      Some((rand::thread_rng().gen_range(0, node.choices.len()), Vec::new()))
    }
  }
  
  fn has_similarity_scores (&self, node: &GenericNode, directory: &SimilarMovesDirectory) -> bool {
    directory.attacks.get(& self.attack).is_some()
  }
  fn get_some_similar_moves (&self, node: &GenericNode, directory: &SimilarMovesDirectory) -> Vec<(SimilarMoveIndex, SimilarMoveData)> {
    get_some_similar_moves (directory.attacks.get(& self.attack), node.state_globals.similarity_index.clone(), 30)
  }
  fn add_similarity_score (&self, node: &GenericNode, directory: &mut SimilarMovesDirectory, score: f64) {
    add_similarity_score (directory.attacks.entry (self.attack.clone()).or_insert (Default::default()), node.state_globals.similarity_index.clone(), SimilarMoveSituation {state: node.state.clone()}, score);
  }
}
impl GenericNodeType for ExecuteRecruit {
  fn initialize_choices (&self, node: &GenericNode) -> (Vec<GenericNode>, Option <Box <GenericNodeType>>) {
    let mut new_child = node.new_child_unscored(ChooseAttack);
    let Recruit{dst_x,dst_y,unit_type} = self.recruit.clone();
    new_child.do_moves_on_state(once (fake_wesnoth::Move::Recruit{dst_x,dst_y,unit_type}));
    (vec![new_child], None)
  }
  fn export_moves (&self, node: &GenericNode) -> Vec<fake_wesnoth::Move> {
    let mut result = Vec::new();
    result.extend (self.preparation.to_wesnoth_moves());
    let Recruit{dst_x,dst_y,unit_type} = self.recruit.clone();
    result.push (fake_wesnoth::Move::Recruit{dst_x,dst_y,unit_type});
    result
  }
  fn has_similarity_scores (&self, node: &GenericNode, directory: &SimilarMovesDirectory) -> bool {
    directory.recruit_types.get(& self.recruit.unit_type).is_some() || directory.recruit_locations.get(& [self.recruit.dst_x, self.recruit.dst_y]).is_some()
  }
  fn get_some_similar_moves (&self, node: &GenericNode, directory: &SimilarMovesDirectory) -> Vec<(SimilarMoveIndex, SimilarMoveData)> {
    let mut result = Vec::new();
    result.extend(get_some_similar_moves (directory.recruit_types.get(& self.recruit.unit_type), node.state_globals.similarity_index.clone(), 15).into_iter());
    result.extend(get_some_similar_moves (directory.recruit_locations.get(& [self.recruit.dst_x, self.recruit.dst_y]), node.state_globals.similarity_index.clone(), 15).into_iter());
    result
  }
  fn add_similarity_score (&self, node: &GenericNode, directory: &mut SimilarMovesDirectory, score: f64) {
    add_similarity_score (directory.recruit_types.entry (self.recruit.unit_type).or_insert (Default::default()), node.state_globals.similarity_index.clone(), SimilarMoveSituation {state: node.state.clone()}, score);
    add_similarity_score (directory.recruit_locations.entry ([self.recruit.dst_x, self.recruit.dst_y]).or_insert (Default::default()), node.state_globals.similarity_index.clone(), SimilarMoveSituation {state: node.state.clone()}, score);
  }
}
impl GenericNodeType for FinishTurnLazily {
  fn initialize_choices (&self, node: &GenericNode) -> (Vec<GenericNode>, Option <Box <GenericNodeType>>) {
    let mut moves = Vec::new();
    let mut new_child = node.new_child_unscored(ChooseAttack);
    
    let mut state_after = (*node.state).clone();
    /*let mut reaches = node.state_globals.reaches.clone();
    loop {
            let action = state_after.locations.iter()
              .filter_map (| location | location.unit.as_ref())
              .filter(|unit| unit.side == state_after.current_side)
              .flat_map (| unit | {
                let coords = [unit.x, unit.y];
                let state_after = &state_after;
                reaches[&coords].list.iter().filter_map(move | &(destination, moves_left) | {
                  if state_after.geta(destination).unit.is_none() {
                    Some(fake_wesnoth::Move::Move {
                      src_x: coords[0], src_y: coords[1],
                      dst_x: destination [0], dst_y: destination [1],
                      moves_left: moves_left
                    })
                  }
                  else {
                    None
                  }
                })
              })
              .chain(::std::iter::once (fake_wesnoth::Move::EndTurn))
              .map(|action| {
                (action.clone(), ::naive_ai::evaluate_move (& state_after, &action, false))
              })
              .max_by (|a,b| a.1.partial_cmp(&b.1).unwrap())
              .unwrap().0;
            
            fake_wesnoth::apply_move (&mut state_after, &mut Vec::new(), & action);
            moves.push (action.clone());
            
            match action {
              fake_wesnoth::Move::EndTurn => break,
              _=>(),
            }
            update_reaches_after_move (&mut reaches, & state_after, & action);
    }*/
    moves = ::naive_ai::play_turn_fast(&mut state_after, naive_ai::PlayTurnFastParameters{
      allow_combat: false,
      .. Default::default()
    });
    new_child.set_state(state_after);
    (vec![new_child], Some(Box::new(FinishTurnLazily(moves))))
  }
  fn export_moves (&self, _node: &GenericNode) -> Vec<fake_wesnoth::Move> {
    self.0.clone()
  }
  fn has_similarity_scores (&self, node: &GenericNode, directory: &SimilarMovesDirectory) -> bool {
    directory.finish_turns.get(& (node.state.turn, node.state.current_side)).is_some()
  }
  fn get_some_similar_moves (&self, node: &GenericNode, directory: &SimilarMovesDirectory) -> Vec<(SimilarMoveIndex, SimilarMoveData)> {
    get_some_similar_moves (directory.finish_turns.get(& (node.state.turn, node.state.current_side)), node.state_globals.similarity_index.clone(), 30)
  }
  fn add_similarity_score (&self, node: &GenericNode, directory: &mut SimilarMovesDirectory, score: f64) {
    add_similarity_score (directory.finish_turns.entry ((node.state.turn, node.state.current_side)).or_insert (Default::default()), node.state_globals.similarity_index.clone(), SimilarMoveSituation {state: node.state.clone()}, score);
  }
}
impl GenericNodeType for ChooseHowToClearSpace {
  fn initialize_choices (&self, node: &GenericNode) -> (Vec<GenericNode>, Option <Box <GenericNodeType>>) {
    let mut result = Vec::new();
    let mut movers = HashMap::new();
    let state_hack = node.state.clone();
    let state_globals = node.state_globals.clone();
    let get = | movers: &HashMap<[i32;2], Option<[i32;2]>>, location |->Option<[i32;2]> {
      movers.get (&location).cloned().unwrap_or_else (|| {
        state_hack.geta (location).unit.as_ref().map (| unit | [unit.x, unit.y])
      })
    };
    // Currently, we don't allow moving the attacker out of the way so that something else can move to its original location. We should support this eventually. But it's simpler to implement without it being allowed.
    let forbidden: HashSet<[i32;2]> = self.desired_empty.iter().cloned().chain(self.desired_moves.iter().map (| locations | locations.1)).chain(self.desired_moves.iter().map (| locations | locations.0)).collect();
    // Really, at some point it gets too complicated, and we want a hard limit to make sure the AI doesn't stack-overflow or whatever
    if self.steps > 8 { return (result, None); }
    for (index, planned_move) in self.plan.planned_moves.iter().enumerate() {
      let destination = get (&movers, planned_move.1);
      //printlnerr!("uh {:?} sddd {:?}", planned_move, destination);
      if destination.is_none() {
        let source = get (&movers, planned_move.0).unwrap();
        movers.insert (planned_move.0, None);
        movers.insert (planned_move.1, Some(source));
      }
      else {
        for adjacent in fake_wesnoth::adjacent_locations (& node.state.map, planned_move.1) {
          if node.state.geta (adjacent).unit.as_ref().map_or (false, | other | other.side != node.state.current_side) { continue; }
          if forbidden.contains(&adjacent) { continue; }
          
          // we may try moving the blocking unit (at destination) out of the way first
          let mut new_info = self.clone();
          new_info.steps += 1;
          let previous = new_info.plan.planned_moves.iter().position (| k | k.1 == planned_move.1);
          // TODO: should we exclude changes that reduce the amount a unit is moving?
          let (blocker_original_location, insert_index) = match previous {
            None => (planned_move.1, index),
            Some (previous) => (new_info.plan.planned_moves.remove (previous).0, min(index, previous)),
          };
          if let Some (moves_left) = state_globals.reaches [&blocker_original_location].get (adjacent) {
            // all changes must increase the amount of movement, to avoid infinite loops
            let previous_moves_left = state_globals.reaches [&blocker_original_location].get (planned_move.1).unwrap();
            if moves_left < previous_moves_left {
              new_info.plan.planned_moves.insert (insert_index, (blocker_original_location, adjacent));
              result.push (node.new_child (new_info));
            }
          }
          
          // we may also try continuing the movement of the blocked unit
          if let Some (moves_left) = state_globals.reaches [&planned_move.0].get(adjacent) {
            let mut new_info = self.clone();
            new_info.steps += 1;
            // all changes must increase the amount of movement, to avoid infinite loops
            let previous_moves_left = state_globals.reaches [&planned_move.0].get (planned_move.1).unwrap();
            if moves_left < previous_moves_left {
              new_info.plan.planned_moves[index].1 = adjacent;
              result.push (node.new_child (new_info));
            }
          }
        }
        return (result, None);
      }
    }
    for desired_empty in self.desired_empty.iter().cloned().chain(self.desired_moves.iter().filter_map (| locations | if locations.1 != locations.0 { Some(locations.1) } else { None })) {
      let destination = get (&movers, desired_empty);
      if destination.is_some() && !self.desired_moves.iter().any(|loc| loc.0 == desired_empty) {
        for adjacent in fake_wesnoth::adjacent_locations (& node.state.map, desired_empty) {
          if node.state.geta (adjacent).unit.as_ref().map_or (false, | other | other.side != node.state.current_side) { continue; }
          if forbidden.contains(&adjacent) { continue; }
          
          let mut new_info = self.clone();
          new_info.steps += 1;
          if let Some (_moves_left) = state_globals.reaches [&desired_empty].get(adjacent) {
            new_info.plan.planned_moves.insert (0, (desired_empty, adjacent));
            result.push (node.new_child (new_info));
          }
        }
        return (result, None);
      }
    }
    
    let final_moves = self.finalize();
    let wesnoth_moves = final_moves.to_wesnoth_moves();
    let mut new_child = node.new_child_dynamic ((*self.follow_up)(final_moves));
    new_child.do_moves_on_state(wesnoth_moves.into_iter());
    result.push (new_child);
    (result, None)
  }  
  fn has_similarity_scores (&self, node: &GenericNode, directory: &SimilarMovesDirectory) -> bool {
    (*self.follow_up)(self.finalize()).has_similarity_scores(node, directory)
  }
  fn get_some_similar_moves (&self, node: &GenericNode, directory: &SimilarMovesDirectory) -> Vec<(SimilarMoveIndex, SimilarMoveData)> {
    (*self.follow_up)(self.finalize()).get_some_similar_moves(node, directory)
  }
  fn add_similarity_score (&self, node: &GenericNode, directory: &mut SimilarMovesDirectory, score: f64) {
    (*self.follow_up)(self.finalize()).add_similarity_score(node, directory, score)
  }
}


enum StepIntoResult {
  PlayedOutWithScores(Vec<f64>),
  TurnedOutToBeImpossible,
}
use self::StepIntoResult::{PlayedOutWithScores, TurnedOutToBeImpossible};

pub fn choose_moves (state: & State)->(GenericNode, Vec<Move>) {
  let mut root = GenericNode {
    state: Arc::new (state.clone()),
    tree: Arc::new(TreeGlobals {
      starting_turn: state.turn,
      starting_side: state.current_side,
      similar_moves: Default::default(),
    }),
    state_globals: Arc::new(StateGlobals {
      reaches: Arc::new(generate_reaches(state)),
      similarity_index: Default::default(),
    }),
    visits: 0,
    total_score: 0.0,
    naive_score: 0.0,
    choices: Vec::new(),
    node_type: Box::new(ChooseAttack), 
  };

  for _ in 0..7000 {
    root.step_into ();
  }
  
  let mut result = Vec::new();
  {
    let mut node = &root;
    loop {
      result.extend (node.node_type.export_moves(&node));
      match result.last() {
        Some(& fake_wesnoth::Move::Attack{..})
        | Some(& fake_wesnoth::Move::Recruit{..})
        | Some(& fake_wesnoth::Move::EndTurn) => break,
        _ => (),
      };
      node = node.choices.iter()
        .max_by_key (|a| a.visits)
        .unwrap();
    }
  }
    
  (root, result)
}

impl GenericNode {
  fn new_child<T: GenericNodeType> (&self, node_type: T) -> GenericNode {
    let mut result = self.new_child_unscored(node_type);
    result.update_similarity_score();
    result
  }
  fn new_child_unscored<T: GenericNodeType> (&self, node_type: T) -> GenericNode {
    self.new_child_dynamic(Box::new (node_type))
  }

  fn new_child_dynamic (&self, node_type: Box<GenericNodeType>) -> GenericNode {
    GenericNode {
      state: self.state.clone(), tree: self.tree.clone(), state_globals: self.state_globals.clone(),
      visits: 0, total_score: 0.0,
      naive_score: 0.0,
      choices: Vec::new(),
      node_type: node_type,
    }
  }

  fn set_state (&mut self, new_state: State) {
    self.state = Arc::new(new_state);
    self.state_globals = Arc::new(StateGlobals {
      reaches: Arc::new(generate_reaches(&self.state)),
      similarity_index: similar_move_index(&SimilarMoveSituation{state:self.state.clone()}, self.node_type.focal_point(self))
    });
  }
  fn do_moves_on_state <I: Iterator<Item=fake_wesnoth::Move>>(&mut self, actions: I) {
    let mut state_after = (*self.state).clone();
    let mut reaches = (*self.state_globals.reaches).clone();
    for action in actions {
      fake_wesnoth::apply_move (&mut state_after, &mut Vec::new(), & action);
      update_reaches_after_move (&mut reaches, &state_after, & action);
    }
    self.state = Arc::new(state_after);
    self.state_globals = Arc::new(StateGlobals {
      reaches: Arc::new(reaches),
      similarity_index: similar_move_index(&SimilarMoveSituation{state:self.state.clone()}, self.node_type.focal_point(self))
    });
  }
  fn update_similarity_score (&mut self) {
    self.state_globals = Arc::new(StateGlobals {
      reaches: self.state_globals.reaches.clone(),
      similarity_index: similar_move_index(&SimilarMoveSituation{state:self.state.clone()}, self.node_type.focal_point(self))
    });
  }

  
  fn step_into (&mut self)->StepIntoResult {
    /*
    
    If NONE of my children have similarity scores yet:
    – do a naive play out, recording scores for a couple turns' worth of moves from the play out.
    If ALL of my children have similarity scores:
    – make an MCTS-like choice, combining weighted winrate with uncertainty
    If SOME of my children have similarity scores, but some don't:
    – we want to explore them all eventually, if this node is visited enough times (as some of them may be unique to this node).
      However, doing it immediately would likely bury newly discovered nodes by having them try out many bad novel moves first, since wesnoth's branching factor is too high.
      So we alternate exploration and exploitation.
    
    */
  
    //printlnerr!("{:?}", self.node_type);
    let scores = if let Some(scores) = self.state.scores.clone() {
      scores
    }
    /*else if self.state.current_side == self.tree.starting_side && self.state.turn == self.tree.starting_turn + 3 {
      ::naive_ai::evaluate_state(&self.state)
    }*/
    else if self.visits == 0 && ((self.state.turn, self.state.current_side) >= (self.tree.starting_turn+2, self.tree.starting_side)) {
      let mut playout_state = (*self.state).clone();
      while playout_state.scores.is_none() && playout_state.turn < self.tree.starting_turn + 5 {
        let turn = playout_state.turn;
        ::naive_ai::play_turn_fast (&mut playout_state, naive_ai::PlayTurnFastParameters{
          allow_combat: true,
          stop_at_combat: false,
          exploit_kills: false,
          pursue_villages: turn < self.tree.starting_turn + 4,
          evaluate_move_parameters: naive_ai::EvaluateMoveParameters {
            accurate_combat: false,
            aggression: 2.0,
            .. Default::default()
          },
          .. Default::default()
        });
      }
      //printlnerr!(" Playout ran until {:?}", playout_state.turn);
      playout_state.scores.clone().unwrap_or_else (|| ::naive_ai::evaluate_state(&playout_state))
    }
    else {
      if self.choices.is_empty() {
        let (a,b) = self.node_type.initialize_choices (&self);
        self.choices = a;
        if let Some(b) = b {self.node_type = b;}
      }
      
      let choice = match self.node_type.make_choice_override (self) {
        Some((index, additions)) => {
          self.choices.extend (additions);
          index
        },
        None => {
          
          
          match self.choices.len() {
            0 => return TurnedOutToBeImpossible,
            1 => 0,
            _ => {
              let guard = self.tree.similar_moves.lock().unwrap();
              let similar_moves = &*guard;
              
              let mut scored_moves = 0;
              for choice in self.choices.iter() {
                if choice.node_type.has_similarity_scores (choice, similar_moves) {
                  scored_moves += 1;
                }
              }
              
              /*
                instead of just exploration versus exploitation, we actually have THREE choices here:
                1) Choose the move with the best naive score, whether or not it has a similarity score
                2) Choose a move with no similarity score yet (presumably, the one with the highest naive score)
                3) Choose a move with the best similarity score (possibly with an uncertainty bonus)
                
                intuitive goals:
                – with very few visits, we should usually be doing 1.
                – With very many visits, we should usually be doing 3.
                – Somewhere in between those, we need to be doing 2 a bunch.
                – Unlike straight MCTS, we want the "playouts" to leverage the accumulated guesses, so we should sometimes do 3 before we're done doing 2.
                
                notes:
                – if you've done 1 at least once, 3 is probably better because 3 already has data on whether the move from 1 worked. But it's possible to have similarity scores only for bad moves when the naive-best move is actually the best move. So it makes sense to force 1 at least once.
                – If the naive-best move HASN'T been explored yet, 2 is equivalent to 1.
                – The FIRST choice may be a conflict between 1 and 3. In the case where a few choices have similarity scores, 1 seems better. If most of the choices have similarity scores, it's more complex – do we go for the good-looking novel move or the established leader?
                
                Since I haven't come up with a coherent argument for how to prioritize, I'm just going to alternate.
              */
              
              let (index, naive_best) = self.choices.iter().enumerate().max_by_key(|&(a,b)|OrderedFloat(b.naive_score)).unwrap();
              
              if !naive_best.node_type.has_similarity_scores (naive_best, similar_moves) {
                index
              }
              else {
                if ((self.visits & 1) == 0) && self.choices.iter().any(|a| !a.node_type.has_similarity_scores (a, similar_moves)) {
                  self.choices.iter().enumerate().filter(|&(a,b)| !b.node_type.has_similarity_scores (b, similar_moves)).max_by_key(|&(a,b)|OrderedFloat(b.naive_score)).unwrap().0
                }
                else {
                  let c=0.2;
                  let c_log_visits = c*((self.visits+1) as f64).ln();
                  self.choices.iter().enumerate().filter_map(|(index,choice)| {
                    let exact_score = choice.total_score;
                    let exact_weight = choice.visits as f64 * distance_weight(0.0);
                    let mut total_weight = exact_weight;
                    let mut total_score = exact_score;
                    for (similar_index, similar) in choice.node_type.get_some_similar_moves (choice, similar_moves).into_iter() {
                      let weight = distance_weight (
                        (similar_index.reference_point.0 - choice.state_globals.similarity_index.reference_point.0).abs()
                        //similarity_distance(&similar.situation, &SimilarMoveSituation{state:self.state.clone()}, self.node_type.focal_point(&self))
                        
                      );
                      total_score += similar.total_score * weight;
                      total_weight += similar.visits * weight;
                    };
                    
                    // I choose to limit the certainty granted by non-exact moves, so that it can't indefinitely postpone getting more exact scores.
                    let uncertainty_bonus = (c_log_visits/
                      min(OrderedFloat(total_weight), OrderedFloat(exact_weight*2.0+10.0)).0
                    ).sqrt();
                    let score = total_score / total_weight + uncertainty_bonus;
                  
                    Some((index, score))
                  }).max_by_key(|&(a,b)|OrderedFloat(b)).unwrap().0
                }
              }
            },
          }
        },
      };
      
      let scores = match self.choices [choice].step_into () {
        PlayedOutWithScores(scores) => scores,
        TurnedOutToBeImpossible => {
          self.choices.remove(choice);
          if self.choices.is_empty() { return TurnedOutToBeImpossible; }
          return self.step_into();
        },
      };
      
      scores
    };
    
    self.total_score += scores[self.state.current_side];
    self.visits += 1;
    
    {
      let mut guard = self.tree.similar_moves.lock().unwrap();
      let similar_moves = &mut*guard;
      self.node_type.add_similarity_score(self, similar_moves, scores[self.state.current_side]);
    }
    
    PlayedOutWithScores(scores)
  }

}
