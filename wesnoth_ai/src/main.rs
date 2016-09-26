#![feature (plugin, custom_derive)]
#![plugin (serde_macros)]

extern crate serde;
extern crate serde_json;
extern crate rand;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use rand::{Rng, random};

/// One individual "organism" shareable with Lua.
/// Represents a function from game state to move evaluations.
#[derive (Clone, Serialize, Deserialize)]
struct Organism {
  signature: String,
  layer_sizes: Vec<usize>,
  weights_by_input: HashMap<String, Vec<LayerWeights>>,
  output_weights: Matrix,
}

#[derive (Clone, Serialize, Deserialize)]
struct LayerWeights {
  hidden_matrix: Matrix,
  input_matrix: Matrix,
}
#[derive (Clone, Serialize, Deserialize)]
struct Matrix {
  input_size: usize,
  output_size: usize,
  weights: Vec<f32>,
}

const UNIT_SIZE: usize = 12;

thread_local! {
  static INPUTS: HashMap <String, usize> = {
    let mut result = HashMap::new();
    result.insert ("add_unit".to_string(), UNIT_SIZE);
    result
  }
}

fn random_matrix (input_size: usize, output_size: usize)->Matrix {
  let mut result = Matrix {
    input_size: input_size,
    output_size: output_size,
    weights: Vec::with_capacity (input_size*output_size),
  };
  for _ in 0..input_size*output_size {
    result.weights.push (random::<f32>() * 2.0 - 1.0);
  }
  result
}

fn random_organism (layer_sizes: Vec<usize>)->Organism {
  let last_layer_size = layer_sizes.last().unwrap().clone();
  let mut result = Organism {
    signature: rand::thread_rng().gen_ascii_chars().take (20).collect(),
    layer_sizes: layer_sizes.clone(),
    weights_by_input: HashMap::new(),
    output_weights: random_matrix (last_layer_size, 1),
  };
  INPUTS.with (| inputs| for (name, size) in inputs {
    let mut vect = vec![LayerWeights{
      input_matrix: random_matrix (size.clone(), layer_sizes[0]),
      hidden_matrix: random_matrix (layer_sizes[0], layer_sizes[0]),}];
    for index in 1.. layer_sizes.len() {
      vect.push (LayerWeights{
        input_matrix: random_matrix (layer_sizes[index - 1], layer_sizes[index]),
        hidden_matrix: random_matrix (layer_sizes[index], layer_sizes[index]),});
    }
    result.weights_by_input.insert (name.clone(), vect);
  });
  result
}

#[derive (Clone, Serialize, Deserialize)]
struct Memory {
  layers: Vec<Vec<f32>>,
}
#[derive (Clone, Serialize, Deserialize)]
struct NeuralInput {
  input_type: String,
  vector: Vec<f32>
}

fn multiply_into (input: &[f32], output: &mut [f32], matrix: & Matrix) {
  assert_eq!(input.len(), matrix.input_size);
  assert_eq!(output.len(), matrix.output_size);
  for input_index in 0..matrix.input_size {
    for output_index in 0..matrix.output_size {
      output [output_index] += input [input_index]*matrix.weights [output_index + input_index*matrix.output_size]
    }
  }
}


fn next_memory (organism: & Organism, memory: & Memory, input: &NeuralInput)->Memory {
  let mut result = Memory {layers: Vec::new()};
  for layer in 0..organism.layer_sizes.len() {
    let mut next_layer = vec![0.0; organism.layer_sizes [layer]];
    let layer_weights = &organism.weights_by_input.get (& input.input_type).unwrap()[layer];
    multiply_into (&memory.layers [layer], &mut next_layer, & layer_weights.hidden_matrix);
    multiply_into (if layer == 0 {&input.vector} else {& memory.layers [layer - 1]}, &mut next_layer, & layer_weights.input_matrix);
    for item in next_layer.iter_mut() {*item = item.tanh();}
    result.layers.push (next_layer);
  }
  result
}

fn initial_memory (organism: & Organism)->Memory {
  let mut result = Memory {layers: Vec::new()};
  for layer in 0..organism.layer_sizes.len() {
    let next_layer = vec![0.0; organism.layer_sizes [layer]];
    result.layers.push (next_layer);
  }
  result 
}

fn evaluate_move (organism: & Organism, memory: & Memory, input: & NeuralInput)->f32 {
  if input.input_type == "end_turn" {return 0.0;}
  let mut output = vec![0.0];
  multiply_into (&next_memory (organism, memory, input).layers.last().unwrap(), &mut output, & organism.output_weights);
  output [0]
}


#[derive (Clone, Serialize, Deserialize)]
struct Attack {
  damage: i32,
  number: i32,
  damage_type: String,
  range: String,
  // TODO: specials
}
#[derive (Clone, Serialize, Deserialize)]
struct Side {
  gold: i32,
  enemies: HashSet <usize>,
}


#[derive (Clone, Serialize, Deserialize)]
struct Unit {
  x: i32,
  y: i32,
  side: usize,
  alignment: i32,
  attacks_left: i32,
  canrecruit: bool,
  cost: i32,
  experience: i32,
  hitpoints: i32,
  level: i32,
  max_experience: i32,
  max_hitpoints: i32,
  max_moves: i32,
  moves: i32,
  resting: bool,
  slowed: bool,
  poisoned: bool,
  not_living: bool,
  zone_of_control: bool,
  defense: HashMap<String, i32>,
  movement_costs: HashMap <String, i32>,
  resistance: HashMap <String, i32>,
  attacks: Vec<Attack>,
  // TODO: abilities
}

#[derive (Clone, Serialize, Deserialize)]
struct Location {
  terrain: String,
  village_owner: usize,
  unit: Option <Box <Unit>>,
}
#[derive (Clone, Serialize, Deserialize)]
struct TerrainInfo{
  keep: bool,
  castle: bool,
  village: bool,
  healing: i32,
}

#[derive (Clone, Serialize, Deserialize)]
struct Faction {
  recruits: Vec<Unit>,
  leaders: Vec<Unit>,
}

#[derive (Clone, Serialize, Deserialize)]
struct WesnothConfig {
  unit_type_examples: HashMap <String, Unit>,
  terrain_info: HashMap <String, TerrainInfo>,
  factions: Vec<Faction>,
}

#[derive (Clone, Serialize, Deserialize)]
struct WesnothMap {
  config: Arc <WesnothConfig>,
  width: i32,
  height: i32,
  locations: Vec<Location>,
  starting_locations: Vec<[i32; 2]>,
}

#[derive (Clone, Serialize, Deserialize)]
struct WesnothState {
  map: Arc <WesnothMap>,
  current_side: usize,
  locations: Vec<Location>,
  sides: Vec<Side>,
}
impl WesnothState {
  fn get (&self, x: i32,y: i32)->&Location {& self.locations [(x+y*self.map.width) as usize]}
  fn get_mut (&mut self, x: i32,y: i32)->&mut Location {&mut self.locations [(x+y*self.map.width) as usize]}
  fn is_enemy (&self, side: usize, other: usize)->bool {self.sides [side].enemies.contains (& other)}
}

#[derive (Clone, Serialize, Deserialize)]
enum WesnothMove {
  Move {
    src_x: i32, src_y: i32, dst_x: i32, dst_y: i32,
  },
  Attack {
    src_x: i32, src_y: i32, dst_x: i32, dst_y: i32, attack_x: i32, attack_y: i32, weapon: usize,
  },
  Recruit {
    dst_x: i32, dst_y: i32, unit_type: String,
  },
  EndTurn,
}

fn represent_bool (value: bool)->f32 {if value {1.0} else {0.0}}

fn represent_unit (state: & WesnothState, unit: & Unit)->Vec<f32> {
  vec![
    unit.x as f32, unit.y as f32,
    unit.moves as f32, unit.attacks_left as f32,
    unit.hitpoints as f32, (unit.max_experience - unit.experience) as f32,
    represent_bool (!state.is_enemy (state.current_side, unit.side)), represent_bool (unit.canrecruit),
    unit.max_hitpoints as f32, unit.max_moves as f32,
    represent_bool (unit.slowed), represent_bool (unit.poisoned), represent_bool (unit.not_living),
    unit.alignment as f32,
    represent_bool (unit.zone_of_control),
    unit.resistance.get ("blade").unwrap().clone() as f32,
    unit.resistance.get ("pierce").unwrap().clone() as f32,
    unit.resistance.get ("impact").unwrap().clone() as f32,
    unit.resistance.get ("fire").unwrap().clone() as f32,
    unit.resistance.get ("cold").unwrap().clone() as f32,
    unit.resistance.get ("arcane").unwrap().clone() as f32,

    
  ]
}

fn represent_wesnoth_move (state: &WesnothState, input: & WesnothMove)->NeuralInput {
  match input {
    &WesnothMove::Move {src_x, src_y, dst_x, dst_y} => NeuralInput {
      input_type: "move".to_string(),
      vector: [dst_x as f32, dst_y as f32].into_iter().cloned().chain(represent_unit (state, &state.get (src_x, src_y).unit.as_ref().unwrap())).collect()
    },
    & WesnothMove::Attack {src_x, src_y, dst_x, dst_y, attack_x, attack_y, weapon} => unimplemented!(),
    &WesnothMove::Recruit {dst_x, dst_y, ref unit_type} => {
      let mut example = state.map.config.unit_type_examples.get (unit_type).unwrap().clone();
      example.side = state.current_side;
      example.x = dst_x;
      example.y = dst_y;
      
      NeuralInput {
        input_type: "move".to_string(),
        vector: represent_unit (state, & example),
      }
    },
    & WesnothMove::EndTurn => unreachable!(),
  }
}
fn apply_wesnoth_move (state: &mut WesnothState, input: & WesnothMove)->Vec<NeuralInput> {
  unimplemented!()
}

struct Replay {
  initial_state: Arc <WesnothState>,
  final_state: Arc <WesnothState>,
  neural_moves: Vec<NeuralInput>,
  wesnoth_moves: Vec<WesnothMove>,
  neural_inputs: Vec<NeuralInput>,
  branches: Vec<Replay>,
  scores_by_side: Vec<f32>,
}


fn analyze_fitness (replay: & Replay, analyzer: & Organism)->f32 {
  let mut memory = initial_memory (analyzer);
  for neural_input in replay.neural_inputs.iter() {
    memory = next_memory (analyzer, &memory, neural_input);
  }
  assert!(!replay.branches.is_empty());
  let mut choices: Vec<_> = replay.branches.iter().map (| branch | (
    branch.scores_by_side [replay.final_state.current_side],
    evaluate_move (analyzer, &memory, &branch.neural_moves [0])
  )).collect();

  let mut unadjusted = 0.0;
  let mut best_possible = 0.0;
  let mut worst_possible = 0.0;
  
  choices.sort_by (|a, b| a.1.partial_cmp(&b.1).unwrap());
  for (index, choice) in choices.iter().enumerate() {
    unadjusted += (choices.len() - index) as f32 * choice.0;
  }
  choices.sort_by (|a, b| a.0.partial_cmp(&b.0).unwrap());
  for (index, choice) in choices.iter().enumerate() {
    worst_possible += (choices.len() - index) as f32 * choice.0;
    best_possible += (1 + index) as f32 * choice.0;
  }
  
  (unadjusted - worst_possible)/(best_possible - worst_possible)
}


fn main() {
  println!("{}", serde_json::to_string (&random_organism(vec![5,5])).unwrap());
}
