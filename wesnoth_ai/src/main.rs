#![feature (plugin, custom_derive)]
#![plugin (serde_macros)]

extern crate serde;
extern crate serde_json;
extern crate rand;

use std::collections::HashMap;
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
struct WesnothState {
  current_side: usize,
  
}

#[derive (Clone, Serialize, Deserialize)]
enum WesnothInput {

}

fn apply_wesnoth_input (state: &mut WesnothState, input: & WesnothInput)->Vec<NeuralInput> {
  unimplemented!()
}

struct Replay {
  initial_state: Arc <WesnothState>,
  final_state: Arc <WesnothState>,
  neural_moves: Vec<NeuralInput>,
  wesnoth_inputs: Vec<WesnothInput>,
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
