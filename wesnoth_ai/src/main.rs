#![feature (plugin, custom_derive, slice_patterns)]
#![plugin (serde_macros)]

extern crate serde;
extern crate serde_json;
extern crate rand;

macro_rules! printlnerr(
    ($($arg:tt)*) => { {use std::io::Write;
        let r = writeln!(&mut ::std::io::stderr(), $($arg)*);
        r.expect("failed printing to stderr");
    } }
);

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use rand::{Rng, random};

mod fake_wesnoth;
use fake_wesnoth::{Map as WesnothMap, State as WesnothState};
mod rust_lua_shared;
use rust_lua_shared::*;

/// One individual "organism" shareable with Lua.
/// Represents a function from game state to move evaluations.
#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Organism {
  signature: String,
  layer_sizes: Vec<usize>,
  weights_by_input: HashMap<String, Vec<LayerWeights>>,
  output_weights: Matrix,
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct LayerWeights {
  hidden_matrix: Matrix,
  input_matrix: Matrix,
  bias: Vec<f32>,
}
#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Matrix {
  input_size: usize,
  output_size: usize,
  weights: Vec<f32>,

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
    let vect = layer_sizes.iter().enumerate().map (| (index, layer_size) | {
      LayerWeights{
        input_matrix: random_matrix (if index == 0 {size.clone()} else {layer_sizes[index - 1]}, layer_size.clone()),
        hidden_matrix: random_matrix (layer_size.clone(), layer_size.clone()),
        bias: rand::thread_rng().gen_iter().take (layer_size.clone()).collect(),
      }
    }).collect();
    result.weights_by_input.insert (name.clone(), vect);
  });
  result
}

use rand::distributions::exponential::Exp1;
fn mutated_organism (original: & Organism)->Organism {
  let Exp1(mutation_rate) = random();
  let Exp1(mutation_size) = random();
  let mut result = original.clone();
  for weights in result.weights_by_input.iter_mut() {
    for something in weights.1.iter_mut() {
      mutate_vector (&mut something.hidden_matrix.weights, mutation_rate, mutation_size);
      mutate_vector (&mut something.input_matrix.weights, mutation_rate, mutation_size);
      mutate_vector (&mut something.bias, mutation_rate, mutation_size);
    }
  }
  mutate_vector (&mut result.output_weights.weights, mutation_rate, mutation_size);
  result
}
fn mutate_vector (vector: &mut Vec<f32>, mutation_rate: f64, mutation_size: f64) {
  for value in vector.iter_mut() {
    if mutation_rate > random::<f64>()*100.0 {
      *value += ((random::<f64>() *2.0f64 - 1.0f64) * mutation_size*0.2f64) as f32;
    }
  }
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Memory {
  layers: Vec<Vec<f32>>,
}
#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct NeuralInput {
  input_type: String,
  vector: Vec<f32>
}

/*
pub struct Replay {
  initial_state: Arc <fake_wesnoth::State>,
  final_state: Arc <fake_wesnoth::State>,
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

fn play_move (state: &mut fake_wesnoth::State, replay: &mut Replay, action: & WesnothMove) {
  
}*/

fn generate_starting_state (map: Arc <fake_wesnoth::Map>, players: Vec<Arc <Organism>>)->fake_wesnoth::State {
  let mut side_assignments: Vec<_> = (0.. players.len()).collect();
  rand::thread_rng().shuffle (&mut side_assignments);
  let mut locations = map.locations.clone();
  let mut sides = Vec::new();
  for (index, player) in players.into_iter().enumerate() {
    let faction = rand::thread_rng().choose (&map.config.factions).unwrap();
    let mut leader = Box::new (map.config.unit_type_examples.get (rand::thread_rng().choose (&faction.leaders).unwrap()).unwrap().clone());
    leader.x = map.starting_locations [index][0];
    leader.y = map.starting_locations [index][1];
    leader.side = index;
    leader.moves = leader.max_moves;
    leader.attacks_left = 1;
    leader.canrecruit = true;
    let location_index =((leader.x-1)+(leader.y-1)*map.width) as usize;
    locations [location_index].unit = Some (leader);
    let mut enemies = HashSet::new(); enemies.insert ((index + 1) % 2);
    sides.push (fake_wesnoth::Side {
      gold: 40,
      enemies: enemies,
      recruits: faction.recruits.clone(),
      player: player.clone(),
      memory: initial_memory (& player),
    });
  }
  fake_wesnoth::State {
    map: map,
    current_side: 0,
    locations: locations,
    sides: sides,
    time_of_day: 0,
    turn: 1,
    max_turns: 30,
    scores: None,
  }
}
fn compete (map: Arc <fake_wesnoth::Map>, players: Vec<Arc <Organism>>)->Vec<f32> {
  printlnerr!("Beginning competition...");
  let start = ::std::time::Instant::now();
  let mut state = generate_starting_state (map, players);
  while state.scores.is_none() {
    let choice = choose_move (&mut state);
    fake_wesnoth::apply_move (&mut state, &choice);
  }
  let duration = start.elapsed();
  printlnerr!("Competition completed in {} seconds + {} nanoseconds", duration.as_secs(), duration.subsec_nanos());
  state.scores.unwrap()
}
//fn play_game (player: & Organism, map: & fake_wesnoth::Map)->Replay {
  //
//}

use std::fs::File;
use std::io::Read;
fn main() {
  let mut f = File::open("tiny_close_relation_default.json").unwrap();
  let mut s = String::new();
  f.read_to_string(&mut s).unwrap();
  let tiny_close_relation_data: Arc<fake_wesnoth::Map> = serde_json::from_str(&s).unwrap();
  let map = tiny_close_relation_data;
  
  struct Stats {
    rating: f32,
  }
  fn random_organism_default()->(Arc<Organism>, Stats) {(Arc::new (random_organism (vec![50, 50, 50])), Stats {rating: 0.0})}
  let mut organisms = Vec::new();
  for iteration in 0..1000 {
    let was_empty = organisms.is_empty();
    while organisms.len() < 10 {
      organisms.push (random_organism_default());
      if was_empty || random::<f32> () <0.2 {
        organisms.push (random_organism_default());
      }
      else {
        let new_organism = Arc::new (mutated_organism (&organisms [0].0));
        organisms.push ((new_organism, Stats {rating: 0.0}));
      }
    }
    for index in 0..(organisms.len()-1) {
      if organisms [index].1.rating <= organisms [index+1].1.rating + 2.0 {
        let results = compete (map.clone(), vec![organisms [index].0.clone(), organisms [index + 1].0.clone()]);
        organisms [index].1.rating += results [0];
        organisms [index + 1].1.rating += results [1];
      }
      
    }
    organisms.sort_by (|a, b| b.1.rating.partial_cmp(&a.1.rating).unwrap());
    organisms.retain (| &(_, Stats {ref rating})| *rating >= 0.0);
  }
  
  println!("return [============================[{}]============================]", serde_json::to_string (& organisms [0].0).unwrap());
}
