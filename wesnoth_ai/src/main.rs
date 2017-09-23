#![feature (rustc_macro, slice_patterns)]

extern crate serde;
extern crate serde_json;
#[macro_use]
extern crate serde_derive;
extern crate rand;
extern crate crossbeam;

macro_rules! printerr(
    ($($arg:tt)*) => { {use std::io::Write;
        let r = write!(&mut ::std::io::stderr(), $($arg)*);
        r.expect("failed printing to stderr");
    } }
);
macro_rules! printlnerr(
    ($($arg:tt)*) => { {use std::io::Write;
        let r = writeln!(&mut ::std::io::stderr(), $($arg)*);
        r.expect("failed printing to stderr");
    } }
);

use std::collections::{HashMap, HashSet};
use std::sync::{Arc,Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use rand::{Rng, random};
use serde::Serialize;
use serde::de::DeserializeOwned;

mod fake_wesnoth;
mod rust_lua_shared;
mod naive_ai;
mod simple_lookahead_ai;
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
  bias: Vec<f64>,
}
#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Matrix {
  input_size: usize,
  output_size: usize,
  weights: Vec<f64>,

}


fn random_matrix (input_size: usize, output_size: usize)->Matrix {
  let mut result = Matrix {
    input_size: input_size,
    output_size: output_size,
    weights: Vec::with_capacity (input_size*output_size),
  };
  for _ in 0..input_size*output_size {
    result.weights.push (random::<f64>() * 2.0 - 1.0);
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
fn mutate_vector (vector: &mut Vec<f64>, mutation_rate: f64, mutation_size: f64) {
  for value in vector.iter_mut() {
    if mutation_rate > random::<f64>()*100.0 {
      *value += ((random::<f64>() *2.0f64 - 1.0f64) * mutation_size*0.2f64) as f64;
    }
  }
}

#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct Memory {
  layers: Vec<Vec<f64>>,
}
#[derive (Clone, Serialize, Deserialize, Debug)]
pub struct NeuralInput {
  input_type: String,
  vector: Vec<f64>
}

/*
pub struct Replay {
  initial_state: Arc <fake_wesnoth::State>,
  final_state: Arc <fake_wesnoth::State>,
  neural_moves: Vec<NeuralInput>,
  wesnoth_moves: Vec<WesnothMove>,
  neural_inputs: Vec<NeuralInput>,
  branches: Vec<Replay>,
  scores_by_side: Vec<f64>,
}


fn analyze_fitness (replay: & Replay, analyzer: & Organism)->f64 {
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
    unadjusted += (choices.len() - index) as f64 * choice.0;
  }
  choices.sort_by (|a, b| a.0.partial_cmp(&b.0).unwrap());
  for (index, choice) in choices.iter().enumerate() {
    worst_possible += (choices.len() - index) as f64 * choice.0;
    best_possible += (1 + index) as f64 * choice.0;
  }
  
  (unadjusted - worst_possible)/(best_possible - worst_possible)
}

fn play_move (state: &mut fake_wesnoth::State, replay: &mut Replay, action: & WesnothMove) {
  
}*/

fn generate_starting_state (map: Arc <fake_wesnoth::Map>, players: &mut Vec<Box <fake_wesnoth::Player>>)->fake_wesnoth::State {
  let mut side_assignments: Vec<_> = (0.. players.len()).collect();
  rand::thread_rng().shuffle (&mut side_assignments);
  let mut locations = map.locations.clone();
  let mut sides = Vec::new();
  for (index, _) in players.into_iter().enumerate() {
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
    });
  }
  let state = fake_wesnoth::State {
    map: map,
    current_side: 0,
    locations: locations,
    sides: sides,
    time_of_day: rand::thread_rng().gen_range (0, 6),
    turn: 1,
    max_turns: 10,
    scores: None,
  };
  for player in players.iter_mut() {
    player.turn_started (& state);
  }
  state
}

fn make_player (map: & fake_wesnoth::Map, organism: Arc <Organism>)->Box <fake_wesnoth::Player> {
  Box::new (NeuralPlayer {
    organism: organism.clone(),
    memory: initial_memory (& organism),
    unit_moves: vec![None; (map.width*map.height) as usize],
  })
}

fn draw_state (state: & fake_wesnoth::State) {
  use std::io;
  
  for second in 1.. state.map.height+1 {
    for first in 1..state.map.width+1 {
      let location = state.get (first, second);
      if let Some (unit) = location.unit.as_ref() {
        printerr! ("{:2} {}, ", unit.hitpoints, unit.moves);
      }
      else {
        printerr! ("{:5} ", location.terrain);
      }
    }
    printerr! ("\n");
  }
  
  let mut input = String::new();
  io::stdin().read_line (&mut input).unwrap();
}

fn compete (map: Arc <fake_wesnoth::Map>, mut players: Vec<Box <fake_wesnoth::Player >>)->Vec<f64> {
  //printlnerr!("Beginning competition...");
  //let start = ::std::time::Instant::now();
  let mut state = generate_starting_state (map, &mut players);
  while state.scores.is_none() {
    //draw_state (& state);
    let choice = players [state.current_side].choose_move (&mut state);
    fake_wesnoth::apply_move (&mut state, &mut players, &choice);
  }
  //let duration = start.elapsed();
  //printlnerr!("Competition completed in {} seconds + {} nanoseconds", duration.as_secs(), duration.subsec_nanos());
  state.scores.unwrap()
}
//fn play_game (player: & Organism, map: & fake_wesnoth::Map)->Replay {
  //
//}

fn play_show_game (map: Arc <fake_wesnoth::Map>, mut players: Vec<Box <fake_wesnoth::Player >>) {
  let mut state = generate_starting_state (map, &mut players);
  while state.scores.is_none() {
    draw_state (& state);
    let choice = players [state.current_side].choose_move (&mut state);
    fake_wesnoth::apply_move (&mut state, &mut players, &choice);
  }
}

const DEFAULT_TRAINING_TIME: u64 = 120;
fn random_organism_default()->Arc<Organism> {
  let layers = rand::thread_rng().gen_range (1, 4);
  let organism = random_organism (vec![((122500/layers) as f64).sqrt() as usize; layers]);
  Arc::new (organism)
}
fn random_mutant_default(organism: & Organism)->Arc<Organism> {
  if random::<f64> () <0.2 {
    random_organism_default()
  }
  else {
    Arc::new (mutated_organism (organism))
  }
}

fn original_training (map: Arc <fake_wesnoth::Map>, seconds: u64)->Arc <Organism> {
  struct Stats {
    rating: f64,
  }

  let mut organisms = Vec::new();
  let start = ::std::time::Instant::now();
  let mut iteration: usize = 0;
  let mut organism_count: usize = 10;
  while start.elapsed().as_secs() < seconds {
    iteration += 1;
    let was_empty = organisms.is_empty();
    while organisms.len() < organism_count {
      organisms.push ((random_organism_default(), Stats {rating: 0.0}));
      if was_empty {
        organisms.push ((random_organism_default(), Stats {rating: 0.0}));
      }
      else {
        let new_organism = random_mutant_default(&organisms [0].0);
        organisms.push ((new_organism, Stats {rating: 0.0}));
      }
    }
    let mut any_games = false;
    for index in 0..(organisms.len()-1) {
      if organisms [index].1.rating <= organisms [index+1].1.rating + 2.0 {
        let results = compete (map.clone(), vec![make_player (&map, organisms [index].0.clone()), make_player (&map, organisms [index + 1].0.clone())]);
        organisms [index].1.rating += results [0];
        organisms [index + 1].1.rating += results [1];
        any_games = true;
      }
    }
    if !any_games { organism_count += 1; }
    organisms.sort_by (|a, b| b.1.rating.partial_cmp(&a.1.rating).unwrap());
    organisms.retain (| &(_, Stats {ref rating})| *rating >= 0.0);
  }
  printlnerr!("Original training completed {} iterations ", iteration);
  organisms [0].0.clone()
}

use std::{thread,time};
use std::sync::mpsc::channel;
use crossbeam::sync::{MsQueue as Exchange};

fn first_to_beat_the_champion_training (map: Arc <fake_wesnoth::Map>, seconds: u64)->Arc <Organism> {
  let mut champion = random_organism_default();
  let mut turnovers: usize = 0;
  let start = ::std::time::Instant::now();
  let mut games: usize = 0;
  while start.elapsed().as_secs() < seconds {
    let wins_needed = ((turnovers + 2) as f64).log2() as i32;
    let (send, receive) = channel();
    let (count_send, count_receive) = channel();
    for _ in 0..3 {
      let send = send.clone();
      let count_send = count_send.clone();
      let champion = champion.clone();
      let map = map.clone();
      thread::spawn (move | | {
        'a: loop {
          let challenger = random_mutant_default (&champion);
          for _ in 0..wins_needed { 
            let results = compete (map.clone(), vec![make_player (&map, champion.clone()), make_player (&map, challenger.clone())]);
            if let Err (_) = count_send.send (()) {return;}
            if results [1] <= 0.0 {
              continue 'a;
            }
          }
          let _ = send.send(challenger);
          return;
        }
      });
    }
    while start.elapsed().as_secs() < seconds {
      if let Ok (_) = count_receive.try_recv() {
        games += 1;
      }
      if let Ok (new_champion) = receive.try_recv() {
        champion = new_champion;
        turnovers += 1;
        break;
      }
      thread::sleep(time::Duration::from_millis(1));
    }
  }
  printlnerr!("Champion training used {} games, with {} turnovers", games, turnovers);
  champion
}



fn ranked_lineages_training (map: Arc <fake_wesnoth::Map>, seconds: u64)->Arc <Organism> {
  struct Lineage {
    id: usize,
    members: Vec<Member>,
  }
  #[derive (Clone)]
  struct Member {
    organism: Arc <Organism>,
    id: usize,
    parent_rank: i32,
    rank: i32,
    games: i32,
    lineage_id: usize,
  }
  let needed_games = Exchange::<Option <Vec<Member>>>::new();
  let game_results = Exchange::new();
  
  crossbeam::scope (| scope | {
    let mut lineages = Vec::<Lineage>::new();
    let mut next_id: usize = 0;
    let start = ::std::time::Instant::now();
    let mut games: usize = 0;

    for _ in 0..3 {
      let needed_games = & needed_games;
      let game_results = & game_results;
      let map = & map;
      scope.spawn (move || {
        while let Some (mut game) = needed_games.pop() {
          let results = compete (map.clone(), vec![make_player (&map, game [0].organism.clone()), make_player (&map, game [1].organism.clone())]);
          game [0].rank += results [0] as i32;
          game [1].rank += results [1] as i32;
          game [0].games += 1;
          game [1].games += 1;
          game_results.push (game);
        }
      });
    }
    let mut games_planned: usize = 0;
    let mut lineage_count: usize = 3;
    while start.elapsed().as_secs() < seconds {
      let settling = start.elapsed().as_secs() >= seconds - 5;
      //lineages.retain (| lineage | !lineage.members.is_empty());
      while !settling && lineages.len() < lineage_count {
        lineages.push (Lineage {
          id: next_id,
          members: vec![Member {
            organism: random_organism_default(),
            id: next_id + 1,
            parent_rank: 0, rank: 0, games: 0, lineage_id: next_id}]
        });
        next_id += 2;
      }
      for lineage in lineages.iter_mut() {
        lineage.members.sort_by_key (| member | -member.rank);
        if !settling && lineage.members.len() > 1 && lineage.members[0].rank >lineage.members [lineage.members.len() - 1].rank + 6 {
          lineage.members.pop();
        }
        while !settling && lineage.members.len() < 4 {
          if lineage.members.is_empty() {
            lineage.members.push (Member {
              organism: random_organism_default(),
              id: next_id,
                          parent_rank: 0, rank: 0, games: 0, lineage_id: lineage .id
            });
          }
          else {
            let new_member = Member {
              organism: random_mutant_default(&lineage.members [0].organism),
              id: next_id,
                        parent_rank: lineage.members [0].rank, rank: lineage.members [0].rank, games: 0, lineage_id: lineage .id
            };
            lineage.members.push (new_member);
          }
          next_id += 1;
        }
      }
      lineages.sort_by_key (| lineage | -lineage.members[0].rank);
      let mut failures = 0;
      while games_planned < 5 && failures < 50 {
        let lineages = (
          rand::thread_rng().choose (&lineages).unwrap(),
          rand::thread_rng().choose (&lineages).unwrap());
        if lineages.0 .id != lineages.1 .id && !lineages.0.members.is_empty() && !lineages.1.members.is_empty() {
          let indices = (
            rand::thread_rng().gen_range (0, lineages.0.members.len()),
            rand::thread_rng().gen_range (0, lineages.1.members.len()),
          );
          let competitors = vec![
            lineages.0.members [indices.0].clone(),
            lineages.1.members [indices.1].clone(),
          ];
          if (competitors[0].rank - competitors[1].rank).abs() <= 5 {
            games_planned += 1;
            needed_games.push (Some (competitors));
          }
          else { failures += 1; }
        }
        else { failures += 1; }
      }
      if failures == 50 { lineage_count += 1; }
      while let Some (game) = game_results.try_pop() {
        games += 1;
        games_planned -= 1;
        for member in game.into_iter() {
          if let Some (lineage) = lineages.iter_mut().find (| lineage | lineage .id == member.lineage_id){
            if let Some (index) = lineage.members.iter().position (| member2 | member2.id == member.id) {
              if settling || member.rank >= member.parent_rank || member.games >= (1 << (member.parent_rank - member.rank)) {
                lineage.members [index] = member;
              }
              else {
                lineage.members.remove (index);
              }
            }
          }
        }
      }
      thread::sleep(time::Duration::from_millis(1));
    }
    for _ in 0..3 {needed_games.push (None);}
    printlnerr!("Lineages training used {} games ", games);
    
    
    lineages [0].members [0].organism.clone()
  })
}



fn against_naive_training (map: Arc <fake_wesnoth::Map>, seconds: u64)->Arc <Organism> {
  #[derive (Clone)]
  struct Contestant {
    organism: Arc <Organism>,
    wins: i32,
    games: i32,
  }
  let champion = Arc::new (Mutex::new (Contestant {
    organism: random_organism_default(),
    wins: 0, games: 0,
  }));
  let turnovers = Arc::new (AtomicUsize::new (0));
  let games = Arc::new (AtomicUsize::new (0));
  let mut threads = Vec::new();

  for _ in 0..3 {
    let champion = champion.clone();
    let map = map.clone();
    let turnovers = turnovers.clone();
    let games = games.clone();
        
    threads.push (thread::spawn (move | | {
      let start = ::std::time::Instant::now();
      'a: while start.elapsed().as_secs() < seconds {
        let mut challenger = Contestant {
          organism: random_mutant_default (&champion.lock().unwrap().organism),
          wins: 0, games: 0,
        };
        loop { 
          let results = compete (map.clone(), vec![make_player (&map, challenger.organism.clone()), Box::new (naive_ai::Player::new(&map))]);
          challenger.games += 1;
          games.fetch_add (1, Ordering::Relaxed);
          
          if results [0] > 0.0 {
            challenger.wins += 1;
            //printlnerr!("Win {}/{}", challenger.wins, challenger.games);
          }
          else {
            //printlnerr!("Lose {}/{}", challenger.wins, challenger.games);
            if challenger.wins == 0 {
              continue 'a;
            }
          }
          let mut lock = champion.lock().unwrap();
          if lock.games >= challenger.games && lock.wins*challenger.games > challenger.wins*lock.games {
            continue 'a;
          }
          if challenger.games > lock.games || (challenger.games == lock.games && lock.wins*challenger.games < challenger.wins*lock.games) {
            *lock = challenger.clone();
            printlnerr!("Set {}/{}", challenger.wins, challenger.games);
            let desired_champion_games = (((start.elapsed().as_secs() + 2) as f64).log2()*20.0) as i32;
            if challenger.games >= desired_champion_games {
              turnovers.fetch_add (1, Ordering::Relaxed);
              continue 'a;
            }
          }
        }
      }
    }));
  }
  for thread in threads {thread.join().unwrap();}
  let result;
  {
    let lock = champion.lock().unwrap();
    result = lock.organism.clone();
    printlnerr!("Against-naive training used {} games, with {} turnovers. Current champion scored {}/{}", games.load (Ordering::Relaxed), turnovers.load (Ordering::Relaxed), lock.wins, lock.games);
  }
  for _ in 0..3 {
    play_show_game (map.clone(), vec![make_player (&map, result.clone()), Box::new (naive_ai::Player::new(&map))]);
  }
  result
}





fn tournament (map: Arc <fake_wesnoth::Map>, organisms: Vec<(Arc <Organism>, & 'static str)>)->Arc <Organism> {
  let mut total_scores = vec![0.0; organisms.len()];
  for (first_index, first) in organisms.iter().enumerate() {
    for (second_index, second) in organisms.iter().enumerate() {
      if first_index != second_index {
        for _ in 0..10 {
          let results = compete (map.clone(), vec![make_player (&map, first.0.clone()), make_player (&map, second.0.clone())]);
          total_scores [first_index] += results [0];
          total_scores [second_index] += results [1];
        }
      }
    }
  }
  printlnerr!("Tournament scores:");
  let mut best_index = 0;
  for (index, stuff) in organisms.iter().enumerate() {
    printlnerr!("{}: {}", stuff.1, total_scores [index]);
    if total_scores [index] > total_scores [best_index] {
      best_index = index;
    }
  }
  organisms [best_index].0.clone()
}

use std::fs::File;
use std::io::{Read,BufRead};
fn main_old() {
  let mut f = File::open("tiny_close_relation_default.json").unwrap();
  let mut s = String::new();
  f.read_to_string(&mut s).unwrap();
  let tiny_close_relation_data: Arc<fake_wesnoth::Map> = serde_json::from_str(&s).unwrap();
  let map = tiny_close_relation_data;
    
  fn do_test <Input: Serialize, Output: Serialize, Function: Fn (Input)->Output> (input: Input, function_name: &str, function: Function) {
    println!( "{{
      tested_function = [=[{}]=],
      input = [=[{}]=],
      output = [=[{}]=]}},", function_name, serde_json::to_string (& input).unwrap(), serde_json::to_string (& function (input)).unwrap());
  }
  
  let winner = tournament (map.clone(), vec![
    (against_naive_training(map.clone(), 1200), "against_naive_training"),
    (random_organism_default(), "no training"),
    (original_training (map.clone(), DEFAULT_TRAINING_TIME), "original_training"),
    (first_to_beat_the_champion_training (map.clone(), DEFAULT_TRAINING_TIME), "first_to_beat_the_champion_training"),
    (ranked_lineages_training(map.clone(), DEFAULT_TRAINING_TIME), "ranked_lineages_training"),
  ]);
  
  println!("return {{
    organism = [============================[{}]============================],
    tests = {{", serde_json::to_string (& winner).unwrap());
  
  for _ in 0..20 {
    do_test (random::<f64>()*50.0 - 25.0, "hyperbolic_tangent", hyperbolic_tangent);
  }
  let test_organism =random_organism (vec![3, 3]);
  do_test (
    (test_organism.clone(),
      initial_memory (&test_organism),
      NeuralInput {input_type: "turn_started".to_string(), vector: vec![40.0,4.0,40.0,4.0]}),
    "next_memory",
    |(organism, memory, input)| next_memory (&organism, &memory, &input));
  
  println!("}},
  }}")
}

fn receive_from_lua<R: BufRead, T: DeserializeOwned>(mut reader: R)->T {
  loop {
    let mut line = String::new();
    reader.read_line (&mut line).unwrap();
    
    if line == "Lua_to_Rust_transfer\n" {
      line.clear();
      reader.read_line (&mut line).unwrap();
      return serde_json::from_str(&line).unwrap()
    }
    //line.truncate(30);
    print!("Received {}", line);
  }
}
#[derive (Serialize)]
struct Message<T> {
  index: usize,
  data: T
}
use std::cell::Cell;
thread_local! {
  static NEXT_SERIAL_NUMBER: Cell <usize> = Cell::new (0);
}
fn new_serial_number()->usize {
  NEXT_SERIAL_NUMBER.with (| cell | {
    let result = cell.get();
    cell.set (result + 1);
    result
  })
}
fn send_to_lua <T: Serialize> (path: &Path, value: T) {
  let mut file = File::create (path).unwrap();
  serde_json::to_writer (&mut file, &Message {index: new_serial_number(), data: value}).unwrap();
}


use std::path::Path;
use std::io;
use fake_wesnoth::Player;
fn main() {
  let stdin = io::stdin();
  let mut input = stdin.lock();
  let path_arg = ::std::env::args().nth(1).unwrap();
  let path = Path::new(&path_arg);
  loop {
    let state: fake_wesnoth::State = receive_from_lua(&mut input);
    println!("\n\n\nReceived data from Wesnoth!\n\n\n");
    //let mut player = naive_ai::Player::new(&*state.map);
    let mut player = simple_lookahead_ai::Player::new (| state, side | Box::new (naive_ai::Player::new(&*state.map)));
    let choice = player.choose_move (&state);
    send_to_lua (&path, choice);
  }
}
