
use std::sync::mpsc::Receiver;
use super::*;
use {send_to_lua, fake_wesnoth};

use std::path::Path;
use std::io;
use std::time::Duration;
use fake_wesnoth::Player;
pub fn main_loop(path: &Path, receiver: Receiver <fake_wesnoth::State>) {
  loop {
    if let Ok(state) = receiver.try_recv() {
      println!("Received data from Wesnoth!");
      //let mut player = naive_ai::Player::new(&*state.map);
      let mut player = simple_lookahead_ai::Player::new (| state, side | Box::new (naive_ai::Player::new(&*state.map)));
      let choice = player.choose_move (&state);
      send_to_lua (&path, choice);
    }
    
    thread::sleep (Duration::from_millis(10));
  }
}
