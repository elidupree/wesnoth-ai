use std::sync::mpsc::Receiver;
use super::*;
use {send_to_lua, fake_wesnoth};

use std::path::Path;
use std::io;
use std::time::Duration;
use fake_wesnoth::Player;

widget_ids!(struct StateIds {
  hexes[]
});

pub fn side_color (side: usize)->conrod::color::Color {
  match side {
    0 => conrod::color::RED,
    1 => conrod::color::BLUE,
    2 => conrod::color::GREEN,
    _=> conrod::color::WHITE,
  }
}

pub fn draw_state (interface: &mut conrod::UiCell, state: & fake_wesnoth::State, offsets: [f64; 2]) {
  let hex_size = 40f64;
  let meta_width = 40f64;
  let map_size = [hex_size*state.map.width as f64, hex_size*(state.map.height as f64 + 0.5)];
  widget::Rectangle::outline_styled (map_size, conrod::widget::primitive::line::Style::solid().color (side_color (state.current_side)))
    .xy ([offsets[0] + meta_width + map_size[0]/2.0, offsets[1] - map_size[1]/2.0])
    .set(interface.widget_id_generator().next(), interface);
  
  let side_height = 30f64;
  for (index, info) in state.sides.iter().enumerate() {
    widget::Text::new(&format!("{}", info.gold))
      .xy ([offsets[0] + meta_width/2.0, offsets[1] - side_height*(index as f64 + 0.5)])
        .color(side_color (index))
        .font_size(14)
        .set(interface.widget_id_generator().next(), interface);
  }
  
  for x in 1..(state.map.width+1) {
    let vertical_offset = offsets [1] - if (x & 1) == 0 {hex_size} else {hex_size/2.0};
    let horizontal_offset = offsets [0] + meta_width + hex_size/2.0;
    for y in 1..(state.map.height+1) {
      let location = state.get (x,y);
      let rectangle_id = interface.widget_id_generator().next();
      let owner_color = side_color (location.village_owner.unwrap_or (999));
      widget::Rectangle::outline_styled ([hex_size, hex_size], conrod::widget::primitive::line::Style::solid().color (owner_color))
        .xy ([(x-1) as f64*hex_size + horizontal_offset, -(y-1) as f64*hex_size + vertical_offset])
        .set(rectangle_id, interface);
        
      widget::Text::new(&format!("{}", location.terrain))
        .middle_of (rectangle_id)
        .color(owner_color)
        .font_size(10)
        .set(interface.widget_id_generator().next(), interface);
        
      if let Some(unit) = location.unit.as_ref() {
        widget::Rectangle::fill_with ([hex_size/4.0, hex_size*unit.hitpoints as f64/unit.unit_type.max_hitpoints as f64], side_color (unit.side))
          .bottom_left_of (rectangle_id)
          .set(interface.widget_id_generator().next(), interface);
        widget::Rectangle::fill_with ([hex_size/8.0, hex_size*unit.experience as f64/unit.unit_type.max_experience as f64], side_color (unit.side))
          .bottom_left_of (rectangle_id).right(hex_size/4.0)
          .set(interface.widget_id_generator().next(), interface);
        widget::Rectangle::fill_with ([hex_size/8.0, hex_size*unit.moves as f64/10f64], side_color (unit.side))
          .bottom_right_of (rectangle_id)
          .set(interface.widget_id_generator().next(), interface);
      }
    }
  }
}

#[derive (Clone)]
struct DisplayedState {
  depth: usize,
  size: [f64; 2],
  state: Arc <fake_wesnoth::State>,
  text: String,
}

use conrod::{self, widget, Colorable, Positionable, Widget};
use conrod::backend::glium::glium::{self, Surface};
pub fn main_loop(path: &Path, receiver: Receiver <fake_wesnoth::State>) {
  const WIDTH: u32 = 600;
  const HEIGHT: u32 = 800;

  let mut events_loop = glium::glutin::EventsLoop::new();
  let window = glium::glutin::WindowBuilder::new()
    .with_title("wesnoth-ai")
    .with_dimensions(WIDTH, HEIGHT);
  let context = glium::glutin::ContextBuilder::new()
    .with_vsync(true)
    .with_multisampling(4);
  let display = glium::Display::new(window, context, &events_loop).unwrap();

  let mut ui = conrod::UiBuilder::new([WIDTH as f64, HEIGHT as f64]).build();

  widget_ids!(struct Ids {
    text
  });
  let ids = Ids::new(ui.widget_id_generator());

  const FONT_PATH: &'static str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/assets/fonts/NotoSans/NotoSans-Regular.ttf");
  ui.fonts.insert_from_file(FONT_PATH).unwrap();

  let mut renderer = conrod::backend::glium::Renderer::new(&display).unwrap();
  let image_map = conrod::image::Map::<glium::texture::Texture2d>::new();
  let mut events = Vec::new();
  
  let mut current_state: Option <Arc<fake_wesnoth::State>> = None;
  let mut redraw = true;
  let (ai_sender, ai_receiver) = channel();
  let (tree_sender, tree_receiver) = channel();
  let mut proceeding = false;//true;
  
  let mut states_display = Vec::new();
  let mut which_displayed = 0;
  let mut focused = 0;
  let depth_width = 30.0;

  'render: loop {
    if let Ok(state) = receiver.try_recv() {
      println!("Received data from Wesnoth!");
      let state = Arc::new(state);
      current_state = Some(state.clone());
      redraw = true;
      let sender = ai_sender.clone();
      let tree_sender = tree_sender.clone();
      
      thread::spawn (move | | {
        //let mut player = naive_ai::Player::new(&*state.map);
        //let mut player = simple_lookahead_ai::Player::new (| state, side | Box::new (naive_ai::Player::new(&*state.map)));
        let mut player = monte_carlo_ai::Player::new (| state, side | Box::new (naive_ai::Player::new(&*state.map)));
        let choice = player.choose_move (&state);
        let _ = sender.send (choice);
        let _ = tree_sender.send (player.last_root.unwrap());
      });
      
      //hack 
      /*states_display.clear();
      let mut playout_state = (**current_state.as_ref().unwrap()).clone();
      let starting_turn = playout_state.turn;
      states_display.push (Arc::new (playout_state.clone()));
      let mut players: Vec<_> = playout_state.sides.iter().map (| _side | Box::new (naive_ai::Player::new(&*playout_state.map)) as Box<Player>).collect();
      while playout_state.scores.is_none() && playout_state.turn < starting_turn + 10 {
        let choice = players [playout_state.current_side].choose_move (& playout_state) ;
        fake_wesnoth::apply_move (&mut playout_state, &mut players, & choice);
        states_display.push (Arc::new (playout_state.clone()));
      }*/
    }
    if proceeding {
      if let Ok(response) = ai_receiver.try_recv() {
        send_to_lua (&path, response);
      }
    }
    if let Ok(root) = tree_receiver.try_recv() {
      //let layers = Vec::new();
      states_display.clear();
      let mut frontier = vec![(root, [0.0, 1.0])];
      let mut depth = 0;
      while !frontier.is_empty() {
        let mut next_frontier = Vec::new();
        for (mut node, size) in frontier {
          let diff = size[1]-size[0];
          node.moves.sort_by_key (|a| a.visits);
          let mut prior_visits = Cell::new(0);
          let node_visits = node.visits;
          next_frontier.extend(
            node.moves.into_iter()
            .flat_map(|a| {
              a.determined_outcomes.into_iter().map(|out| {
                let out_size = [size[0] + diff*prior_visits.get() as f64/node_visits as f64,
                   size[0] + diff*(prior_visits.get()+out.visits) as f64/node_visits as f64];
                prior_visits.set(prior_visits.get() + out.visits);
                (out, out_size)
              })
            })
          );
          states_display.push (DisplayedState {
            depth: depth,
            size: size,
            state: node.state.clone(),
            text: format!("{:.2}", node.total_score/node.visits as f64),
          });
        }
        depth += 1;
        frontier = next_frontier;
      }
      redraw = true;
    }
    
    events.clear();
    events_loop.poll_events(|event| {events.push (event) ;});
    for event in events.drain(..) {
      match event.clone() {
        glium::glutin::Event::WindowEvent { event, .. } => {
          match event {
            glium::glutin::WindowEvent::Closed |
            glium::glutin::WindowEvent::KeyboardInput {
              input: glium::glutin::KeyboardInput {
                virtual_keycode: Some(glium::glutin::VirtualKeyCode::Escape),
                ..
              },
              ..
            } => break 'render,
            glium::glutin::WindowEvent::KeyboardInput {
              input: glium::glutin::KeyboardInput {
                virtual_keycode: Some(code),
                state: glium::glutin::ElementState::Pressed,
                ..
              },
              ..
            } => {
              match code {
                glium::glutin::VirtualKeyCode::A => {proceeding = !proceeding;}
                _=>(),
              }
            },
            glium::glutin::WindowEvent::MouseMoved {position: (x,y), ..} => {
              let depth = (x/depth_width) as usize;
              let height = 1.0-(y/HEIGHT as f64);
              which_displayed = states_display.iter()
                .position(|a|a.depth == depth && a.size[0] < height && a.size[1] > height).unwrap_or(99999999);
              redraw = true;
            },
            glium::glutin::WindowEvent::MouseInput {state: glium::glutin::ElementState::Pressed, ..} => {
              focused = which_displayed;
              redraw = true;
            },
            _ => (),
          }
        }
        _ => (),
      };    

      let input = match conrod::backend::winit::convert_event(event, &display) {
        None => continue,
        Some(input) => input,
      };
      ui.handle_event(input);
      /*let ui = &mut ui.set_widgets();
      
      widget::Text::new("Hello World!")
        .middle_of(ui.window)
        .color(conrod::color::WHITE)
        .font_size(32)
        .set(ids.text, ui);*/
    };
    
    if redraw {
      let ui = &mut ui.set_widgets();
      if let Some(state) = current_state.as_ref() {
        draw_state (ui, &state, [0.0,0.0]);
      }
      if let Some(displayed_state) = states_display.get(which_displayed) {
        draw_state (ui, &displayed_state.state, [0.0,200.0]);
      }
      if focused >= states_display.len() {focused = 0;}
      for (index, displayed_state) in states_display.iter().enumerate() {
        let focused = states_display.get (focused).unwrap();
        if displayed_state.size [0] > focused.size[1] || displayed_state.size [1] < focused.size[0] { continue; }
        let focused_diff = focused.size [1] - focused.size [0];
        let state = &displayed_state.state;
        
        let diff = displayed_state.size[1] - displayed_state.size[0];
        if diff < focused_diff * 0.01 { continue; }
        let pos = [
          -(WIDTH as f64)/2.0 + depth_width/2.0 + depth_width*displayed_state.depth as f64,
          -(HEIGHT as f64)/2.0 + HEIGHT as f64 * (displayed_state.size[0]-focused.size[0] + diff/2.0)/focused_diff
        ];
        if index == which_displayed {
          widget::Rectangle::fill_with ([depth_width, HEIGHT as f64*diff/focused_diff], side_color (state.current_side))
        }
        else {
          widget::Rectangle::outline_styled ([depth_width, HEIGHT as f64*diff/focused_diff], conrod::widget::primitive::line::Style::solid().color (side_color (state.current_side)))
        }
          .xy (pos)
          .set(ui.widget_id_generator().next(), ui);
        widget::Text::new(&displayed_state.text)
          .color(side_color (state.current_side))
          .font_size(12)
          .xy (pos)
          .set(ui.widget_id_generator().next(), ui);
      }
      redraw = false;
    }
    
    if let Some(primitives) = ui.draw_if_changed() {
      renderer.fill(&display, primitives, &image_map);
      let mut target = display.draw();
      target.clear_color(0.0, 0.0, 0.0, 1.0);
      renderer.draw(&display, &mut target, &image_map).unwrap();
      target.finish().unwrap();
    }
    
    thread::sleep (Duration::from_millis(10));
  }
}