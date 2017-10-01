use std::sync::mpsc::Receiver;
use super::*;
use {send_to_lua, fake_wesnoth};

use std::path::Path;
use std::io;
use std::time::Duration;
use fake_wesnoth::Player;


use conrod::{self, widget, Colorable, Positionable, Widget};
use conrod::backend::glium::glium::{self, Surface};
pub fn main_loop(path: &Path, receiver: Receiver <fake_wesnoth::State>) {
  const WIDTH: u32 = 400;
  const HEIGHT: u32 = 200;

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

  'render: loop {
    if let Ok(state) = receiver.try_recv() {
      println!("Received data from Wesnoth!");
      //let mut player = naive_ai::Player::new(&*state.map);
      let mut player = simple_lookahead_ai::Player::new (| state, side | Box::new (naive_ai::Player::new(&*state.map)));
      let choice = player.choose_move (&state);
      send_to_lua (&path, choice);
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
      let ui = &mut ui.set_widgets();
      
      widget::Text::new("Hello World!")
        .middle_of(ui.window)
        .color(conrod::color::WHITE)
        .font_size(32)
        .set(ids.text, ui);
    };
    
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