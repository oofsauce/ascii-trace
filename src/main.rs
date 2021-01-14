extern crate nalgebra_glm as glm;
extern crate pancurses;

use glm::{vec3, vec2, vec4, TVec3, TVec2, Mat4x4, mat4x4, rotate_x, rotate_y, rotate_z, identity, U4};
use pancurses::{endwin, initscr, noecho, Input, resize_term};
use rand::{thread_rng, Rng};
use std::rc::Rc;
use std::cell::{Ref, RefCell};
use std::cmp::Ordering;
use std::sync::{Arc, RwLock};

mod matrixes;
// use crate::matrixes;

struct Scene {
  objects: Vec<Rc<RefCell<dyn Renderable>>>
}

struct Material {
  albedo: f32,
}

struct Sphere {
  material: Material,
  center: TVec3<f32>,
  radius: f32,
}

struct Plane {
  material: Material,
  center: TVec3<f32>,
  size: TVec2<f32>,
}

trait Renderable {
  fn ray_intersect(&self, source: &TVec3<f32>, dir: &TVec3<f32>) -> Option<f32>;
  fn get_normal(&self, hit: &TVec3<f32>) -> TVec3<f32>;
  fn material(&self) -> &Material;
}

impl Renderable for Sphere {
  fn ray_intersect(&self, source: &TVec3<f32>, dir: &TVec3<f32>) -> Option<f32> {
    let l = &self.center - source;
    let tca = glm::dot(&l, dir);
    let d2 = glm::magnitude2(&l) - tca * tca;
    if d2 > self.radius * self.radius {
      return None;
    }
    let thc = (self.radius * self.radius - d2).sqrt();
    let mut t0 = tca - thc;
    let t1 = tca + thc;
    if t0 < 0_f32 {
      t0 = t1;
    }
    if t0 < 0_f32 {
      return None;
    }
    return Some(t0);
  }
  
  fn get_normal(&self, hit: &TVec3<f32>) -> TVec3<f32> {
    glm::normalize(&(hit - self.center))
  }

  fn material(&self) -> &Material {
    return &self.material;
  }
}

impl Renderable for Plane {
  
  fn ray_intersect(&self, source: &TVec3<f32>, dir: &TVec3<f32>) -> Option<f32> {
    if dir.y.abs() > 1e-3 {
      let d = (-source.y - self.center.y)/dir.y;
      let pt = source + dir*d;
      if d > 0. && pt.x.abs() < 10. && pt.z < -1. && pt.z > -30. {
        return Some(d);
      }
    }
    None
  }

  fn get_normal(&self, hit: &TVec3<f32>) -> TVec3<f32> {
    vec3(0., 1., 0.)
  }

  fn material(&self) -> &Material {
    return &self.material;
  }
}

const LUT: &[u8] = " .,-~:;=!*#$@".as_bytes();

struct IntersectResult {
  dist: f32,
  hit: TVec3<f32>,
  normal: TVec3<f32>,
  obj: Rc<RefCell<dyn Renderable>>,
}

impl Ord for IntersectResult {
  fn cmp(&self, other: &Self) -> Ordering {
    if self.dist < other.dist {
      Ordering::Less
    } else if self.dist > other.dist {
      Ordering::Greater
    } else {
      Ordering::Equal
    }
  }
}

impl Eq for IntersectResult {}

impl PartialOrd for IntersectResult {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl PartialEq for IntersectResult {
  fn eq(&self, other: &Self) -> bool {
    self.dist == other.dist
}
}


fn scene_intersect<'a>(source: &TVec3<f32>, dir: &TVec3<f32>, scene: Arc<RwLock<Scene>>) -> Option<IntersectResult> {
  let mut min_dist = f32::MAX;

  (&scene.read().unwrap().objects).into_iter().map(|obj| {
    return match obj.borrow().ray_intersect(source, dir) {
      Some(dist) => {
        if dist >= min_dist {
          return None;
        }
        min_dist = dist;
        let hit = source + dir*dist;
        return Some(IntersectResult {
          dist: dist,
          hit: hit,
          normal: obj.clone().borrow().get_normal(&hit),
          obj: obj.clone(),
        });
        //dot: f32 = glm::dot(&normal, &vec3(1., 1., 1.));
      },
      None => None,
    }
  }).filter(|x| x.is_some()).min_by(|a,b| {
    a.as_ref().unwrap().cmp(&b.as_ref().unwrap())
  }).unwrap_or_default()
}

fn cast_ray(source: &TVec3<f32>, dir: &TVec3<f32>, scene: Arc<RwLock<Scene>>) -> f32 {

  match scene_intersect(source, dir, scene.clone()) {
    Some(result) => {
      let light_dir = vec3(1., 1., 1.);
      let dot: f32 = glm::dot(&result.normal, &light_dir);
      0.1 + dot.max(0_f32) * &result.obj.borrow().material().albedo 
      // * match scene_intersect(&(result.hit + result.normal * 1e-3), &light_dir, scene.clone()) {
      //   Some(_) => 0.,
      //   None => 1.,
      // }
    },
    None => 0.,
  }
  //1.
}

// currently using view_angles also for target_pos (cos im lazy)
fn render(win: &pancurses::Window, scene: Arc<RwLock<Scene>>, view_pos: &TVec3<f32>, view_angles: &TVec3<f32>) {
  let size = win.get_max_yx();
  let w = size.1 as f32;
  let h = size.0 as f32;
  win.mv(0, 0);
  let fov: f32 = std::f32::consts::PI/3.;
  for j in 0..size.0 {
    for i in 0..size.1 {
      // let x: f32 = (2.* (i as f32 + 0.5) / (size.1 as f32 - 1.))*(fov / 2. ).tan() * size.1 as f32/size.0 as f32;
      // let y: f32 = -(2.* (j as f32 + 0.5) / (size.0 as f32 - 1.))*(fov / 2. ).tan();
      // let dir: TVec3<f32> = glm::normalize(&vec3(x,y,-1.));
      
      // let forward = vec3(eulerA.y.sin() * eulerA.x.cos(), eulerA.x.cos(), eulerA.y.cos() * eulerA.x.cos());
      // let up      = vec3(eulerA.y.sin() * eulerA.x.cos(), eulerA.x.sin(), eulerA.y.cos() * eulerA.x.cos() );
      // //let up = vec3(0.,1.,0.);
      // let right = glm::cross(&up, &forward);
      let px = i as f32;
      let py = j as f32;

      // normalized device coords (screenspace [0,1] both axes)
      let ndcx = (px + 0.5) / w;
      let ndcy = (py + 0.5) / h;

      // screen space coords ([-1,1] both axes, origin at center)
      let fov_fac = (fov/2.).tan();
      let ssx = (2. * ndcx - 1.) * (w/h) * fov_fac;
      // inverted to flip y axis
      let ssy = (1. - 2. * ndcy) * fov_fac * 2.; // * 2, as 1 character is twice as tall as it is wide (roughly)

      // let c2w = matrixes::fps_matrix(view_pos, &view_angles.xy());

      let c2w = matrixes::look_at_matrix(view_pos, view_angles);

      //let c2w =  rotate_x(&rotate_y(&identity(), eulerA.y), eulerA.x);

      // let dir_x =  ((i as f32 + 0.5) - w/2.);
      // let dir_y = (-((j as f32 + 0.5) * 2.) + h/2.);
      // let dir_z = (-h / (2.* (fov/2.).tan() ));

      //let dir = forward * dir_z + right * dir_x + up * dir_y;

      let mut lum = cast_ray(view_pos, &glm::normalize(&(c2w * vec4(ssx, ssy, -1., 0.)).xyz()), scene.clone());
      if lum < 0. {
        lum = 0.;
      }
      if lum > 1. {
        lum = 1.;
      }
      //lum = (i % 2) as f32;
      win.addch(LUT[(lum * 12.) as usize] as char);
    }
  }
}

fn main() {
  let window = initscr();
  window.printw("Hello Rust");
  window.refresh();
  window.nodelay(true);
  resize_term(20, 50);
  noecho();
  let scene = Arc::new(RwLock::new(Scene {
    objects: Vec::new()
  }));
  // scene.write().unwrap().objects.push(Rc::new(RefCell::new(Plane {
  //   center: vec3(0.,50.,0.),
  //   size: vec2(0.,0.),
  //   material: Material {
  //     albedo: 0.5,
  //   }
  // })));
  //let mut balls: Vec<Box<Sphere>> = Vec::new();
  let mut rng = thread_rng();
  for _ in 0..3 {
    scene.write().unwrap().objects.push(Rc::new(RefCell::new(Sphere {
      center: vec3(rng.gen_range(-20_f32..20_f32), rng.gen_range(-20_f32..20_f32), rng.gen_range(-20_f32..20_f32)),
      radius: rng.gen_range(2_f32..4_f32),
      material: Material {
        albedo: 1.,
      }
    })));
  }
  let sphere = Rc::new(RefCell::new(Sphere {
    center: vec3(0., 0., 0.),
    radius: 4.,
    material: Material {
      albedo: 1.,
    }
  }));
  scene.write().unwrap().objects.push(sphere.clone());
  //let sphere = (scene.objects.last().unwrap());//.downcast::<Sphere>();
  //let mut s = sphere.as_mut();
  //let mut s = Sphere {center: vec3(0., 0., -16.), radius: 4.};
  let mut time: f32 = 0.;
  let radius:f32 = 10.;
  loop {
    render(&window, Arc::clone(&scene), &vec3(time.sin()*radius,1.,time.cos()*radius), &vec3(0.,0.,0.));
    //sphere.borrow_mut().center = vec3(time.sin()*3., (time*5.).cos()*3.-5., -16.);
    time += 0.05;
    match window.getch() {
      Some(Input::KeyDC) => break,
      Some(_) => (),
      None => (),
    }
  }
  endwin();
}
