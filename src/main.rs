extern crate nalgebra_glm as glm;
extern crate pancurses;

use glm::{vec3, vec2, vec4, TVec3, TVec2, Mat4x4, mat4x4, rotate_x, rotate_y, rotate_z, identity, U4};
use pancurses::{endwin, initscr, noecho, Input, resize_term};
use rand::{thread_rng, Rng};
use std::rc::Rc;
use std::cell::{Ref, RefCell};
use std::cmp::Ordering;
use std::sync::{Arc, RwLock, Mutex};

mod matrixes;
// use crate::matrixes;

struct Scene {
  objects: Vec<Arc<Mutex<dyn Renderable + Send>>>,
  fov: f32,
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

unsafe impl Send for Sphere {

}

unsafe impl Send for Plane {

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
  
  // yoinked from https://stackoverflow.com/questions/5666222/3d-line-plane-intersection/18543221#18543221
  fn ray_intersect(&self, source: &TVec3<f32>, dir: &TVec3<f32>) -> Option<f32> {
    let normal = vec3(0., 1., 0.);
    let dot = glm::dot(&normal, dir);
    if dot.abs() > 1e-3 {
      let w = source - self.center;
      let d = -glm::dot(&normal, &w) / dot;
      let pt = (source + (dir * d)) + self.center;
      if d > 0. && pt.x.abs() < self.size.x && pt.z.abs() < self.size.y {
        return Some(d);
      }
    }
    None
  }

  fn get_normal(&self, _hit: &TVec3<f32>) -> TVec3<f32> {
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
  obj: Arc<Mutex<dyn Renderable >>,
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

impl PartialEq for IntersectResult{
  fn eq(&self, other: &Self) -> bool {
    self.dist == other.dist
}
}


fn scene_intersect(source: &TVec3<f32>, dir: &TVec3<f32>, scene: Arc<RwLock<Scene>>) -> Option<IntersectResult> {
  let mut min_dist = f32::MAX;

  (&scene.read().unwrap().objects).into_iter().map(|obj| {
    let info = obj.lock().unwrap().ray_intersect(source, dir);
    return match info {
      Some(dist) => {
        if dist >= min_dist {
          return None;
        }
        min_dist = dist;
        let hit = source + dir * dist;
        // let normal = obj.lock().unwrap().get_normal(&hit);
        return Some(IntersectResult {
          dist: dist,
          hit: hit,
          normal: obj.lock().unwrap().get_normal(&hit),
          obj: obj.clone()
        });
      },
      None => None,
    }
    // return Some(IntersectResult {
    //   dist: 1.,
    //   hit: vec3(0.0,0.0,0.0),
    //   normal: vec3(0.0,1.0,0.0),
    //   obj: obj.clone()
    // });
    // return match obj.lock().unwrap().ray_intersect(source, dir) {
    //   Some(dist) => {
    //     if dist >= min_dist {
    //       return None;
    //     }
    //     min_dist = dist;
    //     let hit = source + dir*dist;
    //     return Some(IntersectResult {
    //       dist: dist,
    //       hit: hit,
    //       normal: obj.clone().lock().unwrap().get_normal(&hit),
    //       obj: obj.clone(),
    //     });
    //     //dot: f32 = glm::dot(&normal, &vec3(1., 1., 1.));
    //   },
    //   None => None,
    // }
  }).filter(|x| x.is_some()).min_by(|a,b| {
    a.as_ref().unwrap().cmp(&b.as_ref().unwrap())
  }).unwrap_or_default()
}

fn cast_ray(source: &TVec3<f32>, dir: &TVec3<f32>, scene: Arc<RwLock<Scene>>) -> f32 {

  let camera_hit = scene_intersect(source, dir, scene.clone());
  if camera_hit.is_none() {
    return 0.
  }
  let camera_hit = camera_hit.unwrap();

  let light_dir = glm::normalize(&vec3(1., 1., 1.));
  let dot: f32 = glm::dot(&camera_hit.normal, &light_dir);

  let biased_point = camera_hit.hit + camera_hit.normal * 0.001;
  let shadow_hit = scene_intersect(&biased_point, &light_dir, scene.clone());
  let shadow_val = match shadow_hit {
    Some(_) => 0.,
    None => 1.
  };
  let albedo = camera_hit.obj.lock().unwrap().material().albedo;
  0.1 + dot.max(0_f32) * &albedo.clone() * shadow_val
  // match scene_intersect(source, dir, scene.clone()) {
  //   Some(result) => {
  //     let light_dir = glm::normalize(&vec3(1., 1., 1.));
  //     let dot: f32 = glm::dot(&result.normal, &light_dir);
  //     // depth render
  //     // 1.-(result.dist/50.)

  //     // normal render
  //     // 0.1
  //     0.1 + dot.max(0_f32) * &result.obj.lock().unwrap().material().albedo 
  //     * match scene_intersect(&(result.hit + result.normal * 0.001), &light_dir, scene.clone()) {
  //       Some(_) => 0.,
  //       None => 1.,
  //     }
  //   },
  //   None => 0.,
  // }
}


// currently using view_angles also for target_pos (cos im lazy)
fn render(win: &pancurses::Window, scene: Arc<RwLock<Scene>>, view_pos: &TVec3<f32>, view_angles: &TVec3<f32>, rt: &tokio::runtime::Runtime) {
  let size = win.get_max_yx();
  // let fov: f32 = std::f32::consts::PI/1.2; // ortho fov

  let mut handles = Vec::<tokio::task::JoinHandle<(i32,i32,char)>>::with_capacity((size.1 * size.0) as usize);

  let out = rt.block_on(async {
    for j in 0..size.0 {
      for i in 0..size.1 {
        let w = size.1 as f32;
        let h = size.0 as f32;

        let scene = scene.clone();

        let view_pos = view_pos.clone();
        let view_angles = view_angles.clone();
        handles.push(tokio::spawn(async move {
          let px = i as f32;
          let py = j as f32;
  
          // normalized device coords (screenspace [0,1] both axes)
          let ndcx = (px + 0.5) / w;
          let ndcy = (py + 0.5) / h;
  
          // screen space coords ([-1,1] both axes, origin at center)
          let fov_fac = (scene.read().unwrap().fov/2.).tan();
          let ssx = (2. * ndcx - 1.) * (w/h) * fov_fac;
          // inverted to flip y axis
          let ssy = (1. - 2. * ndcy) * fov_fac * 2.; // * 2, as 1 character is twice as tall as it is wide (roughly)
  
          let c2w = matrixes::look_at_matrix(&view_pos, &view_angles);
  
          // // normal perspective rays
          let mut lum = cast_ray(&view_pos, &glm::normalize(&(c2w * vec4(ssx, ssy, -1., 0.)).xyz()), scene);
          // let mut lum = cast_ray(&view_pos, &glm::normalize(&vec3(0.,0.,-1.0)), scene);
  
          // let mut lum = 1.;
          if lum < 0. {
            lum = 0.;
          }
          if lum > 1. {
            lum = 1.;
          }
          // return 'a';
          // return (j,i, LUT[(j as usize % 12)] as char);
          return (j, i, LUT[(lum * 12.) as usize] as char);
        }));
        
      }
    }
    futures::future::join_all(handles).await
  });
  

  out.into_iter().for_each(|res| {
    let res = res.unwrap();
    // win.mvaddch(res.0,res.1,res.0.to_string().chars().last().unwrap());
    win.mvaddch(res.0,res.1,res.2);
  });

  // (0..(size.0*size.1)).into_par_iter().map(|idx| {
  //   let i = idx / size.1;
  //   let j = idx % size.1;
  //   let px = i as f32;
  //   let py = j as f32;

  //   // normalized device coords (screenspace [0,1] both axes)
  //   let ndcx = (px + 0.5) / w;
  //   let ndcy = (py + 0.5) / h;

  //   // screen space coords ([-1,1] both axes, origin at center)
  //   let fov_fac = (fov/2.).tan();
  //   let ssx = (2. * ndcx - 1.) * (w/h) * fov_fac;
  //   // inverted to flip y axis
  //   let ssy = (1. - 2. * ndcy) * fov_fac * 2.; // * 2, as 1 character is twice as tall as it is wide (roughly)

  //   // let c2w = matrixes::fps_matrix(view_pos, &view_angles.xy());

  //   let c2w = matrixes::look_at_matrix(view_pos, view_angles);

  //   //let c2w =  rotate_x(&rotate_y(&identity(), eulerA.y), eulerA.x);

  //   // let dir_x =  ((i as f32 + 0.5) - w/2.);
  //   // let dir_y = (-((j as f32 + 0.5) * 2.) + h/2.);
  //   // let dir_z = (-h / (2.* (fov/2.).tan() ));

  //   //let dir = forward * dir_z + right * dir_x + up * dir_y;

  //   // orthographic rendering (only works with look_at matrix)
  //   // let ray_start = view_pos + (c2w * vec4(ssx, ssy, 0., 0.)).xyz();
  //   // let ray_dir = glm::normalize(&(view_angles - view_pos));
  //   // let mut lum = cast_ray(&ray_start, &ray_dir, scene.clone());

  //   // normal perspective rays
  //   let mut lum = cast_ray(view_pos, &glm::normalize(&(c2w * vec4(ssx, ssy, -1., 0.)).xyz()), scene.clone());


  //   if lum < 0. {
  //     lum = 0.;
  //   }
  //   if lum > 1. {
  //     lum = 1.;
  //   }
  //   //lum = (i % 2) as f32;
  //   //win.addch(LUT[(lum * 12.) as usize] as char);
  // });
}

fn main() {
  let window = initscr();
  window.printw("Hello Rust");
  window.refresh();
  window.nodelay(true);
  resize_term(40, 100);
  noecho();
  let scene = Arc::new(RwLock::new(Scene {
    objects: Vec::new(),
    fov: std::f32::consts::PI/3.
  }));
  scene.write().unwrap().objects.push(Arc::new(Mutex::new(Plane {
    center: vec3(0.0, -5.0, 0.0),
    size: vec2(100.0, 100.0),
    material: Material {
      albedo: 0.5,
    }
  })));
  //let mut balls: Vec<Box<Sphere>> = Vec::new();
  let mut rng = thread_rng();
  for _ in 0..3 {
    scene.write().unwrap().objects.push(Arc::new(Mutex::new(Sphere {
      center: vec3(rng.gen_range(-10_f32..10_f32), rng.gen_range(-5_f32..5_f32), rng.gen_range(-10_f32..10_f32)),
      radius: rng.gen_range(2_f32..4_f32),
      material: Material {
        albedo: 1.,
      }
    })));
  }
  let sphere = Arc::new(Mutex::new(Sphere {
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
  let radius:f32 = 12.;
  let mut rt = tokio::runtime::Runtime::new().unwrap();
  loop {

    render(&window, Arc::clone(&scene), &vec3(time.sin()*radius,3.,time.cos()*radius), &vec3(0.,0.,0.), &rt);
    // render(&window, Arc::clone(&scene), &vec3(10.0, 10.0, 10.0), &vec3(0., 0., 0.));

    // render(&window, Arc::clone(&scene), &vec3(7.0, 7.0, 10.0), &vec3(0., 0., 0.));


    // sphere.borrow_mut().center = vec3(0.0,0.0,time.sin());
    // sphere.borrow_mut().radius = (3. + (time*2.).sin()) * 2.0;
    time += 0.04;
    match window.getch() {
      Some(Input::KeyDC) => break,
      Some(_) => (),
      None => (),
    }
  }
  endwin();
}
