extern crate nalgebra_glm as glm;
use glm::{vec3, vec2, vec4, TVec3, TVec2, TMat4, Mat4x4, mat4x4, rotate_x, rotate_y, rotate_z, identity, U4};

pub fn fps_matrix(view_pos: &TVec3<f32>, view_angles: &TVec2<f32>) -> TMat4<f32>{
    let pitch = view_angles.x;
    let yaw   = view_angles.y;
    
    let cos_pitch = pitch.cos();
    let sin_pitch = pitch.sin();
    let cos_yaw   = yaw.cos();
    let sin_yaw   = yaw.sin();

    // im too stupid for this
    let xaxis = vec3(cos_yaw           , 0.       , -sin_yaw          );
    let yaxis = vec3(sin_yaw * sin_pitch,  cos_pitch, cos_yaw * sin_pitch);
    let zaxis = vec3(sin_yaw * cos_pitch, -sin_pitch, cos_pitch * cos_yaw);

    return mat4x4(
        xaxis.x, yaxis.x, zaxis.x, 0.,
        xaxis.y, yaxis.y, zaxis.y, 0.,
        xaxis.z, yaxis.z, zaxis.z, 0.,
        -glm::dot(&xaxis, view_pos), -glm::dot(&yaxis, view_pos), -glm::dot(&zaxis, view_pos), 1.
    );
}

pub fn look_at_matrix(view_pos: &TVec3<f32>, target_pos: &TVec3<f32>) -> TMat4<f32>{
    let up = glm::vec3(0.,1.,0.);
    let zaxis = glm::normalize(&(view_pos - target_pos));    // The "forward" vector.
    let xaxis = glm::normalize(&glm::cross(&up, &zaxis));// The "right" vector.
    let yaxis = glm::cross(&zaxis, &xaxis);     // The "up" vector.
 
    return mat4x4(
        xaxis.x, yaxis.x, zaxis.x, 0.,
        xaxis.y, yaxis.y, zaxis.y, 0.,
        xaxis.z, yaxis.z, zaxis.z, 0.,
        -glm::dot( &xaxis, &target_pos ), -glm::dot( &yaxis, &target_pos ), -glm::dot( &zaxis, &target_pos ), 1.
    );
}