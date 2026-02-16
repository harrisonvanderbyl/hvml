
#include "vector/mat4.hpp"

struct Camera {
    float32x3 position;
    float32x3 forward;
    float32x3 up;
    float fov;
    float aspect;
    float near_plane;
    float far_plane;
    
    Camera() : position(0, 0, 5), forward(0, 0, 0), up(0, 1, 0), 
               fov(45.0f), aspect(1.0f), near_plane(0.1f), far_plane(1000.0f) {}

 

    
     mat4 getProjectionMatrix() {
        mat4 result;
        float tanHalfFov = tan(fov * 0.5f * M_PI / 180.0f);
        
        result[0] = float32x4(1.0f / (aspect * tanHalfFov), 0, 0, 0);
        result[1] = float32x4(0, 1.0f / tanHalfFov, 0, 0);
        result[2] = float32x4(0, 0, -(far_plane + near_plane) / (far_plane - near_plane), -1);
        result[3] = float32x4(0, 0, -(2.0f * far_plane * near_plane) / (far_plane - near_plane), 0);
        
        return result;
    }
    
    mat4 getViewMatrix() {
        // Simple lookAt implementation
        float32x3 f = forward; // forward
        
        // Normalize f
        f = f.normalized();
        
         // Normalize up
        
        // Right = f x up
        float32x3 r = f.cross(up.normalized());
        
        // Normalize r
        
        // Up = r x f
        float32x3 u = r.cross(f);
        
        mat4 result;
        result[0] = float32x4(r[0], u[0], -f[0], 0);
        result[1] = float32x4(r[1], u[1], -f[1], 0);
        result[2] = float32x4(r[2], u[2], -f[2], 0);
        result[3] = float32x4(
            -(r.x() * position.x() + r.y() * position.y() + r.z() * position.z()),
            -(u.x() * position.x() + u.y() * position.y() + u.z() * position.z()),
            f.x() * position.x() + f.y() * position.y() + f.z() * position.z(),
            1
        );
        
        return result;
    }


    void bind(GLuint shader_program) {
        // Set camera uniforms
        getViewMatrix().bind(shader_program, "view");
        getProjectionMatrix().bind(shader_program, "projection");
        (getViewMatrix() * getProjectionMatrix() ).bind(shader_program, "projectionview");
        getViewMatrix().inverse().bind(shader_program, "inv_view");
        getProjectionMatrix().inverse().bind(shader_program, "inv_projection");
    }
        
};
