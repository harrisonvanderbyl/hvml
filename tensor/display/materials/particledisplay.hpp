#ifndef TENSOR_DISPLAY_MATERIALS_PARTICLE_HPP
#define TENSOR_DISPLAY_MATERIALS_PARTICLE_HPP

#include "display/materials/shader.hpp"

struct __shader ParticleShader : public ShaderProgram
{
    mat4 model;
    mat4 view;
    mat4 projection;

    // sampler2D texture1;

    // === Vertex → Fragment interpolants ===
    

    void vertex(
        float32x3 position,
        float32x2 temperatureopacity,
        uint84 color
    ){
       
        // Early kill invisible particles
        if (temperatureopacity.y() < 0.5f) {
            gl_PointSize = 0.0f;
            gl_Position  = float32x4(0.0f);
            return;
        }

        float32x4 worldPos = model * float32x4(floor(position), 1.0);
        float32x4 worldposnature = model * float32x4(position, 1.0);


        // Sphere radius in world space (Y axis scale)
        float radius = color.w() > 0.8f ? 1.0f : 2.0f;

        float32x4 eyePos = view * worldposnature;
        float distance   = max(length(eyePos.xyz()), 0.0001f);

        // Perspective-correct point size
        gl_PointSize = (radius * 2048.0f * 3.0f) / distance;
        gl_Position  = projection * eyePos;

        float32x4 camerapos = inverse(view) * float32x4(0,0,0,1);

        fragment(
            radius,
            worldposnature,
            camerapos,
            color
        );
    }

    void fragment(
        float vRadius,
        float32x4 centerpos,
        float32x4 camPos,
        uint84 vColor
    ){
        // Convert gl_FragCoord from window coordinates to NDC
        // gl_FragCoord.xy is in pixels, gl_FragCoord.z is depth [0,1]
        float32x2 screenSize = float32x2(1024.0f, 1024.0f); // Use actual viewport size
        
        // Normalized Device Coordinates (NDC): [-1, 1] range
        float32x2 ndc_xy = (gl_FragCoord.xy() / screenSize) * 2.0f - 1.0f;
        float ndc_z = gl_FragCoord.z() * 2.0f - 1.0f; // Convert [0,1] to [-1,1]
        
        // Clip space position (before perspective divide)
        float32x4 clipPos = float32x4(ndc_xy, ndc_z, 1.0f);
        
        // View space position
        float32x4 viewSpacePos = inverse(projection) * clipPos;
        viewSpacePos /= viewSpacePos.w(); // Perspective divide
        
        // World space position
        float32x4 worldSpacePos = inverse(view) * viewSpacePos;
        
        float32x3 raycastDir = normalize(worldSpacePos.xyz() - camPos.xyz());

            // Ray-sphere intersection test
        float32x3 oc = camPos.xyz() - centerpos.xyz();
        float a = dot(raycastDir,raycastDir);
        float b = 2.0f * dot(oc,raycastDir);
        float c = dot(oc,oc) - vRadius * vRadius;
        float discriminant = b * b - 4 * a * c;
        if (discriminant < 0) {
            discard(); // No intersection, discard the fragment
        } else {

            float t = (-b - sqrt(discriminant)) / (2.0f * a);
            float32x3 intersectionPoint = camPos.xyz() + float32x3(t,t,t) * raycastDir;
            float32x4 viewSpaceIntersection = view * float32x4(intersectionPoint, 1.0f);
            float32x4 clipSpaceIntersection = projection * viewSpaceIntersection;
            gl_FragDepth = clipSpaceIntersection.z() / clipSpaceIntersection.w();

            if(vColor.w() < 0.5f){
                // through water as though its a sphere
                float distthroughwater = length(intersectionPoint - worldSpacePos.xyz())/(vRadius*2.0f);
                FragColor = float32x4(vColor.x(), vColor.y(), vColor.z(), distthroughwater);
                return;
            }
            // Optional: Calculate the intersection point and normal for lighting
            float32x3 normal = normalize(intersectionPoint - centerpos.xyz());
            // You can use the normal for lighting calculations if desired
            // , global light from directly above
                float lightIntensity = max(dot(normal, float32x3(0, 1, 0)), 0.0f);
                uint84 baseColor = vColor;
                uint84 finalColor = uint84(baseColor.xyz() * lightIntensity, baseColor.w());
                FragColor = finalColor;

                // calc fragment depth based on intersection point
                
        }
        // float distancefromcenter = length(worldSpacePos.xyz() - centerpos.xyz());
        // if (distancefromcenter > 1.0){
        //     discard();
        // }

        
        // FragColor = float32x4(worldSpacePos.xyz() / 1000.0, vColor.w());
    }
};

// __weak ShaderSourceOb global_optimized_sphere_point_material = Shader<ParticleShader>();

#endif // TENSOR_DISPLAY_MATERIALS_PARTICLE_HPP
