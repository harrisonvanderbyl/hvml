#ifndef TENSOR_DISPLAY_MATERIALS_PARTICLE_HPP
#define TENSOR_DISPLAY_MATERIALS_PARTICLE_HPP

#include "display/materials/shader.hpp"

struct __shader ParticleShader : public ShaderProgram
{
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 projectionview;
    mat4 inv_view;
    mat4 inv_projection;
    float depthOnly;

    sampler2D texture1;
    sampler2D screentexture;

    // === Vertex → Fragment interpolants ===
    

    void vertex(
        float32x3 position,
        float32x2 temperatureopacity,
        float neighborsfilled,
        uint84 color
    ){
        // Early kill invisible particles
        if (neighborsfilled < 0.5f) {
            gl_PointSize = 0.0f;
            gl_Position  = float32x4(0.0f);
            return;
        }
        
        float32x4 worldPos = model * float32x4(floor(position), 1.0);
        float32x4 worldposnature = model * float32x4(position, 1.0);
        
        // Sphere radius in world space
        float radiusgrow = 0.25f;
        float radius = 0.0f;
        if(color.w() < 0.8f){
            radius = 0.0f;
            radiusgrow = 0.5f;
        }
        
        radius += float(neighborsfilled) * radiusgrow; 
        radius = max(radius, 0.0f);
        
        float32x4 eyePos = view * worldposnature;
        gl_Position = projection * eyePos;
        
        // Correct perspective-aware point size calculation
        // Get the distance from camera to particle in view space
        float distance = length(eyePos.xyz());
        
        // Extract focal length from projection matrix
        // For perspective projection, projection[1][1] = 1/tan(fov/2)
        float focalLength = projection[1][1];
        
        // Screen height (use actual viewport height, e.g., 1024.0)
        float screenHeight = 1024.0f;
        
        // Calculate point size: (radius * focalLength * screenHeight) / distance
        // Multiply by 2 because radius is half the diameter
        gl_PointSize = (radius * focalLength * screenHeight) / distance;

        float32x4 camPos = float32x4(inv_view[3][0], inv_view[3][1], inv_view[3][2], 1.0);
        
        fragment(
            radius,
            worldposnature,
            camPos,
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
        float32x2 partuv = gl_PointCoord - 0.5f;
        float distSq = dot(partuv, partuv); 
        if(distSq > 0.25f){ // Outside the circle (radius 0.5 in point sprite space)
            discard(); // Kill fragments outside the circle
            return;
        }

        if(depthOnly > 0.5f){
            // For depth-only pass, output a simple color and depth
            FragColor = uint84(0, 0, 0, 255); // Dummy output
            gl_FragDepth = gl_FragCoord.z(); // Simple depth
            return;
        }
        // if(length(partuv - 0.5f) > 0.49f){
        //     // Show edge of circle with a thin outline

        //     FragColor = uint84(vColor.xyz(), 1.0f);
        //     gl_FragDepth = gl_FragCoord.z(); // Use the original depth for the inner part
        //     return;
        // }

        float32x2 screenSize = float32x2(1024.0f, 1024.0f); // Use actual viewport size
        
        // Normalized Device Coordinates (NDC): [-1, 1] range
        float32x2 ndcxybn = (gl_FragCoord.xy() / screenSize);
        float32x2 ndc_xy = ndcxybn * 2.0f - 1.0f;
        float ndc_z = gl_FragCoord.z() * 2.0f - 1.0f; // Convert [0,1] to [-1,1]
        
        // Clip space position (before perspective divide)
        float32x4 clipPos = float32x4(ndc_xy, ndc_z, 1.0f);
        
        // View space position
        float32x4 viewSpacePos = inv_projection * clipPos;
        viewSpacePos /= viewSpacePos.w(); // Perspective divide
        
        // World space position
        float32x4 worldSpacePos = inv_view * viewSpacePos;
        
        float32x3 raycastDir = normalize(worldSpacePos.xyz() - camPos.xyz());

            // Ray-sphere intersection test
        float32x3 oc = camPos.xyz() - centerpos.xyz();
        float a = dot(raycastDir,raycastDir);
        float b = 2.0f * dot(oc,raycastDir);
        float c = dot(oc,oc) - vRadius * vRadius;
        float discriminant = b * b - 4 * a * c;
        if (discriminant < 0) {
            discard(); // No intersection, discard the fragment
            return;
        } else {

            float t = (-b - sqrt(discriminant)) / (2.0f * a);
            float32x3 intersectionPoint = camPos.xyz() + float32x3(t,t,t) * raycastDir;
            float32x4 clipSpaceIntersection = projectionview * float32x4(intersectionPoint, 1.0f);
            gl_FragDepth = clipSpaceIntersection.z() / clipSpaceIntersection.w();

            float32x3 normal = normalize(intersectionPoint - centerpos.xyz());

            if(vColor.w() < 0.5f){
    // Simple screen-space distortion
                float distortionStrength = 0.01f;
                
                // Use surface normal to distort UV
                float32x2 distortion = normal.xy() * distortionStrength;
                float32x2 distortedUV = clamp(ndcxybn + distortion, float32x2(0.0f), float32x2(1.0f));
                
                // Sample background
                uint84 backgroundColor = texture2D(screentexture, distortedUV);
                
                // Fresnel for edge brightness
                float32x3 viewDir = normalize(camPos.xyz() - intersectionPoint);
                float fresnel = pow(1.0f - abs(dot(viewDir, normal)), 3.0f);
                
                // Water color with tint
                float32x3 waterTint = float32x3(0.85f, 0.92f, 1.0f);
                float32x3 finalColor = backgroundColor.xyz() * waterTint + float32x3(fresnel * 0.43f);
                
                FragColor = uint84(finalColor, 1.0f);
                return;
            }
            // Optional: Calculate the intersection point and normal for lighting
            // You can use the normal for lighting calculations if desired
            // , global light from directly above
            float lightIntensity = max(dot(normal, float32x3(0, 1, 0)), 0.0f);

            // triplanar uv mapping
            // if distance > 50, do simple planar mapping to save performance
            if (length(camPos.xyz() - intersectionPoint) > 50.0f){
                float32x2 uv = intersectionPoint.xz() * 0.1f;
                uv = float32x2(mod(uv.x(), 1.0f), mod(uv.y(), 1.0f));
                uint84 baseColor = texture2D(texture1, uv);
                uint84 finalColor = uint84(baseColor.xyz() * lightIntensity, baseColor.w() * vColor.w());
                FragColor = finalColor;
                return;
            }

            float32x3 absNormal = abs(normal);
            float32x2 uv;
            float32x2 uv1;
            float32x2 uv2;
            uv = intersectionPoint.yz() * 0.1f;
            uv1 = intersectionPoint.xz() * 0.1f;
            uv2 = intersectionPoint.xy() * 0.1f;
            // wrap uv
            uv = float32x2(mod(uv.x(), 1.0f), mod(uv.y(), 1.0f));
            uv1 = float32x2(mod(uv1.x(), 1.0f), mod(uv1.y(), 1.0f));
            uv2 = float32x2(mod(uv2.x(), 1.0f), mod(uv2.y(), 1.0f));

            // wait triplanar sphere mapping
            uint84 base1 = texture2D(texture1, uv);
            uint84 base2 = texture2D(texture1, uv1);
            uint84 base3 = texture2D(texture1, uv2);

            // Blend the three textures based on the normal's absolute components
            float total = absNormal.x() + absNormal.y() + absNormal.z();
            float32x3 weights = absNormal / total;
            uint84 baseColor = vColor * (base1 * weights.x() + base2 * weights.y() + base3 * weights.z());
            uint84 finalColor = uint84(baseColor.xyz() * lightIntensity, baseColor.w());
            FragColor = finalColor;
                
        }
    }
};

// __weak ShaderSourceOb global_optimized_sphere_point_material = Shader<ParticleShader>();

#endif // TENSOR_DISPLAY_MATERIALS_PARTICLE_HPP
