#version 330 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;
layout (location = 3) in vec3 in_tangent; // The loader needs this!

uniform mat4 mvp;
uniform mat4 model;

out vec2 v_uv;
out vec3 v_world_pos;
out vec3 v_normal; 
out mat3 TBN;

void main()
{
    vec4 world_pos = model * vec4(in_position, 1.0);
    gl_Position = mvp * vec4(in_position, 1.0);

    v_uv = in_uv;
    v_world_pos = world_pos.xyz;
    v_normal = normalize(mat3(model) * in_normal);

    vec3 T;
    // Check if the tangent is valid (GLB will have real values, Cube will have 0,0,0)
    if (length(in_tangent) < 0.1) {
        // Fallback for Cubes: Create a fake tangent based on the normal
        T = normalize(cross(v_normal, vec3(0.0, 1.0, 0.0)));
        if (length(T) < 0.1) T = normalize(cross(v_normal, vec3(0.0, 0.0, 1.0)));
    } else {
        // Use the real tangent from the GLB
        T = normalize(mat3(model) * in_tangent);
    }

    vec3 N = v_normal;
    vec3 B = normalize(cross(N, T));
    TBN = mat3(T, B, N);
}