#version 330 core

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
in vec3 in_tangent;

uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
}