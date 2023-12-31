#version 330 core

layout (location = 0) in vec2 Pos;
layout (location = 1) in vec2 TextCoord;

uniform mat4 world; 

out vec2 texture_coord;

void main()
{
        gl_Position = world * vec4(Pos.x, Pos.y, 0.0, 1.0);
        texture_coord = TextCoord;
}
