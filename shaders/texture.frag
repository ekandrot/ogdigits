#version 330 core
        
out vec4 FragColor;
in vec2 texture_coord;

uniform sampler2D sampler;

void main()
{
        FragColor = texture2D(sampler, texture_coord.st);
}
