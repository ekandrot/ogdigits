#version 330 core
        
out vec4 FragColor;
in vec2 texture_coord;

uniform sampler2D sampler;
uniform vec4 fg_color;
uniform vec4 bk_color;

void main()
{
        vec4 color = texture2D(sampler, texture_coord.st);
        FragColor = fg_color * color.a;
}
