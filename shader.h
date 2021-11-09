#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>


struct Shader {
        Shader(const char *vert_file_name, const char *frag_file_name);
        void use() {
                glUseProgram(shaderProgram);
        }

        GLuint shaderProgram;
};

struct ColorShader : Shader {
        ColorShader() : Shader("shaders/color.vert", "shaders/color.frag") {
                gWorldLocation = glGetUniformLocation(shaderProgram, "gWorld");
                if (gWorldLocation == -1) {
                        throw "*** couldn't find gWorld in ColorShader\n";
                }
        }

        GLint gWorldLocation;
};

struct TextShader : Shader {
        TextShader() : Shader("shaders/text.vert", "shaders/text.frag")
        {
                sampler_location = glGetUniformLocation(shaderProgram, "sampler");
                if (sampler_location == -1) {
                        throw "*** couldn't find sampler in TextShader\n";
                }

                fg_color_location = glGetUniformLocation(shaderProgram, "fg_color");
                if (fg_color_location == -1) {
                        throw "*** couldn't find fg_color in TextShader\n";
                }

                // bk_color_location = glGetUniformLocation(shaderProgram, "bk_color");
                // if (bk_color_location == -1) {
                //         throw "*** couldn't find bk_color in TextShader\n";
                // }

                world_location = glGetUniformLocation(shaderProgram, "world");
                if (world_location == -1) {
                        throw "*** couldn't find world in TextShader\n";
                }
        }

        void use(int which_unit) {
                glUseProgram(shaderProgram);
                glUniform1i(sampler_location, which_unit);
        }

        void set_fg_color(float r, float g, float b) {
                glUniform4f(fg_color_location, r, g, b, 1.0);
        }

        GLint fg_color_location;
        // GLint bk_color_location;
        GLint world_location;
private:
        GLint sampler_location;
};

struct TextureShader : Shader {
        TextureShader() : Shader("shaders/texture.vert", "shaders/texture.frag")
        {
                sampler_location = glGetUniformLocation(shaderProgram, "sampler");
                if (sampler_location == -1) {
                        throw "*** couldn't find sampler in TextureShader\n";
                }

                world_location = glGetUniformLocation(shaderProgram, "world");
                if (world_location == -1) {
                        throw "*** couldn't find world in TextureShader\n";
                }
        }

        void use(int which_unit) {
                glUseProgram(shaderProgram);
                glUniform1i(sampler_location, which_unit);
        }

        GLint world_location;
private:
        GLint sampler_location;
};
