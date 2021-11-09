#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
#include <chrono>

#include "shader.h"



inline auto now() noexcept { return std::chrono::high_resolution_clock::now(); }



struct circle_obj {
        circle_obj(int num_edges) {
                num_points = num_edges + 2;     // a fan has 2 more points than edges
                // const uint num_floats =  num_points * 2;      // 2d, so two floats per point
                // const uint mem_size = num_floats * sizeof(GLfloat);      // allocation size

                std::vector<GLfloat> data;
                // start the circle_fan in the center of the circle
                data.push_back(0);
                data.push_back(0);
                // loop over the requested number of circle edge points
                for (int i=0; i<num_edges; ++i) {
                        data.push_back(cosf(2 * M_PI * i/(float)num_edges));
                        data.push_back(sinf(2 * M_PI * i/(float)num_edges));
                }
                // finish the circle_fan with our starting edge point
                data.push_back(1);
                data.push_back(0);

                glGenVertexArrays(1, &VAO);
                glBindVertexArray(VAO);

                // pointer to circle triangle_fan coords for attribute 0
                glGenBuffers(1, &VBO);  
                glBindBuffer(GL_ARRAY_BUFFER, VBO);
                glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
                glEnableVertexAttribArray(0);
                glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(GLfloat), data.data(), GL_STATIC_DRAW);
                glBindBuffer(GL_ARRAY_BUFFER, 0);

                glBindVertexArray(0);
        }


        GLuint VAO;
        GLuint VBO;
        GLsizei num_points;
};

extern circle_obj *circle_32;



extern GLuint VAO, COLOR_BO, EBO;
extern GLuint square_vao, texture_coord_vbo;
extern GLuint line_vao, line_vbo;

extern ColorShader *color_shader;
extern TextShader *text_shader;
extern TextureShader *texture_shader;

extern float window_ratio;

extern float width_unit_per_pixel;
extern float height_unit_per_pixel;
