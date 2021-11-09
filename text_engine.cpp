#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <vector>

#include "opengl_globals.h"
#include "math_3d.h"
#include "image_loader.h"
#include "text_engine.h"

const char *chars_in_texture = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_-~`[]{}\\|;:',.<>/?";
int char_to_index[256];

static GLuint texture_obj;


//-------------------------------------------------------------------------------------------


std::vector<float> letter_boxes;


static void find_boundaries_row(uint8_t *bits, int w, int top, int bottom)
{
        int state = 0;  // 0-looking for start, 1-in letter
        int start = 0;
        bool in_letter{false};
        for (int x=0; x<w; ++x) {
                in_letter = false;
                for (int y=top; y<bottom; ++y) {
                        // look for non-zero alpha in RGBA format
                        if (bits[4*(w*y + x) + 3] != 0) {
                                in_letter = true;
                                break;
                        }
                }

                if (state == 0 && in_letter) {
                        state = 1;
                        start = x;
                } else if (state == 1 && !in_letter) {
                        state = 0;

                        letter_boxes.push_back(start);
                        letter_boxes.push_back(bottom);

                        letter_boxes.push_back(x);
                        letter_boxes.push_back(bottom);

                        letter_boxes.push_back(start);
                        letter_boxes.push_back(top);

                        letter_boxes.push_back(x);
                        letter_boxes.push_back(top);
                }
        }

        // check if letter runs into the right boudary
        if (state == 1 && in_letter) {
                letter_boxes.push_back(start);
                letter_boxes.push_back(bottom);

                letter_boxes.push_back(w);
                letter_boxes.push_back(bottom);

                letter_boxes.push_back(start);
                letter_boxes.push_back(top);

                letter_boxes.push_back(w);
                letter_boxes.push_back(top);
        }
}

static void find_boundaries_24(uint8_t *bits, int w)
{
        for (int y=0; y<220; y+=29) {
                find_boundaries_row(bits, w, y, y+29);
        }

        for (int i=0; i<letter_boxes.size(); ++i) {
                letter_boxes[i] /= (float)w;
        }

        // std::cout << "letters found:  " << letter_boxes.size()/8 << std::endl;

        for (int i=0; i<256; ++i) {
                char_to_index[i] = -1;
        }
        for (unsigned int i=0; chars_in_texture[i]!=0; ++i) {
                char_to_index[(unsigned int)chars_in_texture[i]] = i;
        }
}


// static float render_imaged_char(unsigned int which_char, float scale, float spacing, float x, float y)
// {
//         if (which_char == ' ') {
//                 spacing += 6/29.0;
//                 return spacing;
//         }

//         int index = char_to_index[which_char];
//         if (index < 0) return spacing;  // unknown char requested

//         float ratio = (letter_boxes[8*index+2] - letter_boxes[8*index]) / (29/256.0);

//         Matrix4f mat;
//         mat *= translation_matrix(x, y, 0);
//         mat *= scale_matrix(window_ratio,1,1);
//         mat *= scale_matrix(scale);
//         mat *= translation_matrix(spacing, 0, 0);
//         mat *= scale_matrix(ratio, 1, 1);
//         glUniformMatrix4fv(text_shader->world_location, 1, GL_FALSE, mat.glformat());

//         spacing += (ratio + 1/29.0);

//         // set the sample coords
//         glNamedBufferSubData(texture_coord_vbo, 0, sizeof(GLfloat)*8, letter_boxes.data()+(8*index));

//         // glEnable(GL_BLEND);
//         // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  
//         glDrawElements(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, 0);
//         // glDisable(GL_BLEND);

//         return spacing;
// }


static float render_imaged_char(unsigned int which_char, int pixels, float spacing, float x, float y)
{
        if (which_char == ' ') {
                spacing += pixels/4.0;
                return spacing;
        }

        int index = char_to_index[which_char];
        if (index < 0) return spacing;  // unknown char requested

        float ratio = (letter_boxes[8*index+2] - letter_boxes[8*index]) / (29.0 / 256);

        Matrix4f mat;
        mat *= translation_matrix(x, y, 0);
        mat *= scale_matrix(width_unit_per_pixel, height_unit_per_pixel, 1);
        mat *= translation_matrix(spacing, 0, 0);
        mat *= scale_matrix(pixels);
        mat *= scale_matrix(ratio, 1, 1);
        glUniformMatrix4fv(text_shader->world_location, 1, GL_FALSE, mat.glformat());

        spacing += ratio * pixels + 1;

        // set the sample coords
        glNamedBufferSubData(texture_coord_vbo, 0, sizeof(GLfloat)*8, letter_boxes.data()+(8*index));

        // glEnable(GL_BLEND);
        // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  
        glDrawElements(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, 0);
        // glDisable(GL_BLEND);

        return spacing;
}


//ek need to return the bounding box of the rendered text
float render_text(const std::string str, int pixels, float x, float y, TEXT_OFFSET which)
{
        if (which == PIXEL_OFFSET) {
                x *= width_unit_per_pixel;
                y *= height_unit_per_pixel;
        }

        glBindVertexArray(square_vao);
        text_shader->use(0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_obj);

        text_shader->set_fg_color(0, 0, 0);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  

        float spacing = 0;
        int i = 0;
        while (str[i] != 0) {
                spacing = render_imaged_char((unsigned int)str[i], pixels, spacing, x, y);
                ++i;
        }
        glDisable(GL_BLEND);
        glBindVertexArray(0);

        if (which == PIXEL_OFFSET) return spacing;

        return spacing * width_unit_per_pixel + x;
}

// //ek need to return the bounding box of the rendered text
// void render_text(const std::string str, float scale, float x, float y)
// {
//         glBindVertexArray(square_vao);
//         text_shader->use(0);
//         glActiveTexture(GL_TEXTURE0);
//         glBindTexture(GL_TEXTURE_2D, texture_obj);

//         text_shader->set_fg_color(0, 0, 0);

//         glEnable(GL_BLEND);
//         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  

//         float spacing = 0;
//         // char time_str[256]={"abcdefghijklmnopqrstuvwxyz?"};
//         // // get_time(time_str);
//         int i = 0;
//         while (str[i] != 0) {
//                 spacing = render_imaged_char((unsigned int)str[i], scale, spacing, x, y);
//                 ++i;
//         }
//         glDisable(GL_BLEND);
//         glBindVertexArray(0);
// }


void load_text_engine()
{
        uint8_t *font_bits = load_png("text24.png");
        find_boundaries_24(font_bits, 256);
        const GLsizei texture_width = 256;
        const GLsizei texture_height = 256;
        glGenTextures(1, &texture_obj);
        glBindTexture(GL_TEXTURE_2D, texture_obj);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture_width, texture_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, font_bits);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);       //GL_NEAREST
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);       //GL_NEAREST
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        free(font_bits);
}