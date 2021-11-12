#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <thread>
#include <fstream>
#include <sstream>
#include <vector>
#include <charconv>

#include "math_3d.h"
#include "shader.h"
#include "text_engine.h"
#include "image_loader.h"

#include "ogmain.h"


//-------------------------------------------------------------------------------------------

const std::string train_data_filename("../data/digits/train.csv");
const std::string test_data_filename("../data/digits/test.csv");
const std::string submit_filename("submit.csv");


struct Data : client_renderer {
        virtual void render();

        Data() : number_offset(0) {}

        std::vector<int> Y;
        std::vector<int> x;
        int number_offset;
};

//-------------------------------------------------------------------------------------------

Data *dataset0;
Data *dataset1;

//-------------------------------------------------------------------------------------------

void get_all_values(const char *first, const char *last, std::vector<int> &x) {
    int value(0);
    auto res = std::from_chars(first, last, value);
    while (res.ec == std::errc()) {
        x.push_back(value);
        res = std::from_chars(res.ptr+1, last, value);
    }
}

void load_train_data(const std::string& fname, Data &data)
{
        std::vector<int> &x = data.x;
        std::vector<int> &y = data.Y;

    std::ifstream file(fname);

    std::string str;
    std::getline(file, str);

    // x.reserve(32928000);
    while (std::getline(file, str)) {
        int value(0);
        auto res = std::from_chars(str.c_str(), str.c_str()+str.size(), value);
        y.push_back(value);
        get_all_values(res.ptr+1, str.c_str()+str.size(), x);
    }

    std::cout << "Training data, targets:  " << y.size() << std::endl;
    std::cout << "Training data, data elements:  " << x.size() << std::endl;
    std::cout << "Training data, data elements, num of features:  " << x.size() / y.size() << std::endl;
}

void load_test_data(const std::string& fname, Data &data)
{
        std::vector<int> &x = data.x;
        std::vector<int> &y = data.Y;

    std::ifstream file(fname);

    std::string str;
    std::getline(file, str);

    int element_count(0);
    while (std::getline(file, str)) {
        ++element_count;
        y.push_back(0);         // this is our current guess
        get_all_values(str.c_str(), str.c_str()+str.size(), x);
    }

    std::cout << std::endl;
    std::cout << "Test data, targets:  " << element_count << std::endl;
    std::cout << "Test data, data elements:  " << x.size() << std::endl;
    std::cout << "Test data, data elements, num of features:  " << x.size() / element_count << std::endl;
}

void write_submit(const std::string& fname, const std::vector<int> &y) {
        std::ofstream file(fname);

        file << "ImageId,Label" << std::endl;
        for (size_t i=0; i<y.size(); ++i) {
                file << i+1 << "," << y[i] << std::endl;
        }
}

//------------------------------------------------------------------------------


void client_scroll_callback(GLFWwindow* window, double xoffset, double yoffset, void *blob)
{
        Data *data = (Data*)blob;

        // yoffset seems to be the changing value on my mouse
        // -1 roll towards me
        // +1 roll away from me
        // std::cout << "scroll_callback:  " << xoffset << ", " << yoffset << '\n';
        data->number_offset -= yoffset;

        if (data->number_offset < 0) data->number_offset = 0;
        if (data->number_offset >= data->Y.size()) data->number_offset = data->Y.size() -1;
}

//-------------------------------------------------------------------------------------------

bool selection_key_handler(GLFWwindow* window, int key, int scancode, int action, int mods, void *blob)
{
        Data *data = (Data*)blob;

        if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9 && action == GLFW_PRESS) {
                data->number_offset *= 10;
                data->number_offset += key - GLFW_KEY_0;

                if (data->number_offset >= data->Y.size()) data->number_offset = data->Y.size() -1;

                return true;
        }

        if (key == GLFW_KEY_F && action == GLFW_PRESS) {
                set_client_blob(dataset0);
                return true;
        }
        if (key == GLFW_KEY_G && action == GLFW_PRESS) {
                set_client_blob(dataset1);
                return true;
        }


        if (key == GLFW_KEY_BACKSPACE && action == GLFW_PRESS) {
                data->number_offset /= 10;
                return true;
        }

        return false;
}

//-------------------------------------------------------------------------------------------

static GLuint texture_data_obj;
static void create_Data_texture(const std::vector<int> &x_train, int index)
{
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        float test[28*28];

        for (int i=0; i<28*28; ++i) {
                test[i] = x_train[28*28*index + i] / 255.0;
        }

        const GLsizei texture_width = 28;
        const GLsizei texture_height = 28;
        glBindTexture(GL_TEXTURE_2D, texture_data_obj);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture_width, texture_height, 0, GL_RED, GL_FLOAT, test);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);       //GL_NEAREST
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);       //GL_NEAREST
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void Data::render()
{
        create_Data_texture(x, number_offset);

        glBindVertexArray(square_vao);
        texture_shader->use(0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_data_obj);

        Matrix4f mat;
        mat *= translation_matrix(-1,1,0);
        mat *= scale_matrix(width_unit_per_pixel, height_unit_per_pixel, 1);
        mat *= translation_matrix(0,-200,0);
        mat *= scale_matrix(200);
        glUniformMatrix4fv(texture_shader->world_location, 1, GL_FALSE, mat.glformat());

        // set the sample coords
        GLfloat box[8] = {0,1, 1,1, 0,0, 1,0};
        glNamedBufferSubData(texture_coord_vbo, 0, sizeof(GLfloat)*8, box);

        glDrawElements(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, 0);
        glBindVertexArray(0);


        float offset = render_text("Where it's at:", 24, -1, 250, PIXEL_OFFSET) + 10;
        std::string pos = std::to_string(number_offset);
        render_text(pos.c_str(), 24, offset, 250, PIXEL_OFFSET);

        offset = render_text("What it is:", 24, -1, 0.5) + 10 * width_unit_per_pixel;
        std::string num(1, (char)(Y[number_offset] + '0'));
        render_text(num.c_str(), 24, offset, 0.5);
}

//-------------------------------------------------------------------------------------------

// called after OpenGL is initialized, so the client can create OpenGL objects, add handlers, etc
void init_client_og()
{
        add_key_handler(selection_key_handler);

        glGenTextures(1, &texture_data_obj);
}

//-------------------------------------------------------------------------------------------

int main()
{
        srand(time(NULL));

        unsigned int n = std::thread::hardware_concurrency();
        std::cout << n << " concurrent threads are supported.\n";


        Data training_data;

        auto start_time = now();
        load_train_data(train_data_filename, training_data);
        auto end_time = now();
        auto total_time = duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "Total time loading training data:  " << total_time << " ms" << std::endl;


        Data test_data;

        // std::vector<int> guess_test;
        load_test_data(test_data_filename, test_data);

        dataset0 = &training_data;
        dataset1 = &test_data;

        return og_main(&training_data);
}