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

        Data() : num_examples(0), num_features(0), displayed_index(0) {}

        virtual void load_data(const std::string& fname) = 0;

        std::vector<int> x;
        int num_examples;
        int num_features;

        std::string label;
        int displayed_index;
};


struct TrainingData : Data {
        virtual void render();
        virtual void load_data(const std::string& fname);

        std::vector<int> Y;
};

struct TestingData : Data {
        virtual void render();
        virtual void load_data(const std::string& fname);
};


//-------------------------------------------------------------------------------------------

TrainingData *training_dataset;
TestingData *testing_dataset;

//-------------------------------------------------------------------------------------------

void get_all_values(const char *first, const char *last, std::vector<int> &x) {
    int value(0);
    auto res = std::from_chars(first, last, value);
    while (res.ec == std::errc()) {
        x.push_back(value);
        res = std::from_chars(res.ptr+1, last, value);
    }
}

void TrainingData::load_data(const std::string& fname)
{
        std::ifstream file(fname);
        label = fname;

        std::string str;
        std::getline(file, str);

        // x.reserve(32928000);
        while (std::getline(file, str)) {
                int value(0);
                auto res = std::from_chars(str.c_str(), str.c_str()+str.size(), value);
                Y.push_back(value);
                ++num_examples;
                get_all_values(res.ptr+1, str.c_str()+str.size(), x);
        }
        num_features = x.size() / num_examples;

        std::cout << "Training data, examples:  " << num_examples << std::endl;
        std::cout << "Training data, targets:  " << Y.size() << std::endl;
        std::cout << "Training data, data elements:  " << x.size() << std::endl;
        std::cout << "num of features:  " << num_features << std::endl;
        std::cout << "Training data, data elements, num of features:  " << x.size() / num_examples << std::endl;
}

void TestingData::load_data(const std::string& fname)
{
        std::ifstream file(fname);
        label = fname;

        std::string str;
        std::getline(file, str);

        while (std::getline(file, str)) {
                ++num_examples;
                get_all_values(str.c_str(), str.c_str()+str.size(), x);
        }
        num_features = x.size() / num_examples;

        std::cout << std::endl;
        std::cout << "Test data, examples:  " << num_examples << std::endl;
        std::cout << "Test data, data elements:  " << x.size() << std::endl;
        std::cout << "num of features:  " << num_features << std::endl;
        std::cout << "Test data, data elements, num of features:  " << x.size() / num_examples << std::endl;
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
        data->displayed_index -= yoffset;

        if (data->displayed_index < 0) data->displayed_index = 0;
        if (data->displayed_index >= data->num_examples) data->displayed_index = data->num_examples -1;
}

//-------------------------------------------------------------------------------------------

bool selection_key_handler(GLFWwindow* window, int key, int scancode, int action, int mods, void *blob)
{
        Data *data = (Data*)blob;

        if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9 && action == GLFW_PRESS) {
                data->displayed_index *= 10;
                data->displayed_index += key - GLFW_KEY_0;

                if (data->displayed_index >= data->num_examples) data->displayed_index = data->num_examples -1;

                return true;
        }

        if (key == GLFW_KEY_F && action == GLFW_PRESS) {
                set_client_blob(training_dataset);
                return true;
        }
        if (key == GLFW_KEY_G && action == GLFW_PRESS) {
                set_client_blob(testing_dataset);
                return true;
        }


        if (key == GLFW_KEY_BACKSPACE && action == GLFW_PRESS) {
                data->displayed_index /= 10;
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
        create_Data_texture(x, displayed_index);

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

        //  display help text
        render_text("F to display Training Data", 20, 10, 400, PIXEL_OFFSET);
        render_text("G to display Test Data", 20, 10, 425, PIXEL_OFFSET);
        render_text("<esc> to exit", 20, 10, 450, PIXEL_OFFSET);
        render_text("-- Edit Selection Keys --", 20, 10, 475, PIXEL_OFFSET);
        render_text("0..9 to change selection", 20, 10, 500, PIXEL_OFFSET);
        render_text("Backspace to delete last digit", 20, 10, 525, PIXEL_OFFSET);

        // display information about the selected data set

        const std::string dataset_label_text = "Dataset:  " + label;
        render_text(dataset_label_text.c_str(), 24, 250, 25, PIXEL_OFFSET);

        const std::string examples_label_text = "Examples:  " + std::to_string(num_examples);
        render_text(examples_label_text.c_str(), 24, 250, 50, PIXEL_OFFSET);
        
        const std::string features_label_text = "Features:  " + std::to_string(num_features);
        render_text(features_label_text.c_str(), 24, 250, 75, PIXEL_OFFSET);
        

        const std::string pos_label_text = "Where it's at:  " + std::to_string(displayed_index);
        render_text(pos_label_text.c_str(), 24, 10, 250, PIXEL_OFFSET);

        // display model info, internals, output, and guess(es)
}

void TrainingData::render()
{
        Data::render();

        const std::string guess_label_text = "What it is:  " + std::to_string(Y[displayed_index]);
        render_text(guess_label_text.c_str(), 24, 10, 275, PIXEL_OFFSET);
}

void TestingData::render()
{
        Data::render();
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


        TrainingData training_data;

        auto start_time = now();
        training_data.load_data(train_data_filename);
        auto end_time = now();
        auto total_time = duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "Total time loading training data:  " << total_time << " ms" << std::endl;


        TestingData test_data;

        // std::vector<int> guess_test;
        test_data.load_data(test_data_filename);

        training_dataset = &training_data;
        testing_dataset = &test_data;

        return og_main(&training_data);
}