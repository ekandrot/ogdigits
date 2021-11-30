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

#include "data.h"
#include "fixed_model.h"

#include "ogmain.h"


//-------------------------------------------------------------------------------------------

const std::string train_data_filename("../data/digits/train.csv");
const std::string test_data_filename("../data/digits/test.csv");
const std::string submit_filename("submit.csv");


struct Data_Renderer : public virtual Data, ClientRenderer {
        virtual void render();
};

struct TrainingData_Renderer : public TrainingData, public Data_Renderer {
        virtual void render();
};

struct TestingData_Renderer : public TestingData, public Data_Renderer {
        virtual void render();
};


std::vector<int> guesses_training;
std::vector<int> guesses_testing;

std::vector<int> *guesses = &guesses_training;

//-------------------------------------------------------------------------------------------

TrainingData_Renderer *training_dataset;
TestingData_Renderer *testing_dataset;


void write_submit(const std::string& fname, const std::vector<int> &y) {
        std::ofstream file(fname);

        file << "ImageId,Label" << std::endl;
        for (size_t i=0; i<y.size(); ++i) {
                file << i+1 << "," << y[i] << std::endl;
        }
}

//------------------------------------------------------------------------------


void client_scroll_callback(GLFWwindow* window, double xoffset, double yoffset, ClientRenderer *renderer)
{
        Data_Renderer *data = dynamic_cast<Data_Renderer*>(renderer);

        // yoffset seems to be the changing value on my mouse
        // -1 roll towards me
        // +1 roll away from me
        // std::cout << "scroll_callback:  " << xoffset << ", " << yoffset << '\n';
        data->displayed_index -= yoffset;

        if (data->displayed_index < 0) data->displayed_index = 0;
        if (data->displayed_index >= data->num_examples) data->displayed_index = data->num_examples -1;
}

//-------------------------------------------------------------------------------------------

bool selection_key_handler(GLFWwindow* window, int key, int scancode, int action, int mods, ClientRenderer *renderer)
{
        Data_Renderer *data = dynamic_cast<Data_Renderer*>(renderer);

        if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9 && action == GLFW_PRESS) {
                data->displayed_index *= 10;
                data->displayed_index += key - GLFW_KEY_0;

                if (data->displayed_index >= data->num_examples) data->displayed_index = data->num_examples -1;

                return true;
        }

        if (key == GLFW_KEY_F && action == GLFW_PRESS) {
                set_client_renderer(training_dataset);
                guesses = &guesses_training;
                return true;
        }
        if (key == GLFW_KEY_G && action == GLFW_PRESS) {
                set_client_renderer(testing_dataset);
                guesses = &guesses_testing;
                return true;
        }


        if (key == GLFW_KEY_BACKSPACE && action == GLFW_PRESS) {
                data->displayed_index /= 10;
                return true;
        }

        return false;
}

// should be back in ogmain, with the datasets supplying additional help text for it to render
// instead of this controlling all of the help text that has nothing to do with the datasets
void render_help() {
        render_text("F to display Training Data", 20, 10, 400, PIXEL_OFFSET);
        render_text("G to display Test Data", 20, 10, 425, PIXEL_OFFSET);
        render_text("h to toggle this help", 20, 10, 450, PIXEL_OFFSET);
        render_text("<esc> to exit", 20, 10, 475, PIXEL_OFFSET);
        render_text("-- Edit Selection Keys --", 20, 10, 500, PIXEL_OFFSET);
        render_text("0..9 to change selection", 20, 10, 525, PIXEL_OFFSET);
        render_text("Backspace to delete last digit", 20, 10, 550, PIXEL_OFFSET);
}

//-------------------------------------------------------------------------------------------

static GLuint texture_data_obj;
static void create_Data_texture(const std::vector<float> &x_train, int num_features, int index)
{
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        float test[28*28];

        for (int i=0; i<28*28; ++i) {
                test[i] = (x_train[num_features*index + i] + 1) / 2;
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

void Data_Renderer::render()
{
        // display the selected digit in the left, upper 200x200 pixels

        create_Data_texture(x, num_features, displayed_index);

        glBindVertexArray(square_vao);
        texture_shader->use(0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_data_obj);

        // move to left,upper, then scale to pixels for positioning
        // then scale the unit square to 200x200 pixels in dimensions
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

        const std::string guess_label_text = "What it could be:  " + std::to_string((*guesses)[displayed_index]);
        render_text(guess_label_text.c_str(), 24, 250, 275, PIXEL_OFFSET);
}

void TrainingData_Renderer::render()
{
        Data_Renderer::render();

        const std::string guess_label_text = "What it is:  " + std::to_string(Y[displayed_index]);
        render_text(guess_label_text.c_str(), 24, 10, 275, PIXEL_OFFSET);
}

void TestingData_Renderer::render()
{
        Data_Renderer::render();
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

        FixedModel model;

        TrainingData_Renderer training_data;

        auto start_time = now();
        training_data.load_data(train_data_filename);
        auto end_time = now();
        auto total_time = duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "Total time loading training data:  " << total_time << " ms" << std::endl;

        guesses_training.resize(training_data.num_examples);
        model.predict(&training_data, guesses_training);


        TestingData_Renderer test_data;

        // std::vector<int> guess_test;
        test_data.load_data(test_data_filename);
        guesses_testing.resize(test_data.num_examples);
        model.predict(&test_data, guesses_testing);

        training_dataset = &training_data;
        testing_dataset = &test_data;

        return og_main(&training_data);
}