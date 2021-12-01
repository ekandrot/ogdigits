#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <charconv>
#include <thread>

#include "math_3d.h"
#include "shader.h"
#include "text_engine.h"
#include "image_loader.h"

#include "ogmain.h"


//------------------------------------------------------------------------------

GLuint VAO, COLOR_BO, EBO;
GLuint square_vao, texture_coord_vbo;
GLuint line_vao, line_vbo;

ColorShader *color_shader;
TextShader *text_shader;
TextureShader *texture_shader;
float window_ratio;

circle_obj *circle_32;


ClientRenderer *client_renderer;
int client_update_interval{-1};

//------------------------------------------------------------------------------

#define W_WIDTH 1024
#define W_HEIGHT 768

const char *PREFS_FILE_NAME = "init.txt";

struct Preferences {
        int width;
        int height;
        int x;
        int y;
        int help_on;
};

Preferences prefs;

double mousex, mousey;

bool refresh_needed{true};

bool gWindowWasResized{false};
int gResizedWindowWidth;
int gResizedWindowHeight;

float width_unit_per_pixel = 1;
float height_unit_per_pixel = 1;


void parse_init_window(std::ifstream &f)
{
        std::string line;
        std::getline(f, line);

        while (line != "\n" && line != "") {
                std::istringstream fields(line);
                std::string field;
                fields >> field;
                if (field == "width") {
                        fields >> prefs.width;
                        mousex = prefs.width / 2;
                } if (field == "height") {
                        fields >> prefs.height;
                        mousey = prefs.height / 2;
                } if (field == "x") {
                        fields >> prefs.x;
                } if (field == "y") {
                        fields >> prefs.y;
                } if (field == "help_on") {
                        fields >> prefs.help_on;
                        if (prefs.help_on > 0) {
                                prefs.help_on = 1;
                        } else {
                                prefs.help_on = 0;
                        }
                }
        
                std::getline(f, line);
        }
}

void load_init_file(const char *file_name)
{
        prefs.height = W_HEIGHT;
        prefs.width = W_WIDTH;
        prefs.x = 100;
        prefs.y = 100;
        prefs.help_on = 1;

        std::ifstream infile(file_name);
        if (infile.is_open()) {
                std::string line;
                std::getline(infile, line);

                if (line == "[window]") {
                        parse_init_window(infile);
                }
        }

        window_ratio = prefs.height/(float)prefs.width;
        width_unit_per_pixel = 2.0 / prefs.width;
        height_unit_per_pixel = 2.0 / prefs.height;
}

void write_prefs_file(const char *file_name) {
        std::ofstream infile(file_name);
        infile << "[window]\n";
        infile << "width " << prefs.width << "\n";
        infile << "height " << prefs.height << "\n";
        infile << "x " << prefs.x << "\n";
        infile << "y " << prefs.y << "\n";
        infile << "help_on " << prefs.help_on << "\n";
}

//-------------------------------------------------------------------------------------------

std::vector<bool (*) (GLFWwindow* window, int key, int scancode, int action, int mods, ClientRenderer *renderer)> key_handlers;

bool default_keys_callback(GLFWwindow* window, int key, int scancode, int action, int mods, ClientRenderer *renderer);
bool dialog_keys_callback(GLFWwindow* window, int key, int scancode, int action, int mods, ClientRenderer *renderer);
bool capture_keys_callback(GLFWwindow* window, int key, int scancode, int action, int mods, ClientRenderer *renderer);


void add_key_handler(bool (*function) (GLFWwindow* window, int key, int scancode, int action, int mods, ClientRenderer *renderer))
{
        key_handlers.push_back(function);
}

void remove_key_handler(bool (*function) (GLFWwindow* window, int key, int scancode, int action, int mods, ClientRenderer *renderer))
{
        for (auto i = key_handlers.begin(); i != key_handlers.end(); ++i) {
                if (*i == function) {
                        key_handlers.erase(i);
                        return;
                }
        }
}

bool default_keys_callback(GLFWwindow* window, int key, int scancode, int action, int mods, ClientRenderer *renderer)
{
        if (key == GLFW_KEY_H && action == GLFW_PRESS) {
                prefs.help_on = 1 - prefs.help_on;
                write_prefs_file(PREFS_FILE_NAME);
                return true;        // toggling the help text
        }
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                return true;        // app is done, nothing more to do
        }
        return false;
}

// bool dialog_keys_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
// {
//         if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
//                 in_dialog = false;
//                 remove_key_handler(dialog_keys_callback);
//                 return true;
//         }
//         if (key == GLFW_KEY_ENTER && action == GLFW_PRESS) {
//                 in_dialog = false;
//                 remove_key_handler(dialog_keys_callback);
//                 return true;
//         }

//         return true;    // no one else gets keys when we are in focus
// }

//-------------------------------------------------------------------------------------------

void set_update_interval(int seconds)
{
        client_update_interval = seconds;
}


static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
        for (auto i = key_handlers.rbegin(); i != key_handlers.rend(); ++i) {
                if ((*i)(window, key, scancode, action, mods, client_renderer)) {
                        return;
                }
        }
}

static bool g_cursor_within_window{true};
void cursor_enter_callback(GLFWwindow *window, int entered)
{
        if (entered == GLFW_TRUE) {
                g_cursor_within_window = true;
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
        } else {
                g_cursor_within_window = false;
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
}

void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos)
{
        mousex = xpos;
        mousey = ypos;
}

void mouse_down_event(GLFWwindow* window, int button, int action, int mods)
{

}

// void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
// {
//         // yoffset seems to be the changing value on my mouse
//         // -1 squares should get bigger
//         // +1 squares should get smaller
//         // std::cout << "scroll_callback:  " << xoffset << ", " << yoffset << '\n';
// }

void window_moved_callback(GLFWwindow* window, int xpos, int ypos)
{
        GLFWmonitor *mainMonitor = glfwGetPrimaryMonitor();
        int mxpos, mypos, mwidth, mheight;
        glfwGetMonitorWorkarea(mainMonitor, &mxpos, &mypos, &mwidth, &mheight);

        prefs.x = xpos - mxpos;
        prefs.y = ypos - mypos;

        write_prefs_file(PREFS_FILE_NAME);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
        gWindowWasResized = true;
        gResizedWindowWidth = width;
        gResizedWindowHeight = height;
}

void error_callback(int error, const char* description)
{
        fprintf(stderr, "*** GLFW Error Callback %d: %s\n", error, description);
}

void set_context_version()
{
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        // "This must only be used if the requested OpenGL version is 3.0 or above."
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); 
}

GLFWwindow* create_window_onscreen(const char *window_name)
{
        GLFWmonitor *mainMonitor = glfwGetPrimaryMonitor();
        int mxpos, mypos, mwidth, mheight;
        glfwGetMonitorWorkarea(mainMonitor, &mxpos, &mypos, &mwidth, &mheight);
        // printf("mxpos %d, mypos %d, mwidth %d, mheight %d\n", mxpos, mypos, mwidth, mheight);

        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        GLFWwindow *window = glfwCreateWindow(prefs.width, prefs.height, window_name, NULL, NULL);
        if (!window) {
                return nullptr;
        }
        glfwSetWindowSizeLimits(window, 400, 400, GLFW_DONT_CARE, GLFW_DONT_CARE);
        glfwSetWindowPos(window, mxpos + prefs.x, mypos + prefs.y);
        return window;
}

//-------------------------------------------------------------------------------------------

void render_test_image()
{
        // glBindVertexArray(square_vao);
        // texture_shader->use(0);
        // glActiveTexture(GL_TEXTURE0);
        // glBindTexture(GL_TEXTURE_2D, texture_obj);

        // Matrix4f mat;
        // mat *= scale_matrix(window_ratio,1,1);
        // mat *= scale_matrix(750.0/514,1,1);
        // // mat *= translation_matrix(x, y, 0);
        // // mat *= scale_matrix(scale);
        // // mat *= translation_matrix(spacing, 0, 0);
        // // mat *= scale_matrix(ratio, 1, 1);
        // glUniformMatrix4fv(texture_shader->world_location, 1, GL_FALSE, mat.glformat());

        // // set the sample coords
        // GLfloat box[8] = {0,1, 1,1, 0,0, 1,0};
        // glNamedBufferSubData(texture_coord_vbo, 0, sizeof(GLfloat)*8, box);

        // // glEnable(GL_BLEND);
        // // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  
        // glDrawElements(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, 0);
        // // glBindVertexArray(0);
}

// get the coords in the cursor in world, based on last callback from cursor_pos_update
void get_cursor_coords(float &x, float &y)
{
        x = mousex / (float)prefs.width * 2 - 1;
        y = mousey / (float)prefs.height * 2 - 1;
        y = -y;
}

void draw_cursor()
{
        float in_world_x, in_world_y;
        get_cursor_coords(in_world_x, in_world_y);

        color_shader->use();

        float scale{0.01};
        Matrix4f where(translation_matrix(in_world_x, in_world_y, 0));
        where *= scale_matrix(scale*window_ratio, scale, scale);

        glUniformMatrix4fv(color_shader->gWorldLocation, 1, GL_FALSE, where.glformat());

        glVertexAttrib4f(1, 0.75f, 0, 0, 1);

        glBindVertexArray(circle_32->VAO);
        // glDisableVertexAttribArray(1);
        glDrawArrays(GL_TRIANGLE_FAN, 0, circle_32->num_points);
        // glBindVertexArray(0);
}


void render_scene()
{
        client_renderer->render();

        if (prefs.help_on) {
                render_help();
        }

        draw_cursor();
}

//-------------------------------------------------------------------------------------------

void init_opengl_objects()
{



        color_shader = new ColorShader();
        text_shader = new TextShader();
        texture_shader = new TextureShader();

        circle_32 = new circle_obj(16);


        glGenBuffers(1, &EBO);  
        glGenBuffers(1, &COLOR_BO);


        GLuint VBO;



        GLfloat square_2d_coords[] = {
                0.0f, 0.0f,
                1.0f, 0.0f,
                0.0f, 1.0f,
                1.0f, 1.0f,
        };
        glGenVertexArrays(1, &square_vao);
        glBindVertexArray(square_vao);

        // pointer to square coords for attribute 0
        glGenBuffers(1, &VBO);  
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
        glEnableVertexAttribArray(0);
        glBufferData(GL_ARRAY_BUFFER, sizeof(square_2d_coords), square_2d_coords, GL_STATIC_DRAW);
        // glBindBuffer(GL_ARRAY_BUFFER, 0);

        // pointer to texture coords for attribute 1
        glGenBuffers(1, &texture_coord_vbo);  
        glBindBuffer(GL_ARRAY_BUFFER, texture_coord_vbo);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
        glEnableVertexAttribArray(1);
        glBufferData(GL_ARRAY_BUFFER, sizeof(square_2d_coords), nullptr, GL_DYNAMIC_DRAW);
        // glBindBuffer(GL_ARRAY_BUFFER, 0);

        // one GL_ELEMENT_ARRAY_BUFFER per VAO
        GLushort indices[] = {0, 1, 2, 3};
        glGenBuffers(1, &EBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        // glBindVertexArray(0);




        float vertices[] = {
                0.5f,  0.5f, 0.0f,  // top right
                0.5f, -0.5f, 0.0f,  // bottom right
                -0.5f, -0.5f, 0.0f,  // bottom left
                -0.5f,  0.5f, 0.0f   // top left 
        };  
        // unsigned int indices[] = {  // note that we start from 0!
        //         0, 1, 3,   // first triangle
        //         1, 2, 3    // second triangle
        // };  


        glGenVertexArrays(1, &VAO);  
        glBindVertexArray(VAO);

        glGenBuffers(1, &VBO);  
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        // glGenBuffers(1, &EBO);
        // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        // glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW); 

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);  

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        // glBindVertexArray(0);



        glGenVertexArrays(1, &line_vao);  
        glBindVertexArray(line_vao);

        glGenBuffers(1, &line_vbo);  
        glBindBuffer(GL_ARRAY_BUFFER, line_vbo);
        glBufferData(GL_ARRAY_BUFFER, 8*sizeof(float), nullptr, GL_DYNAMIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);  

        // glBindVertexArray(0);



        // jpeg_test_texture();
}

//-------------------------------------------------------------------------------------------

void set_client_renderer(ClientRenderer *renderer)
{
        client_renderer = renderer;
}

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
        client_scroll_callback(window, xoffset, yoffset, client_renderer);
}

int og_main(ClientRenderer *renderer)
{
        client_renderer = renderer;

        srand(time(NULL));

        load_init_file(PREFS_FILE_NAME);

        if (!glfwInit()) {
                printf("*** Couldn't glfw3 init\n");
                return -1;
        }
        glfwSetErrorCallback(error_callback);

        set_context_version();
        
        GLFWwindow *window = create_window_onscreen("Digits");
        if (window == nullptr) {
                printf("*** Couldn't open a darn window\n");
                glfwTerminate();
                return -1;
        }

        glfwShowWindow(window);
        glfwMakeContextCurrent(window);

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		printf("*** Failed to GLEW it:  %s\n", glewGetErrorString(err));
                glfwTerminate();
		return -1;
	}
	// printf("GL version:  %s\n", glGetString(GL_VERSION));
	// printf("GLEW version:  %s\n", glewGetString(GLEW_VERSION));

        glViewport(0, 0, prefs.width, prefs.height);
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

        init_opengl_objects();
        load_text_engine();

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	// glFrontFace(GL_CCW);
	// glCullFace(GL_BACK);
	// glEnable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);

        add_key_handler(default_keys_callback);


        glfwSetMouseButtonCallback(window, mouse_down_event);
        glfwSetCursorEnterCallback(window, cursor_enter_callback);
        glfwSetCursorPosCallback(window, cursor_pos_callback);
        glfwSetKeyCallback(window, key_callback);
        glfwSetWindowPosCallback(window, window_moved_callback);
        glfwSetScrollCallback(window, scroll_callback);
        glfwSwapInterval(0);

        double last_time = glfwGetTime();
        double xpos_prev, ypos_prev;

       	glfwSetCursorPos(window, prefs.width/2, prefs.height/2);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
        glfwGetCursorPos(window, &xpos_prev, &ypos_prev);


        init_client_og();


        double last_client_called{0};
	while (!glfwWindowShouldClose(window)) {
                double current_time = glfwGetTime();
                double delta_time = current_time - last_time;

                if (client_update_interval > 0) {
                        if (client_update_interval <= current_time - last_client_called) {
                                last_client_called = current_time;
                                std::thread (&ClientRenderer::data_update, renderer).detach();
                        }
                }

                if (delta_time >= 0.01) {
                        last_time = current_time;

                        // update the cursor, if it in the window
                        if (g_cursor_within_window) {
                        	glClearColor(0.5f, 0.75f, 0.5f, 1.0f);
                                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                        } else {
                        	// glClearColor(0.5f, 0.0f, 0.0f, 0.0f);
                                // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                        }

                	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                        refresh_needed = true;
                }

                if (refresh_needed) {
                        refresh_needed = false;

                        if (gWindowWasResized) {
                                gWindowWasResized = false;

                                glViewport(0, 0, gResizedWindowWidth, gResizedWindowHeight);
                                prefs.width = gResizedWindowWidth;
                                prefs.height = gResizedWindowHeight;
                                width_unit_per_pixel = 2.0 / prefs.width;
                                height_unit_per_pixel = 2.0 / prefs.height;
                                window_ratio = prefs.height/(float)prefs.width;
                                write_prefs_file(PREFS_FILE_NAME);
                        }


                        render_scene();
                        glfwSwapBuffers(window);
                }

		glfwPollEvents();
        }

        write_prefs_file(PREFS_FILE_NAME);

        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
}