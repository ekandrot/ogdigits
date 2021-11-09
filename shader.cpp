#include "shader.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>


GLuint create_shader_from_file(const char *file_name, GLenum shader_type)
{
        std::ifstream file(file_name, std::ifstream::in);
        std::stringstream strStream;
        strStream << file.rdbuf();
        std::string vertexShaderSource = strStream.str();
        file.close();

        int  success;
        char infoLog[512];
        const char *c_str;

        GLuint shader_value = glCreateShader(shader_type);
        c_str = vertexShaderSource.c_str();
        glShaderSource(shader_value, 1, &c_str, NULL);
        glCompileShader(shader_value);
        
        glGetShaderiv(shader_value, GL_COMPILE_STATUS, &success);
        if (!success) {
                glGetShaderInfoLog(shader_value, 512, NULL, infoLog);
                if (shader_type == GL_VERTEX_SHADER) {
                        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
                } else if (shader_type == GL_FRAGMENT_SHADER) {
                        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
                } else {
                        std::cout << "ERROR::SHADER::UNKNOW_TYPE::COMPILATION_FAILED\n" << infoLog << std::endl;
                }
        }

        return shader_value;
}


Shader::Shader(const char *vert_file_name, const char *frag_file_name) 
{
        GLuint vertexShader = create_shader_from_file(vert_file_name, GL_VERTEX_SHADER);
        GLuint fragmentShader = create_shader_from_file(frag_file_name, GL_FRAGMENT_SHADER);

        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);

        int  success;
        char infoLog[512];
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success) {
                glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
                std::cout << "ERROR::SHADER::LINKING_FAILED\n" << infoLog << std::endl;
        }

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);  
}