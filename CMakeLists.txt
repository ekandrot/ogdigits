cmake_minimum_required(VERSION 3.16)

project(ogdigits.app)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(ogdigits.app main.cpp)
target_sources(ogdigits.app
        PRIVATE shader.cpp
        PRIVATE image_loader.cpp
        PRIVATE opengl_globals.cpp
        PRIVATE text_engine.cpp)

cmake_policy (SET CMP0072 OLD)

find_package(PNG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(JPEG REQUIRED)

include_directories( "~/arcane/eklib" ${OPENGL_INCLUDE_DIRS} ${PNG_INCLUDE_DIRS}  ${GLFW_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${JPEG_INCLUDE_DIRS} )

target_link_libraries(ogdigits.app ${OPENGL_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARIES} glfw GLEW::GLEW )

set(default_build_type "Release")
