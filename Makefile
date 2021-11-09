# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/reed/arcane/ai/ogdigits

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/reed/arcane/ai/ogdigits

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/reed/arcane/ai/ogdigits/CMakeFiles /home/reed/arcane/ai/ogdigits/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/reed/arcane/ai/ogdigits/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named ogdigits.app

# Build rule for target.
ogdigits.app: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 ogdigits.app
.PHONY : ogdigits.app

# fast build rule for target.
ogdigits.app/fast:
	$(MAKE) -f CMakeFiles/ogdigits.app.dir/build.make CMakeFiles/ogdigits.app.dir/build
.PHONY : ogdigits.app/fast

image_loader.o: image_loader.cpp.o

.PHONY : image_loader.o

# target to build an object file
image_loader.cpp.o:
	$(MAKE) -f CMakeFiles/ogdigits.app.dir/build.make CMakeFiles/ogdigits.app.dir/image_loader.cpp.o
.PHONY : image_loader.cpp.o

image_loader.i: image_loader.cpp.i

.PHONY : image_loader.i

# target to preprocess a source file
image_loader.cpp.i:
	$(MAKE) -f CMakeFiles/ogdigits.app.dir/build.make CMakeFiles/ogdigits.app.dir/image_loader.cpp.i
.PHONY : image_loader.cpp.i

image_loader.s: image_loader.cpp.s

.PHONY : image_loader.s

# target to generate assembly for a file
image_loader.cpp.s:
	$(MAKE) -f CMakeFiles/ogdigits.app.dir/build.make CMakeFiles/ogdigits.app.dir/image_loader.cpp.s
.PHONY : image_loader.cpp.s

main.o: main.cpp.o

.PHONY : main.o

# target to build an object file
main.cpp.o:
	$(MAKE) -f CMakeFiles/ogdigits.app.dir/build.make CMakeFiles/ogdigits.app.dir/main.cpp.o
.PHONY : main.cpp.o

main.i: main.cpp.i

.PHONY : main.i

# target to preprocess a source file
main.cpp.i:
	$(MAKE) -f CMakeFiles/ogdigits.app.dir/build.make CMakeFiles/ogdigits.app.dir/main.cpp.i
.PHONY : main.cpp.i

main.s: main.cpp.s

.PHONY : main.s

# target to generate assembly for a file
main.cpp.s:
	$(MAKE) -f CMakeFiles/ogdigits.app.dir/build.make CMakeFiles/ogdigits.app.dir/main.cpp.s
.PHONY : main.cpp.s

opengl_globals.o: opengl_globals.cpp.o

.PHONY : opengl_globals.o

# target to build an object file
opengl_globals.cpp.o:
	$(MAKE) -f CMakeFiles/ogdigits.app.dir/build.make CMakeFiles/ogdigits.app.dir/opengl_globals.cpp.o
.PHONY : opengl_globals.cpp.o

opengl_globals.i: opengl_globals.cpp.i

.PHONY : opengl_globals.i

# target to preprocess a source file
opengl_globals.cpp.i:
	$(MAKE) -f CMakeFiles/ogdigits.app.dir/build.make CMakeFiles/ogdigits.app.dir/opengl_globals.cpp.i
.PHONY : opengl_globals.cpp.i

opengl_globals.s: opengl_globals.cpp.s

.PHONY : opengl_globals.s

# target to generate assembly for a file
opengl_globals.cpp.s:
	$(MAKE) -f CMakeFiles/ogdigits.app.dir/build.make CMakeFiles/ogdigits.app.dir/opengl_globals.cpp.s
.PHONY : opengl_globals.cpp.s

shader.o: shader.cpp.o

.PHONY : shader.o

# target to build an object file
shader.cpp.o:
	$(MAKE) -f CMakeFiles/ogdigits.app.dir/build.make CMakeFiles/ogdigits.app.dir/shader.cpp.o
.PHONY : shader.cpp.o

shader.i: shader.cpp.i

.PHONY : shader.i

# target to preprocess a source file
shader.cpp.i:
	$(MAKE) -f CMakeFiles/ogdigits.app.dir/build.make CMakeFiles/ogdigits.app.dir/shader.cpp.i
.PHONY : shader.cpp.i

shader.s: shader.cpp.s

.PHONY : shader.s

# target to generate assembly for a file
shader.cpp.s:
	$(MAKE) -f CMakeFiles/ogdigits.app.dir/build.make CMakeFiles/ogdigits.app.dir/shader.cpp.s
.PHONY : shader.cpp.s

text_engine.o: text_engine.cpp.o

.PHONY : text_engine.o

# target to build an object file
text_engine.cpp.o:
	$(MAKE) -f CMakeFiles/ogdigits.app.dir/build.make CMakeFiles/ogdigits.app.dir/text_engine.cpp.o
.PHONY : text_engine.cpp.o

text_engine.i: text_engine.cpp.i

.PHONY : text_engine.i

# target to preprocess a source file
text_engine.cpp.i:
	$(MAKE) -f CMakeFiles/ogdigits.app.dir/build.make CMakeFiles/ogdigits.app.dir/text_engine.cpp.i
.PHONY : text_engine.cpp.i

text_engine.s: text_engine.cpp.s

.PHONY : text_engine.s

# target to generate assembly for a file
text_engine.cpp.s:
	$(MAKE) -f CMakeFiles/ogdigits.app.dir/build.make CMakeFiles/ogdigits.app.dir/text_engine.cpp.s
.PHONY : text_engine.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... ogdigits.app"
	@echo "... image_loader.o"
	@echo "... image_loader.i"
	@echo "... image_loader.s"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
	@echo "... opengl_globals.o"
	@echo "... opengl_globals.i"
	@echo "... opengl_globals.s"
	@echo "... shader.o"
	@echo "... shader.i"
	@echo "... shader.s"
	@echo "... text_engine.o"
	@echo "... text_engine.i"
	@echo "... text_engine.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

