# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_SOURCE_DIR = /home/duongthanh3327/CuocDuaSo/tsr

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/duongthanh3327/CuocDuaSo/tsr/build

# Include any dependencies generated for this target.
include CMakeFiles/calib.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/calib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/calib.dir/flags.make

CMakeFiles/calib.dir/src/calib.cpp.o: CMakeFiles/calib.dir/flags.make
CMakeFiles/calib.dir/src/calib.cpp.o: ../src/calib.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/duongthanh3327/CuocDuaSo/tsr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/calib.dir/src/calib.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/calib.dir/src/calib.cpp.o -c /home/duongthanh3327/CuocDuaSo/tsr/src/calib.cpp

CMakeFiles/calib.dir/src/calib.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/calib.dir/src/calib.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/duongthanh3327/CuocDuaSo/tsr/src/calib.cpp > CMakeFiles/calib.dir/src/calib.cpp.i

CMakeFiles/calib.dir/src/calib.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/calib.dir/src/calib.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/duongthanh3327/CuocDuaSo/tsr/src/calib.cpp -o CMakeFiles/calib.dir/src/calib.cpp.s

CMakeFiles/calib.dir/src/calib.cpp.o.requires:

.PHONY : CMakeFiles/calib.dir/src/calib.cpp.o.requires

CMakeFiles/calib.dir/src/calib.cpp.o.provides: CMakeFiles/calib.dir/src/calib.cpp.o.requires
	$(MAKE) -f CMakeFiles/calib.dir/build.make CMakeFiles/calib.dir/src/calib.cpp.o.provides.build
.PHONY : CMakeFiles/calib.dir/src/calib.cpp.o.provides

CMakeFiles/calib.dir/src/calib.cpp.o.provides.build: CMakeFiles/calib.dir/src/calib.cpp.o


# Object files for target calib
calib_OBJECTS = \
"CMakeFiles/calib.dir/src/calib.cpp.o"

# External object files for target calib
calib_EXTERNAL_OBJECTS =

calib: CMakeFiles/calib.dir/src/calib.cpp.o
calib: CMakeFiles/calib.dir/build.make
calib: dlib_build/libdlib.a
calib: /usr/local/lib/libopencv_shape.so.3.2.0
calib: /usr/local/lib/libopencv_stitching.so.3.2.0
calib: /usr/local/lib/libopencv_superres.so.3.2.0
calib: /usr/local/lib/libopencv_videostab.so.3.2.0
calib: /usr/local/lib/libopencv_viz.so.3.2.0
calib: /usr/lib/x86_64-linux-gnu/libnsl.so
calib: /usr/lib/x86_64-linux-gnu/libSM.so
calib: /usr/lib/x86_64-linux-gnu/libICE.so
calib: /usr/lib/x86_64-linux-gnu/libX11.so
calib: /usr/lib/x86_64-linux-gnu/libXext.so
calib: /usr/lib/x86_64-linux-gnu/libpng.so
calib: /usr/lib/x86_64-linux-gnu/libz.so
calib: /usr/lib/x86_64-linux-gnu/libjpeg.so
calib: /usr/local/lib/libopencv_objdetect.so.3.2.0
calib: /usr/local/lib/libopencv_calib3d.so.3.2.0
calib: /usr/local/lib/libopencv_features2d.so.3.2.0
calib: /usr/local/lib/libopencv_flann.so.3.2.0
calib: /usr/local/lib/libopencv_highgui.so.3.2.0
calib: /usr/local/lib/libopencv_ml.so.3.2.0
calib: /usr/local/lib/libopencv_photo.so.3.2.0
calib: /usr/local/lib/libopencv_video.so.3.2.0
calib: /usr/local/lib/libopencv_videoio.so.3.2.0
calib: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
calib: /usr/local/lib/libopencv_imgproc.so.3.2.0
calib: /usr/local/lib/libopencv_core.so.3.2.0
calib: CMakeFiles/calib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/duongthanh3327/CuocDuaSo/tsr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable calib"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/calib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/calib.dir/build: calib

.PHONY : CMakeFiles/calib.dir/build

CMakeFiles/calib.dir/requires: CMakeFiles/calib.dir/src/calib.cpp.o.requires

.PHONY : CMakeFiles/calib.dir/requires

CMakeFiles/calib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/calib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/calib.dir/clean

CMakeFiles/calib.dir/depend:
	cd /home/duongthanh3327/CuocDuaSo/tsr/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/duongthanh3327/CuocDuaSo/tsr /home/duongthanh3327/CuocDuaSo/tsr /home/duongthanh3327/CuocDuaSo/tsr/build /home/duongthanh3327/CuocDuaSo/tsr/build /home/duongthanh3327/CuocDuaSo/tsr/build/CMakeFiles/calib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/calib.dir/depend

