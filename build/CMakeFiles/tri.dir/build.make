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
CMAKE_SOURCE_DIR = /home/minht57/CDS/CDS_R1_TSR

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/minht57/CDS/CDS_R1_TSR/build

# Include any dependencies generated for this target.
include CMakeFiles/tri.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tri.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tri.dir/flags.make

CMakeFiles/tri.dir/src/tri.cpp.o: CMakeFiles/tri.dir/flags.make
CMakeFiles/tri.dir/src/tri.cpp.o: ../src/tri.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/minht57/CDS/CDS_R1_TSR/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tri.dir/src/tri.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tri.dir/src/tri.cpp.o -c /home/minht57/CDS/CDS_R1_TSR/src/tri.cpp

CMakeFiles/tri.dir/src/tri.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tri.dir/src/tri.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/minht57/CDS/CDS_R1_TSR/src/tri.cpp > CMakeFiles/tri.dir/src/tri.cpp.i

CMakeFiles/tri.dir/src/tri.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tri.dir/src/tri.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/minht57/CDS/CDS_R1_TSR/src/tri.cpp -o CMakeFiles/tri.dir/src/tri.cpp.s

CMakeFiles/tri.dir/src/tri.cpp.o.requires:

.PHONY : CMakeFiles/tri.dir/src/tri.cpp.o.requires

CMakeFiles/tri.dir/src/tri.cpp.o.provides: CMakeFiles/tri.dir/src/tri.cpp.o.requires
	$(MAKE) -f CMakeFiles/tri.dir/build.make CMakeFiles/tri.dir/src/tri.cpp.o.provides.build
.PHONY : CMakeFiles/tri.dir/src/tri.cpp.o.provides

CMakeFiles/tri.dir/src/tri.cpp.o.provides.build: CMakeFiles/tri.dir/src/tri.cpp.o


# Object files for target tri
tri_OBJECTS = \
"CMakeFiles/tri.dir/src/tri.cpp.o"

# External object files for target tri
tri_EXTERNAL_OBJECTS =

tri: CMakeFiles/tri.dir/src/tri.cpp.o
tri: CMakeFiles/tri.dir/build.make
tri: dlib_build/libdlib.a
tri: /opt/ros/kinetic/lib/libopencv_stitching3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_superres3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_videostab3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_aruco3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_bgsegm3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_bioinspired3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_ccalib3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_cvv3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_datasets3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_dpm3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_face3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_fuzzy3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_hdf3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_line_descriptor3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_optflow3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_plot3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_reg3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_saliency3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_stereo3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_structured_light3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_surface_matching3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_text3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_xfeatures2d3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_ximgproc3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_xobjdetect3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_xphoto3.so.3.2.0
tri: /usr/lib/x86_64-linux-gnu/libnsl.so
tri: /usr/lib/x86_64-linux-gnu/libSM.so
tri: /usr/lib/x86_64-linux-gnu/libICE.so
tri: /usr/lib/x86_64-linux-gnu/libX11.so
tri: /usr/lib/x86_64-linux-gnu/libXext.so
tri: /usr/lib/x86_64-linux-gnu/libgif.so
tri: /usr/lib/x86_64-linux-gnu/libpng.so
tri: /usr/lib/x86_64-linux-gnu/libz.so
tri: /usr/lib/x86_64-linux-gnu/libjpeg.so
tri: /usr/lib/libblas.so
tri: /usr/lib/liblapack.so
tri: /usr/lib/x86_64-linux-gnu/libsqlite3.so
tri: /opt/ros/kinetic/lib/libopencv_shape3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_video3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_viz3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_phase_unwrapping3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_rgbd3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_calib3d3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_features2d3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_flann3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_objdetect3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_ml3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_highgui3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_photo3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_videoio3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_imgcodecs3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_imgproc3.so.3.2.0
tri: /opt/ros/kinetic/lib/libopencv_core3.so.3.2.0
tri: CMakeFiles/tri.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/minht57/CDS/CDS_R1_TSR/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tri"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tri.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tri.dir/build: tri

.PHONY : CMakeFiles/tri.dir/build

CMakeFiles/tri.dir/requires: CMakeFiles/tri.dir/src/tri.cpp.o.requires

.PHONY : CMakeFiles/tri.dir/requires

CMakeFiles/tri.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tri.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tri.dir/clean

CMakeFiles/tri.dir/depend:
	cd /home/minht57/CDS/CDS_R1_TSR/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/minht57/CDS/CDS_R1_TSR /home/minht57/CDS/CDS_R1_TSR /home/minht57/CDS/CDS_R1_TSR/build /home/minht57/CDS/CDS_R1_TSR/build /home/minht57/CDS/CDS_R1_TSR/build/CMakeFiles/tri.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tri.dir/depend
