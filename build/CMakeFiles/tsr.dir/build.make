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
CMAKE_SOURCE_DIR = /home/ltadl/CDS/CDS_R1_TSR

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ltadl/CDS/CDS_R1_TSR/build

# Include any dependencies generated for this target.
include CMakeFiles/tsr.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tsr.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tsr.dir/flags.make

CMakeFiles/tsr.dir/src/tsr.cpp.o: CMakeFiles/tsr.dir/flags.make
CMakeFiles/tsr.dir/src/tsr.cpp.o: ../src/tsr.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ltadl/CDS/CDS_R1_TSR/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tsr.dir/src/tsr.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tsr.dir/src/tsr.cpp.o -c /home/ltadl/CDS/CDS_R1_TSR/src/tsr.cpp

CMakeFiles/tsr.dir/src/tsr.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tsr.dir/src/tsr.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ltadl/CDS/CDS_R1_TSR/src/tsr.cpp > CMakeFiles/tsr.dir/src/tsr.cpp.i

CMakeFiles/tsr.dir/src/tsr.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tsr.dir/src/tsr.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ltadl/CDS/CDS_R1_TSR/src/tsr.cpp -o CMakeFiles/tsr.dir/src/tsr.cpp.s

CMakeFiles/tsr.dir/src/tsr.cpp.o.requires:

.PHONY : CMakeFiles/tsr.dir/src/tsr.cpp.o.requires

CMakeFiles/tsr.dir/src/tsr.cpp.o.provides: CMakeFiles/tsr.dir/src/tsr.cpp.o.requires
	$(MAKE) -f CMakeFiles/tsr.dir/build.make CMakeFiles/tsr.dir/src/tsr.cpp.o.provides.build
.PHONY : CMakeFiles/tsr.dir/src/tsr.cpp.o.provides

CMakeFiles/tsr.dir/src/tsr.cpp.o.provides.build: CMakeFiles/tsr.dir/src/tsr.cpp.o


# Object files for target tsr
tsr_OBJECTS = \
"CMakeFiles/tsr.dir/src/tsr.cpp.o"

# External object files for target tsr
tsr_EXTERNAL_OBJECTS =

tsr: CMakeFiles/tsr.dir/src/tsr.cpp.o
tsr: CMakeFiles/tsr.dir/build.make
tsr: dlib_build/libdlib.a
tsr: /usr/local/lib/libopencv_stitching.so.3.2.0
tsr: /usr/local/lib/libopencv_superres.so.3.2.0
tsr: /usr/local/lib/libopencv_videostab.so.3.2.0
tsr: /usr/local/lib/libopencv_aruco.so.3.2.0
tsr: /usr/local/lib/libopencv_bgsegm.so.3.2.0
tsr: /usr/local/lib/libopencv_bioinspired.so.3.2.0
tsr: /usr/local/lib/libopencv_ccalib.so.3.2.0
tsr: /usr/local/lib/libopencv_dpm.so.3.2.0
tsr: /usr/local/lib/libopencv_freetype.so.3.2.0
tsr: /usr/local/lib/libopencv_fuzzy.so.3.2.0
tsr: /usr/local/lib/libopencv_hdf.so.3.2.0
tsr: /usr/local/lib/libopencv_line_descriptor.so.3.2.0
tsr: /usr/local/lib/libopencv_optflow.so.3.2.0
tsr: /usr/local/lib/libopencv_reg.so.3.2.0
tsr: /usr/local/lib/libopencv_saliency.so.3.2.0
tsr: /usr/local/lib/libopencv_stereo.so.3.2.0
tsr: /usr/local/lib/libopencv_structured_light.so.3.2.0
tsr: /usr/local/lib/libopencv_surface_matching.so.3.2.0
tsr: /usr/local/lib/libopencv_tracking.so.3.2.0
tsr: /usr/local/lib/libopencv_xfeatures2d.so.3.2.0
tsr: /usr/local/lib/libopencv_ximgproc.so.3.2.0
tsr: /usr/local/lib/libopencv_xobjdetect.so.3.2.0
tsr: /usr/local/lib/libopencv_xphoto.so.3.2.0
tsr: /usr/lib/x86_64-linux-gnu/libnsl.so
tsr: /usr/lib/x86_64-linux-gnu/libSM.so
tsr: /usr/lib/x86_64-linux-gnu/libICE.so
tsr: /usr/lib/x86_64-linux-gnu/libX11.so
tsr: /usr/lib/x86_64-linux-gnu/libXext.so
tsr: /usr/lib/x86_64-linux-gnu/libpng.so
tsr: /usr/lib/x86_64-linux-gnu/libz.so
tsr: /usr/lib/x86_64-linux-gnu/libjpeg.so
tsr: /usr/local/lib/libopencv_shape.so.3.2.0
tsr: /usr/local/lib/libopencv_viz.so.3.2.0
tsr: /usr/local/lib/libopencv_phase_unwrapping.so.3.2.0
tsr: /usr/local/lib/libopencv_rgbd.so.3.2.0
tsr: /usr/local/lib/libopencv_calib3d.so.3.2.0
tsr: /usr/local/lib/libopencv_video.so.3.2.0
tsr: /usr/local/lib/libopencv_datasets.so.3.2.0
tsr: /usr/local/lib/libopencv_dnn.so.3.2.0
tsr: /usr/local/lib/libopencv_face.so.3.2.0
tsr: /usr/local/lib/libopencv_plot.so.3.2.0
tsr: /usr/local/lib/libopencv_text.so.3.2.0
tsr: /usr/local/lib/libopencv_features2d.so.3.2.0
tsr: /usr/local/lib/libopencv_flann.so.3.2.0
tsr: /usr/local/lib/libopencv_objdetect.so.3.2.0
tsr: /usr/local/lib/libopencv_ml.so.3.2.0
tsr: /usr/local/lib/libopencv_highgui.so.3.2.0
tsr: /usr/local/lib/libopencv_photo.so.3.2.0
tsr: /usr/local/lib/libopencv_videoio.so.3.2.0
tsr: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
tsr: /usr/local/lib/libopencv_imgproc.so.3.2.0
tsr: /usr/local/lib/libopencv_core.so.3.2.0
tsr: CMakeFiles/tsr.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ltadl/CDS/CDS_R1_TSR/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tsr"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tsr.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tsr.dir/build: tsr

.PHONY : CMakeFiles/tsr.dir/build

CMakeFiles/tsr.dir/requires: CMakeFiles/tsr.dir/src/tsr.cpp.o.requires

.PHONY : CMakeFiles/tsr.dir/requires

CMakeFiles/tsr.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tsr.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tsr.dir/clean

CMakeFiles/tsr.dir/depend:
	cd /home/ltadl/CDS/CDS_R1_TSR/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ltadl/CDS/CDS_R1_TSR /home/ltadl/CDS/CDS_R1_TSR /home/ltadl/CDS/CDS_R1_TSR/build /home/ltadl/CDS/CDS_R1_TSR/build /home/ltadl/CDS/CDS_R1_TSR/build/CMakeFiles/tsr.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tsr.dir/depend

