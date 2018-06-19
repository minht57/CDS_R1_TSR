# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

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
CMAKE_SOURCE_DIR = /home/ubuntu/DriverlessCarChallenge_2017-2018/example

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/DriverlessCarChallenge_2017-2018/example

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/cmake-gui -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/ubuntu/DriverlessCarChallenge_2017-2018/example/CMakeFiles /home/ubuntu/DriverlessCarChallenge_2017-2018/example/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/ubuntu/DriverlessCarChallenge_2017-2018/example/CMakeFiles 0
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
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named 1

# Build rule for target.
1 : cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 1
.PHONY : 1

# fast build rule for target.
1/fast:
	$(MAKE) -f CMakeFiles/1.dir/build.make CMakeFiles/1.dir/build
.PHONY : 1/fast

#=============================================================================
# Target rules for targets named haynhove_uy

# Build rule for target.
haynhove_uy: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 haynhove_uy
.PHONY : haynhove_uy

# fast build rule for target.
haynhove_uy/fast:
	$(MAKE) -f CMakeFiles/haynhove_uy.dir/build.make CMakeFiles/haynhove_uy.dir/build
.PHONY : haynhove_uy/fast

#=============================================================================
# Target rules for targets named dlib

# Build rule for target.
dlib: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 dlib
.PHONY : dlib

# fast build rule for target.
dlib/fast:
	$(MAKE) -f dlib_build/CMakeFiles/dlib.dir/build.make dlib_build/CMakeFiles/dlib.dir/build
.PHONY : dlib/fast

#=============================================================================
# Target rules for targets named vanishing-point

# Build rule for target.
vanishing-point: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 vanishing-point
.PHONY : vanishing-point

# fast build rule for target.
vanishing-point/fast:
	$(MAKE) -f lane_detection/CMakeFiles/vanishing-point.dir/build.make lane_detection/CMakeFiles/vanishing-point.dir/build
.PHONY : vanishing-point/fast

#=============================================================================
# Target rules for targets named i2c-pwm

# Build rule for target.
i2c-pwm: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 i2c-pwm
.PHONY : i2c-pwm

# fast build rule for target.
i2c-pwm/fast:
	$(MAKE) -f peripheral_driver/i2c/CMakeFiles/i2c-pwm.dir/build.make peripheral_driver/i2c/CMakeFiles/i2c-pwm.dir/build
.PHONY : i2c-pwm/fast

#=============================================================================
# Target rules for targets named uart

# Build rule for target.
uart: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 uart
.PHONY : uart

# fast build rule for target.
uart/fast:
	$(MAKE) -f peripheral_driver/uart/CMakeFiles/uart.dir/build.make peripheral_driver/uart/CMakeFiles/uart.dir/build
.PHONY : uart/fast

#=============================================================================
# Target rules for targets named kinect-cv2

# Build rule for target.
kinect-cv2: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 kinect-cv2
.PHONY : kinect-cv2

# fast build rule for target.
kinect-cv2/fast:
	$(MAKE) -f stereo_vision/CMakeFiles/kinect-cv2.dir/build.make stereo_vision/CMakeFiles/kinect-cv2.dir/build
.PHONY : kinect-cv2/fast

#=============================================================================
# Target rules for targets named ObjectRecognition

# Build rule for target.
ObjectRecognition: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 ObjectRecognition
.PHONY : ObjectRecognition

# fast build rule for target.
ObjectRecognition/fast:
	$(MAKE) -f ObjectRecognition/CMakeFiles/ObjectRecognition.dir/build.make ObjectRecognition/CMakeFiles/ObjectRecognition.dir/build
.PHONY : ObjectRecognition/fast

#=============================================================================
# Target rules for targets named ObjectDetection

# Build rule for target.
ObjectDetection: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 ObjectDetection
.PHONY : ObjectDetection

# fast build rule for target.
ObjectDetection/fast:
	$(MAKE) -f ObjectDetection/CMakeFiles/ObjectDetection.dir/build.make ObjectDetection/CMakeFiles/ObjectDetection.dir/build
.PHONY : ObjectDetection/fast

#=============================================================================
# Target rules for targets named HAL

# Build rule for target.
HAL: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 HAL
.PHONY : HAL

# fast build rule for target.
HAL/fast:
	$(MAKE) -f HAL/CMakeFiles/HAL.dir/build.make HAL/CMakeFiles/HAL.dir/build
.PHONY : HAL/fast

#=============================================================================
# Target rules for targets named multilane

# Build rule for target.
multilane: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 multilane
.PHONY : multilane

# fast build rule for target.
multilane/fast:
	$(MAKE) -f multilane/CMakeFiles/multilane.dir/build.make multilane/CMakeFiles/multilane.dir/build
.PHONY : multilane/fast

#=============================================================================
# Target rules for targets named SignDetection

# Build rule for target.
SignDetection: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 SignDetection
.PHONY : SignDetection

# fast build rule for target.
SignDetection/fast:
	$(MAKE) -f sign_detection/CMakeFiles/SignDetection.dir/build.make sign_detection/CMakeFiles/SignDetection.dir/build
.PHONY : SignDetection/fast

#=============================================================================
# Target rules for targets named signsRecognizer

# Build rule for target.
signsRecognizer: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 signsRecognizer
.PHONY : signsRecognizer

# fast build rule for target.
signsRecognizer/fast:
	$(MAKE) -f sign_recognize/CMakeFiles/signsRecognizer.dir/build.make sign_recognize/CMakeFiles/signsRecognizer.dir/build
.PHONY : signsRecognizer/fast

#=============================================================================
# Target rules for targets named radon

# Build rule for target.
radon: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 radon
.PHONY : radon

# fast build rule for target.
radon/fast:
	$(MAKE) -f radon/CMakeFiles/radon.dir/build.make radon/CMakeFiles/radon.dir/build
.PHONY : radon/fast

#=============================================================================
# Target rules for targets named extractInfo

# Build rule for target.
extractInfo: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 extractInfo
.PHONY : extractInfo

# fast build rule for target.
extractInfo/fast:
	$(MAKE) -f extract_info/CMakeFiles/extractInfo.dir/build.make extract_info/CMakeFiles/extractInfo.dir/build
.PHONY : extractInfo/fast

#=============================================================================
# Target rules for targets named openni2

# Build rule for target.
openni2: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 openni2
.PHONY : openni2

# fast build rule for target.
openni2/fast:
	$(MAKE) -f openni2/CMakeFiles/openni2.dir/build.make openni2/CMakeFiles/openni2.dir/build
.PHONY : openni2/fast

main_pid/1.o: main_pid/1.cpp.o
.PHONY : main_pid/1.o

# target to build an object file
main_pid/1.cpp.o:
	$(MAKE) -f CMakeFiles/1.dir/build.make CMakeFiles/1.dir/main_pid/1.cpp.o
.PHONY : main_pid/1.cpp.o

main_pid/1.i: main_pid/1.cpp.i
.PHONY : main_pid/1.i

# target to preprocess a source file
main_pid/1.cpp.i:
	$(MAKE) -f CMakeFiles/1.dir/build.make CMakeFiles/1.dir/main_pid/1.cpp.i
.PHONY : main_pid/1.cpp.i

main_pid/1.s: main_pid/1.cpp.s
.PHONY : main_pid/1.s

# target to generate assembly for a file
main_pid/1.cpp.s:
	$(MAKE) -f CMakeFiles/1.dir/build.make CMakeFiles/1.dir/main_pid/1.cpp.s
.PHONY : main_pid/1.cpp.s

main_pid/haynhove_uy.o: main_pid/haynhove_uy.cpp.o
.PHONY : main_pid/haynhove_uy.o

# target to build an object file
main_pid/haynhove_uy.cpp.o:
	$(MAKE) -f CMakeFiles/haynhove_uy.dir/build.make CMakeFiles/haynhove_uy.dir/main_pid/haynhove_uy.cpp.o
.PHONY : main_pid/haynhove_uy.cpp.o

main_pid/haynhove_uy.i: main_pid/haynhove_uy.cpp.i
.PHONY : main_pid/haynhove_uy.i

# target to preprocess a source file
main_pid/haynhove_uy.cpp.i:
	$(MAKE) -f CMakeFiles/haynhove_uy.dir/build.make CMakeFiles/haynhove_uy.dir/main_pid/haynhove_uy.cpp.i
.PHONY : main_pid/haynhove_uy.cpp.i

main_pid/haynhove_uy.s: main_pid/haynhove_uy.cpp.s
.PHONY : main_pid/haynhove_uy.s

# target to generate assembly for a file
main_pid/haynhove_uy.cpp.s:
	$(MAKE) -f CMakeFiles/haynhove_uy.dir/build.make CMakeFiles/haynhove_uy.dir/main_pid/haynhove_uy.cpp.s
.PHONY : main_pid/haynhove_uy.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... 1"
	@echo "... haynhove_uy"
	@echo "... dlib"
	@echo "... vanishing-point"
	@echo "... i2c-pwm"
	@echo "... uart"
	@echo "... kinect-cv2"
	@echo "... ObjectRecognition"
	@echo "... ObjectDetection"
	@echo "... HAL"
	@echo "... multilane"
	@echo "... SignDetection"
	@echo "... signsRecognizer"
	@echo "... radon"
	@echo "... extractInfo"
	@echo "... openni2"
	@echo "... main_pid/1.o"
	@echo "... main_pid/1.i"
	@echo "... main_pid/1.s"
	@echo "... main_pid/haynhove_uy.o"
	@echo "... main_pid/haynhove_uy.i"
	@echo "... main_pid/haynhove_uy.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system
