cmake_minimum_required(VERSION 2.8)

project( CarControl-0.3 )

find_package( OpenCV REQUIRED )

set( OPENNI2_INCLUDE /home/ubuntu/Downloads/2-Linux/OpenNI-Linux-Arm-2.3/Include)
set( OPENNI2_REDIST  /home/ubuntu/Downloads/2-Linux/OpenNI-Linux-Arm-2.3/Redist)
include_directories( ${OPENNI2_INCLUDE} )

link_directories ( ${OPENNI2_REDIST} )
#Add Round1 code
include(CDS_R1_TSR/dlib/dlib/cmake)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
               STRINGS "Debug" "Release")
endif()
IF(MSVC)
  ADD_DEFINITIONS(-DNOMINMAX)
  # Another MSVC quirk. http://stackoverflow.com/questions/6832666/lnk2019-when-including-asio-headers-solution-generated-with-cmake
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP") # For multi-threaded compilation on MSVC
ELSE()
  IF(APPLE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++") # C++11 support on Apple clang
  ELSE()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")# C++11 support on GCC
  ENDIF()
ENDIF()

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
#Add Round1 code
set(DLIB_NO_GUI_SUPPORT OFF) 

include_directories(
      	${OpenCV_INCLUDE_DIRS}
      	${CMAKE_CURRENT_SOURCE_DIR}/stereo_vision
      	${CMAKE_CURRENT_SOURCE_DIR}/lane_detection
      	${CMAKE_CURRENT_SOURCE_DIR}/lane_detection/msac
      	${CMAKE_CURRENT_SOURCE_DIR}/peripheral_driver/i2c
      	${CMAKE_CURRENT_SOURCE_DIR}/peripheral_driver/uart
      	${CMAKE_CURRENT_SOURCE_DIR}/ObjectDetection
      	${CMAKE_CURRENT_SOURCE_DIR}/ObjectRecognition 
      	${CMAKE_CURRENT_SOURCE_DIR}/HAL	
      	${CMAKE_CURRENT_SOURCE_DIR}/multilane
      	${CMAKE_CURRENT_SOURCE_DIR}/include
      	${CMAKE_CURRENT_SOURCE_DIR}/sign_detection
      	${CMAKE_CURRENT_SOURCE_DIR}/sign_recognize
     	  ${CMAKE_CURRENT_SOURCE_DIR}/radon
	      ${CMAKE_CURRENT_SOURCE_DIR}/extract_info
      	${CMAKE_CURRENT_SOURCE_DIR}/openni2
      	
)

link_directories( ${LIB_FREENECT_INSTALL_DIR}/lib
                  ${LIBRARY_OUTPUT_PATH}
)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("Find OpenMP: ${OpenMP_C_FLAGS}, ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_subdirectory(lane_detection)
add_subdirectory(peripheral_driver)
add_subdirectory(stereo_vision)
add_subdirectory(ObjectRecognition)
add_subdirectory(ObjectDetection) 
add_subdirectory(HAL)
add_subdirectory(multilane)
add_subdirectory(sign_detection)
add_subdirectory(sign_recognize)
add_subdirectory(radon)
add_subdirectory(extract_info)
add_subdirectory(openni2)

#set( exec_name8 main_car)
#set( source_files8
#    main_pid/Uy_maincar.cpp
#)
#add_executable( ${exec_name8} ${source_files8} )
#target_link_libraries( ${exec_name8} dlib openni2 SignDetection signsRecognizer radon extractInfo ObjectRecognition vanishing-point i2c-pwm uart kinect-cv2 OpenNI2 HAL multilane ${OpenCV_LIBS})

#set( exec_name1 main_car_2)
#set( source_files1
#    main_pid/Uy_maincar_2.cpp
#)
#add_executable( ${exec_name1} ${source_files1} )
#target_link_libraries( ${exec_name1} dlib openni2 SignDetection signsRecognizer radon extractInfo ObjectRecognition vanishing-point i2c-pwm uart kinect-cv2 OpenNI2 HAL multilane ${OpenCV_LIBS})


# set( exec_name2 lmao)
# set( source_files2
#     main_pid/lmao.cpp
# )
# add_executable( ${exec_name2} ${source_files2} )
# target_link_libraries( ${exec_name2} dlib openni2 SignDetection signsRecognizer radon extractInfo ObjectRecognition vanishing-point i2c-pwm uart kinect-cv2 OpenNI2 HAL multilane ${OpenCV_LIBS})


# set( exec_name3 hiden)
# set( source_files3
#     main_pid/thanh.cpp
# )
# add_executable( ${exec_name3} ${source_files3} )
# target_link_libraries( ${exec_name3} dlib openni2 SignDetection signsRecognizer radon extractInfo ObjectRecognition vanishing-point i2c-pwm uart kinect-cv2 OpenNI2 HAL multilane ${OpenCV_LIBS})

set( exec_name4 haynhove_uy)
set( source_files4
    main_pid/haynhove_uy.cpp
)
add_executable( ${exec_name4} ${source_files4} )
target_link_libraries( ${exec_name4} dlib openni2 SignDetection signsRecognizer i2c-pwm kinect-cv2 OpenNI2 HAL multilane ${OpenCV_LIBS})

set( exec_name4 1)
set( source_files4
    main_pid/1.cpp
)
add_executable( ${exec_name4} ${source_files4} )
target_link_libraries( ${exec_name4} dlib ${OpenCV_LIBS})