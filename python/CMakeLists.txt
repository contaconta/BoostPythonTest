project(BoostNumpyTest)
cmake_minimum_required(VERSION 2.8)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(SRC_DIR "src")

###############################
# set flags
###############################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wnon-virtual-dtor -Woverloaded-virtual")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3 -msse4.1 -msse4.2 -mavx -mavx2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

#if(${CMAKE_BUILD_TYPE} MATCHES "Release")

#endif()

# for Boost.Log
add_definitions("-DBOOST_LOG_DYN_LINK")

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

###############################
# set output path
###############################

set(LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/lib" CACHE PATH
    "Output directory for library" )
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin" CACHE PATH
   "Output directory for executables" )

include_directories("src")

######################
# boost
######################
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(Boost_INCLUDE_DIR "/usr/local/boost/include")
  set(Boost_LIBRARY_DIR "/usr/local/boost/lib")
endif()

FIND_PACKAGE(Boost COMPONENTS serialization filesystem thread program_options system date_time chrono log timer python REQUIRED)
if (Boost_FOUND)
  message(STATUS "Found 'boost library'")
  include_directories(${Boost_INCLUDE_DIR})
  message(STATUS "  boost lib dir: ${Boost_LIBRARY_DIRS}")
  message(STATUS "  boost include dir: ${Boost_INCLUDE_DIR}")
  message(STATUS "  boost link:")
  foreach(var ${Boost_LIBRARIES})
    message(STATUS "    ${var}")
endforeach(var)
else()
  message(FATAL_ERROR "Couldn't find Boost library.")
endif()


##########################
# opencv
##########################
#set(OpenCV_DIR "/usr/local/opencv/share/OpenCV/OpenCVConfig.cmake"
    #    CACHE PATH "The path where OpenCVConfig.cmake is placed")
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
  message(STATUS "Found 'opencv library' ${OpenCV_VERSION}")
else()
  message(FATAL_ERROR "Couldn't find OpenCV")
endif()


##########################
# python boost.python
##########################
#FIND_PACKAGE(PythonLibs REQUIRED)
set(PYTHON_INCLUDE_DIRS "/usr/local/Cellar/python/2.7.6_1/Frameworks/Python.framework/Versions/Current/include/python2.7/")
set(PYTHON_LIBRARIES "/usr/local/Cellar/python/2.7.6_1/Frameworks/Python.framework/Versions/Current/lib/libpython2.7.dylib")
include_directories(${PYTHON_INCLUDE_DIRS})
message(STATUS " python includes dir: ${PYTHON_INCLUDE_DIRS}")
message(STATUS " python libs: ${PYTHON_LIBRARIES}")

##########################
# Boost.Numpy
##########################
set(BOOST_NUMPY_PATH "/usr/local/boost_numpy")
find_library(BoostNumpy 
             NAMES boost_numpy
             HINTS "${BOOST_NUMPY_PATH}/lib/")
if(NOT BoostNumpy)
    message(SEND_ERROR "Boost.Numpy is not found")
endif()
message(STATUS "Boost.Numpy is found at: ${BoostNumpy}")
include_directories("${BOOST_NUMPY_PATH}/include")

##########################
# Eigen
##########################
set(Eigen_INCLUDE_DIR "/usr/local/include/eigen3")
include_directories(${Eigen_INCLUDE_DIR})

##########################
# Lib
##########################
set(numpytest_src src/numpytest.cpp)

add_library(numpytest MODULE ${numpytest_src})
target_link_libraries(numpytest ${Boost_LIBRARIES})
target_link_libraries(numpytest ${OpenCV_LIBS})
target_link_libraries(numpytest ${PYTHON_LIBRARIES} ${BoostNumpy})


