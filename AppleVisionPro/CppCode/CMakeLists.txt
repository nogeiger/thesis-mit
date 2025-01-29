cmake_minimum_required(VERSION 3.2)

project(avp_stream)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Boost libraries
find_package(Boost REQUIRED COMPONENTS filesystem system thread date_time chrono)

# Find Python3
find_package(Python3 REQUIRED COMPONENTS Development)

# Qt CoConcurrent
find_package(Qt5 REQUIRED COMPONENTS Core Concurrent)

# Include directories
set(includeDir
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../Desktop/Explicit-FRI/Libraries/Explicit-cpp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../Desktop/Explicit-FRI/Libraries/myFRIClient/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../Desktop/Explicit-FRI/Libraries/Explicit-cpp/Eigen
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../Desktop/Explicit-FRI/Libraries/FTSensor/include
  ${Python3_INCLUDE_DIRS} # Include Python headers dynamically
  /home/newman_lab/.local/lib/python3.10/site-packages/pybind11/include # Add Pybind11 here
)
include_directories(${includeDir})

# Make all source files visible in the IDE
set(sourceFiles "")
file(GLOB_RECURSE sourceFiles ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Exclude CMake auto-generated files
list(FILTER sourceFiles EXCLUDE REGEX "CMakeFiles/.*CompilerId.*")

# Add executable
add_executable(${PROJECT_NAME} ${sourceFiles})

# Linking
set(EXPLICIT_CPP_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../../../Libraries/Explicit-cpp/lib/libExplicit-cpp.so" CACHE STRING "Explicit-cpp library")
set(FRI_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../../../Libraries/myFRIClient/lib/libFRIClient.a" CACHE STRING "FRI library")
set(SENSOR_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../../../Libraries/FTSensor/lib/libFTSensor.so" CACHE STRING "FTSensor library")

set(linkLibraries
  ${EXPLICIT_CPP_LIBRARY}
  ${FRI_LIBRARY}
  ${SENSOR_LIBRARY}
  ${Boost_LIBRARIES}
  Python3::Python # Link Python libraries dynamically
)

# Link libraries to the executable
target_link_libraries(${PROJECT_NAME} PRIVATE ${linkLibraries})
target_link_libraries(${PROJECT_NAME} PRIVATE Qt5::Core Qt5::Concurrent)

# The output directory
set_target_properties(${PROJECT_NAME}
    PROPERTIES
    DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../build/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../build/Release"
)

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include
  FILES_MATCHING PATTERN "*.h"
)

# Install library
set(libOutputDir ${CMAKE_INSTALL_PREFIX}/lib)
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${libOutputDir}
        ARCHIVE DESTINATION ${libOutputDir}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/application
)
