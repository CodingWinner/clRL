cmake_minimum_required(VERSION 3.10)
project(CLRL LANGUAGES CXX VERSION 1.0.1.0)

# Required packages
find_package(OpenCL CONFIG REQUIRED)
find_package(CLBlast CONFIG REQUIRED)
find_package(clEnvironment REQUIRED)

# Library
add_library(CLRL STATIC src/CLRL.cpp)
target_include_directories(CLRL PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(CLRL PUBLIC OpenCL::OpenCL PUBLIC clblast PUBLIC clEnvironment::clEnvironment)

install(TARGETS CLRL EXPORT CLRLTargets DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT CLRLTargets
    FILE CLRL.cmake
    NAMESPACE CLRL::
    DESTINATION lib/cmake/CLRL
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/CLRLConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CLRLConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/CLRLConfig.cmake"
    INSTALL_DESTINATION install/lib/cmake/CLRL
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CLRLConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CLRLConfigVersion.cmake"
    DESTINATION install/lib/cmake/CLRL
)