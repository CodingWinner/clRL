# Find required packages

find_package(OpenCL CONFIG REQUIRED)
find_package(CLBlast CONFIG REQUIRED)
find_package(clEnvironment REQUIRED)

# Macro options
option(CREATE_SHARED_LIB "Create a shared or static library (off = static, on = shared)" OFF)

# Create lib
if (CREATE_SHARED_LIB)
	add_library(clRL SHARED clRL.cpp)
else()
	add_library(clRL STATIC clRL.cpp)
endif()

set(INCLUDE_PATH include)
if (NOT EXISTS ${INCLUDE_PATH})
    set(INCLUDE_PATH install/include)
endif()

target_link_libraries(clRL PUBLIC OpenCL::OpenCL PUBLIC clblast PUBLIC clEnvironment::clEnvironment)
target_include_directories(clRL PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_PATH}>
    $<INSTALL_INTERFACE:${INCLUDE_PATH}>)

install(TARGETS clRL EXPORT clRLTargets DESTINATION install/lib)
install(DIRECTORY include/ DESTINATION install/include)

install(EXPORT clRLTargets
    FILE clRL.cmake
    NAMESPACE clRL::
    DESTINATION install/lib/cmake/clRL
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/clRLConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/clRLConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/clRLConfig.cmake"
    INSTALL_DESTINATION install/lib/cmake/clRL
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/clRLConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/clRLConfigVersion.cmake"
    DESTINATION install/lib/cmake/clRL
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET clRL PROPERTY CXX_STANDARD 20)
endif()