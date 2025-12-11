cmake_minimum_required(VERSION 3.20)
project(Lab1_3DGraphics)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(Vulkan REQUIRED)

# Veekay library (local copy from MMVlasko/cg)
include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)

FetchContent_Declare(
    vk-bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
    GIT_TAG v1.4.321
)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui
    GIT_TAG v1.92.3
)

set(GLFW_LIBRARY_TYPE STATIC)
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)

FetchContent_MakeAvailable(glfw vk-bootstrap imgui)

# Build Veekay library
add_library(veekay 
    src/veekay/veekay.cpp
    src/veekay/input.cpp
    src/veekay/graphics.cpp
)

target_include_directories(veekay PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(veekay PRIVATE
    glfw
    Vulkan::Vulkan
    vk-bootstrap::vk-bootstrap
)

# Link ImGui sources to Veekay
target_sources(veekay PRIVATE
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
)

set_target_properties(veekay PROPERTIES 
    CXX_STANDARD_REQUIRED TRUE 
    CXX_STANDARD 20
)

if(LINUX)
    target_compile_options(veekay PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
endif()

# ImGUI - try to find as package first, otherwise use subdirectory
find_package(imgui QUIET)
if(NOT imgui_FOUND)
    if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/imgui")
        # ImGUI doesn't have CMakeLists.txt by default, we'll add sources manually
        set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui)
        include_directories(${IMGUI_DIR})
        include_directories(${IMGUI_DIR}/backends)
        
        # Add ImGUI sources
        file(GLOB IMGUI_SOURCES
            "${IMGUI_DIR}/*.cpp"
            "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
            "${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp"
        )
        add_library(imgui STATIC ${IMGUI_SOURCES})
        target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
        target_link_libraries(imgui PUBLIC glfw ${Vulkan_LIBRARIES})
    else()
        message(FATAL_ERROR "ImGUI not found. Please clone it: git clone https://github.com/ocornut/imgui.git third_party/imgui")
    endif()
endif()

# GLM - header-only library
find_path(GLM_INCLUDE_DIR glm/glm.hpp)
if(NOT GLM_INCLUDE_DIR)
    message(FATAL_ERROR "GLM not found. Please install libglm-dev")
endif()

# Include directories
include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${GLM_INCLUDE_DIR})
include_directories(include)

# Source files (упрощенная версия без vulkan_renderer.cpp)
set(SOURCES
    src/main.cpp
    src/cylinder_generator.cpp
    src/sphere_generator.cpp
    src/camera.cpp
    src/math_utils.cpp
)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link Veekay library (Veekay includes GLFW and ImGUI)
target_link_libraries(${PROJECT_NAME} PRIVATE veekay)

# Copy shaders to build directory
set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")

# Copy SPIR-V shaders next to the executable on every build so the app can be run
# from any working directory (e.g. from build/).
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/shaders/vert.spv"
        "${CMAKE_SOURCE_DIR}/shaders/frag.spv"
        "${CMAKE_SOURCE_DIR}/shaders/shadow_vert.spv"
        "${CMAKE_SOURCE_DIR}/shaders/shadow_frag.spv"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders"
    VERBATIM
)

# Copy textures next to the executable so relative paths work from build/.
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${PROJECT_NAME}>/textures"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/textures/owl.ppm"
        "${CMAKE_SOURCE_DIR}/textures/checker.ppm"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/textures"
    VERBATIM
)

# Compiler-specific options
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
endif()

