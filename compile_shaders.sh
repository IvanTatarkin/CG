#!/bin/bash

# Compile GLSL shaders to SPIR-V
# Requires glslc (from Vulkan SDK) or glslangValidator

if command -v glslc &> /dev/null; then
    echo "Using glslc to compile shaders..."
    glslc -fshader-stage=vertex shaders/vert.glsl -o shaders/vert.spv
    glslc -fshader-stage=fragment shaders/frag.glsl -o shaders/frag.spv
    glslc -fshader-stage=vertex shaders/shadow.vert -o shaders/shadow_vert.spv
    glslc -fshader-stage=fragment shaders/shadow.frag -o shaders/shadow_frag.spv
elif command -v glslangValidator &> /dev/null; then
    echo "Using glslangValidator to compile shaders..."
    glslangValidator -V shaders/vert.glsl -o shaders/vert.spv
    glslangValidator -V shaders/frag.glsl -o shaders/frag.spv
    glslangValidator -V shaders/shadow.vert -o shaders/shadow_vert.spv
    glslangValidator -V shaders/shadow.frag -o shaders/shadow_frag.spv
else
    echo "Error: Neither glslc nor glslangValidator found!"
    echo "Please install Vulkan SDK or glslangValidator"
    exit 1
fi

echo "Shaders compiled successfully!"

