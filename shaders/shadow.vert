#version 450

layout(location = 0) in vec3 inPosition;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 normalMatrix;
    mat4 lightSpaceMatrix;
    vec4 cameraPos;
    vec4 ambientColor;
} ubo;

void main() {
    gl_Position = ubo.lightSpaceMatrix * ubo.model * vec4(inPosition, 1.0);
}

