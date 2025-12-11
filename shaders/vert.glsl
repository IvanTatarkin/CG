#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out vec2 fragUV;
layout(location = 4) out vec4 fragPosLightSpace;

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
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    fragPos = worldPos.xyz;
    fragNormal = normalize((ubo.normalMatrix * vec4(inNormal, 0.0)).xyz);
    fragColor = inColor;
    fragUV = inTexCoord;
    fragPosLightSpace = ubo.lightSpaceMatrix * worldPos;
    gl_Position = ubo.projection * ubo.view * worldPos;
}

