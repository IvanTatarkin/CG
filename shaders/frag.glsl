#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 normalMatrix;
    vec4 cameraPos;
    vec4 ambientColor;   // rgb + intensity in w
} ubo;

layout(binding = 1) uniform Material {
    vec4 albedo;
    vec4 specularShininess; // rgb + shininess in w
} material;

layout(binding = 2) uniform DirectionalLight {
    vec4 directionIntensity; // xyz dir (towards scene), w intensity
    vec4 color;
} dirLight;

struct PointLight {
    vec4 positionIntensity; // xyz position, w intensity
    vec4 colorRange;        // rgb color, w range (optional)
};

struct SpotLight {
    vec4 positionIntensity; // xyz position, w intensity
    vec4 directionInnerCos; // xyz direction, w inner cos
    vec4 colorOuterCos;     // rgb color, w outer cos
};

layout(std430, binding = 3) readonly buffer PointLightBuffer {
    PointLight pointLights[];
};

layout(std430, binding = 4) readonly buffer SpotLightBuffer {
    SpotLight spotLights[];
};

layout(binding = 5) uniform LightCounts {
    ivec4 counts; // x: point, y: spot
} lightCounts;

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, float intensity, vec3 lightColor) {
    float diff = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), material.specularShininess.w);
    vec3 base = material.albedo.rgb * fragColor;
    vec3 diffuse = base * diff;
    vec3 specular = material.specularShininess.rgb * spec;
    return (diffuse + specular) * lightColor * intensity;
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(ubo.cameraPos.xyz - fragPos);
    vec3 color = ubo.ambientColor.xyz * ubo.ambientColor.w * material.albedo.rgb * fragColor;

    // Directional light
    vec3 Ld = normalize(-dirLight.directionIntensity.xyz);
    color += blinnPhong(N, V, Ld, dirLight.directionIntensity.w, dirLight.color.rgb);

    // Point lights
    int pc = lightCounts.counts.x;
    for (int i = 0; i < pc; ++i) {
        vec3 toLight = pointLights[i].positionIntensity.xyz - fragPos;
        float dist2 = max(dot(toLight, toLight), 0.0001);
        float dist = sqrt(dist2);
        vec3 L = toLight / dist;
        float attenuation = pointLights[i].positionIntensity.w / dist2;
        if (pointLights[i].colorRange.w > 0.0) {
            float rangeAtten = clamp(1.0 - dist / pointLights[i].colorRange.w, 0.0, 1.0);
            attenuation *= rangeAtten;
        }
        color += blinnPhong(N, V, L, attenuation, pointLights[i].colorRange.rgb);
    }

    // Spot lights
    int sc = lightCounts.counts.y;
    for (int i = 0; i < sc; ++i) {
        vec3 lightToFrag = fragPos - spotLights[i].positionIntensity.xyz;
        float dist2 = max(dot(lightToFrag, lightToFrag), 0.0001);
        float dist = sqrt(dist2);
        vec3 L = -lightToFrag / dist; // direction from fragment to light

        vec3 spotDir = normalize(spotLights[i].directionInnerCos.xyz);
        float cosTheta = dot(-L, spotDir); // angle between light forward and frag direction
        float inner = spotLights[i].directionInnerCos.w;
        float outer = spotLights[i].colorOuterCos.w;
        float angleAtten = clamp((cosTheta - outer) / max(inner - outer, 0.0001), 0.0, 1.0);

        float attenuation = spotLights[i].positionIntensity.w / dist2;
        attenuation *= angleAtten;

        color += blinnPhong(N, V, L, attenuation, spotLights[i].colorOuterCos.rgb);
    }

    outColor = vec4(color, 1.0);
}

