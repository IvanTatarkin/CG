#include <veekay/veekay.hpp>
#include "sphere_generator.h"
#include "camera.h"
#include "math_utils.h"
#include "vertex.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <vector>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <limits>
#include <imgui.h>

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
    alignas(16) glm::mat4 normalMatrix;
    alignas(16) glm::mat4 lightSpaceMatrix;
    alignas(16) glm::vec4 cameraPos;       
    alignas(16) glm::vec4 ambientColor;    
};

struct MaterialData {
    alignas(16) glm::vec4 albedo;              
    alignas(16) glm::vec4 specularShininess;   
};

struct DirectionalLightData {
    alignas(16) glm::vec4 directionIntensity;  
    alignas(16) glm::vec4 color;               
};

struct PointLightData {
    alignas(16) glm::vec4 positionIntensity;   
    alignas(16) glm::vec4 colorRange;          
};

struct SpotLightData {
    alignas(16) glm::vec4 positionIntensity;   
    alignas(16) glm::vec4 directionInnerCos;   
    alignas(16) glm::vec4 colorOuterCos;       
};

struct LightCounts {
    alignas(16) glm::ivec4 counts;             
};

constexpr uint32_t kMaxPointLights = 8;
constexpr uint32_t kMaxSpotLights = 4;
constexpr const char* kDefaultTexturePath = "textures/owl.ppm";
constexpr uint32_t kShadowMapSize = 2048;

struct TextureData {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint32_t> pixels; 
};

static TextureData loadPPM(const std::filesystem::path& path) {
    TextureData out;
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open texture file");
    }
    std::string magic;
    file >> magic;
    if (magic != "P6" && magic != "P3") {
        throw std::runtime_error("PPM must be P6 or P3");
    }
    
    char c;
    file.get(c);
    while (file.peek() == '#') {
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    file >> out.width >> out.height;
    int maxval;
    file >> maxval;
    if (maxval != 255) {
        throw std::runtime_error("Only maxval=255 supported");
    }

    std::vector<uint8_t> rgb;
    rgb.resize(static_cast<size_t>(out.width) * out.height * 3);

    if (magic == "P6") {
        file.get(c); 
        size_t count = rgb.size();
        file.read(reinterpret_cast<char*>(rgb.data()), count);
    } else { 
        for (size_t i = 0; i < rgb.size(); ++i) {
            int v;
            file >> v;
            rgb[i] = static_cast<uint8_t>(v);
        }
    }

    out.pixels.resize(static_cast<size_t>(out.width) * out.height);
    for (size_t i = 0; i < out.width * out.height; ++i) {
        uint8_t r = rgb[i * 3 + 0];
        uint8_t g = rgb[i * 3 + 1];
        uint8_t b = rgb[i * 3 + 2];
        out.pixels[i] = (0xFFu << 24) | (static_cast<uint32_t>(b) << 16) | (static_cast<uint32_t>(g) << 8) | static_cast<uint32_t>(r);
    }
    return out;
}

static TextureData makeFallbackChecker(uint32_t w = 256, uint32_t h = 256, uint32_t tile = 32) {
    TextureData out;
    out.width = w;
    out.height = h;
    out.pixels.resize(static_cast<size_t>(w) * h);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            bool dark = ((x / tile) + (y / tile)) % 2 == 0;
            uint8_t c = dark ? 50 : 200;
            out.pixels[y * w + x] = (0xFFu << 24) | (static_cast<uint32_t>(c) << 16) | (static_cast<uint32_t>(c) << 8) | c;
        }
    }
    return out;
}

static void makePlaneMesh(float halfSize, float y, float uvScale,
                          std::vector<Vertex>& outVertices,
                          std::vector<uint32_t>& outIndices) {
    outVertices.clear();
    outIndices.clear();

    glm::vec3 normal(0.0f, 1.0f, 0.0f);
    glm::vec3 color(1.0f, 1.0f, 1.0f);
    outVertices.push_back(Vertex{glm::vec3(-halfSize, y, -halfSize), normal, color, glm::vec2(0.0f, 0.0f)});
    outVertices.push_back(Vertex{glm::vec3( halfSize, y, -halfSize), normal, color, glm::vec2(uvScale, 0.0f)});
    outVertices.push_back(Vertex{glm::vec3(-halfSize, y,  halfSize), normal, color, glm::vec2(0.0f, uvScale)});
    outVertices.push_back(Vertex{glm::vec3( halfSize, y,  halfSize), normal, color, glm::vec2(uvScale, uvScale)});

    outIndices = {0, 1, 2, 2, 1, 3};
}

static struct {
    std::vector<Vertex> sphereVertices;
    std::vector<uint32_t> sphereIndices;
    std::vector<Vertex> planeVertices;
    std::vector<uint32_t> planeIndices;
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;
    uint32_t planeIndexCount = 0;
    VkBuffer planeVertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory planeVertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer planeIndexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory planeIndexBufferMemory = VK_NULL_HANDLE;
    VkBuffer uniformBuffer = VK_NULL_HANDLE;
    VkDeviceMemory uniformBufferMemory = VK_NULL_HANDLE;
    VkBuffer planeUniformBuffer = VK_NULL_HANDLE;
    VkDeviceMemory planeUniformBufferMemory = VK_NULL_HANDLE;
    VkBuffer materialBuffer = VK_NULL_HANDLE;
    VkDeviceMemory materialBufferMemory = VK_NULL_HANDLE;
    VkBuffer directionalLightBuffer = VK_NULL_HANDLE;
    VkDeviceMemory directionalLightBufferMemory = VK_NULL_HANDLE;
    VkBuffer pointLightBuffer = VK_NULL_HANDLE;
    VkDeviceMemory pointLightBufferMemory = VK_NULL_HANDLE;
    VkBuffer spotLightBuffer = VK_NULL_HANDLE;
    VkDeviceMemory spotLightBufferMemory = VK_NULL_HANDLE;
    VkBuffer lightCountBuffer = VK_NULL_HANDLE;
    VkDeviceMemory lightCountBufferMemory = VK_NULL_HANDLE;
    VkShaderModule vertexShaderModule = VK_NULL_HANDLE;
    VkShaderModule fragmentShaderModule = VK_NULL_HANDLE;
    VkShaderModule shadowVertexShaderModule = VK_NULL_HANDLE;
    VkShaderModule shadowFragmentShaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkPipeline wireframePipeline = VK_NULL_HANDLE;
    VkPipeline shadowPipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSetSphere = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSetPlane = VK_NULL_HANDLE;
    veekay::graphics::Texture* texture = nullptr;
    VkSampler textureSampler = VK_NULL_HANDLE;
    VkImage shadowImage = VK_NULL_HANDLE;
    VkDeviceMemory shadowImageMemory = VK_NULL_HANDLE;
    VkImageView shadowImageView = VK_NULL_HANDLE;
    VkSampler shadowSampler = VK_NULL_HANDLE;
    bool shadowInitialized = false;
    Camera camera;
    float sphereRotationY = 0.0f;
    float sphereRotationX = 0.0f;
    bool autoRotate = true;
    bool wireframeMode = false;
    uint32_t indexCount = 0;
    float fov = 60.0f;
    float baseMoveSpeed = 3.0f;
    float mouseSensitivity = 0.15f;
    bool mouseCaptured = false;
    double lastTime = 0.0;
    glm::vec3 modelPosition = glm::vec3(0.0f);
    glm::vec3 planePosition = glm::vec3(0.0f, -1.2f, 0.0f);
    MaterialData material{
        .albedo = glm::vec4(0.7f, 0.7f, 0.9f, 1.0f),
        .specularShininess = glm::vec4(0.9f, 0.9f, 0.9f, 32.0f)
    };
    DirectionalLightData dirLight{
        .directionIntensity = glm::vec4(-0.2f, -1.0f, -0.3f, 5.0f),
        .color = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f)
    };
    std::vector<PointLightData> pointLights{};
    std::vector<SpotLightData> spotLights{
        {
            glm::vec4(0.0f, 2.5f, 0.0f, 24.0f),
            glm::vec4(0.0f, -1.0f, 0.0f, glm::cos(glm::radians(12.5f))),
            glm::vec4(1.0f, 1.0f, 0.9f, glm::cos(glm::radians(17.5f)))
        }
    };
    LightCounts lightCounts{};
    glm::vec4 ambient{0.05f, 0.05f, 0.05f, 0.5f};
} app_state;

VkShaderModule loadShaderModule(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << path << std::endl;
        return VK_NULL_HANDLE;
    }
    
    size_t size = file.tellg();
    std::vector<uint32_t> buffer(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    file.close();
    
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = size;
    info.pCode = buffer.data();
    
    VkShaderModule result;
    if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    
    return result;
}

void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, 
                  VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(veekay::app.vk_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }
    
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(veekay::app.vk_device, buffer, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(veekay::app.vk_physical_device, &memProperties);
    
    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            memoryTypeIndex = i;
            break;
        }
    }
    
    if (memoryTypeIndex == UINT32_MAX) {
        throw std::runtime_error("failed to find suitable memory type!");
    }
    
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    
    if (vkAllocateMemory(veekay::app.vk_device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }
    
    vkBindBufferMemory(veekay::app.vk_device, buffer, bufferMemory, 0);
}

VkFormat getShadowDepthFormat() {
    return VK_FORMAT_D32_SFLOAT;
}

void createDepthImage(uint32_t width, uint32_t height,
                      VkImageUsageFlags usage,
                      VkImage& image,
                      VkDeviceMemory& imageMemory,
                      VkImageView& imageView) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = getShadowDepthFormat();
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(veekay::app.vk_device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(veekay::app.vk_device, image, &memRequirements);

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(veekay::app.vk_physical_device, &memProperties);

    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memoryTypeIndex = i;
            break;
        }
    }

    if (memoryTypeIndex == UINT32_MAX) {
        throw std::runtime_error("failed to find suitable memory type for image");
    }

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;

    if (vkAllocateMemory(veekay::app.vk_device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory");
    }

    vkBindImageMemory(veekay::app.vk_device, image, imageMemory, 0);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = imageInfo.format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(veekay::app.vk_device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image view");
    }
}

void transitionDepthImage(VkCommandBuffer cmd, VkImage image,
                          VkImageLayout oldLayout, VkImageLayout newLayout,
                          VkPipelineStageFlags srcStage,
                          VkPipelineStageFlags dstStage,
                          VkAccessFlags srcAccess,
                          VkAccessFlags dstAccess) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;

    vkCmdPipelineBarrier(cmd,
                         srcStage, dstStage,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &barrier);
}

void init(VkCommandBuffer cmd) {
    std::cout << "Initializing application..." << std::endl;
    
    
    const int segments = 10;
    app_state.sphereVertices = SphereGenerator::generateSphere(1.0f, segments, glm::vec3(0.5f, 0.8f, 1.0f));
    app_state.sphereIndices = SphereGenerator::generateIndices(segments);
    app_state.indexCount = static_cast<uint32_t>(app_state.sphereIndices.size());

    makePlaneMesh(12.0f, app_state.planePosition.y, 8.0f, app_state.planeVertices, app_state.planeIndices);
    app_state.planeIndexCount = static_cast<uint32_t>(app_state.planeIndices.size());
    
    std::cout << "Generated " << app_state.sphereVertices.size() << " vertices and " 
              << app_state.sphereIndices.size() << " indices" << std::endl;
    
    if (app_state.sphereVertices.empty() || app_state.sphereIndices.empty()) {
        std::cerr << "ERROR: No geometry generated!" << std::endl;
        veekay::app.running = false;
        return;
    }
    
    app_state.camera.setDistance(3.0f);
    app_state.camera.setRotation(0.0f, 0.0f);
    
    VkDeviceSize vertexBufferSize = sizeof(app_state.sphereVertices[0]) * app_state.sphereVertices.size();
    createBuffer(vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 app_state.vertexBuffer, app_state.vertexBufferMemory);
    
    void* data;
    vkMapMemory(veekay::app.vk_device, app_state.vertexBufferMemory, 0, vertexBufferSize, 0, &data);
    memcpy(data, app_state.sphereVertices.data(), (size_t)vertexBufferSize);
    vkUnmapMemory(veekay::app.vk_device, app_state.vertexBufferMemory);
    
    VkDeviceSize indexBufferSize = sizeof(app_state.sphereIndices[0]) * app_state.sphereIndices.size();
    createBuffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 app_state.indexBuffer, app_state.indexBufferMemory);
    
    vkMapMemory(veekay::app.vk_device, app_state.indexBufferMemory, 0, indexBufferSize, 0, &data);
    memcpy(data, app_state.sphereIndices.data(), (size_t)indexBufferSize);
    vkUnmapMemory(veekay::app.vk_device, app_state.indexBufferMemory);

    VkDeviceSize planeVertexBufferSize = sizeof(app_state.planeVertices[0]) * app_state.planeVertices.size();
    createBuffer(planeVertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 app_state.planeVertexBuffer, app_state.planeVertexBufferMemory);

    vkMapMemory(veekay::app.vk_device, app_state.planeVertexBufferMemory, 0, planeVertexBufferSize, 0, &data);
    memcpy(data, app_state.planeVertices.data(), (size_t)planeVertexBufferSize);
    vkUnmapMemory(veekay::app.vk_device, app_state.planeVertexBufferMemory);

    VkDeviceSize planeIndexBufferSize = sizeof(app_state.planeIndices[0]) * app_state.planeIndices.size();
    createBuffer(planeIndexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 app_state.planeIndexBuffer, app_state.planeIndexBufferMemory);

    vkMapMemory(veekay::app.vk_device, app_state.planeIndexBufferMemory, 0, planeIndexBufferSize, 0, &data);
    memcpy(data, app_state.planeIndices.data(), (size_t)planeIndexBufferSize);
    vkUnmapMemory(veekay::app.vk_device, app_state.planeIndexBufferMemory);
    
    VkDeviceSize uniformBufferSize = sizeof(UniformBufferObject);
    createBuffer(uniformBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 app_state.uniformBuffer, app_state.uniformBufferMemory);

    createBuffer(uniformBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 app_state.planeUniformBuffer, app_state.planeUniformBufferMemory);

    VkDeviceSize materialBufferSize = sizeof(MaterialData);
    createBuffer(materialBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 app_state.materialBuffer, app_state.materialBufferMemory);

    VkDeviceSize dirLightBufferSize = sizeof(DirectionalLightData);
    createBuffer(dirLightBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 app_state.directionalLightBuffer, app_state.directionalLightBufferMemory);

    VkDeviceSize pointLightBufferSize = sizeof(PointLightData) * kMaxPointLights;
    createBuffer(pointLightBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 app_state.pointLightBuffer, app_state.pointLightBufferMemory);

    VkDeviceSize spotLightBufferSize = sizeof(SpotLightData) * kMaxSpotLights;
    createBuffer(spotLightBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 app_state.spotLightBuffer, app_state.spotLightBufferMemory);

    VkDeviceSize countBufferSize = sizeof(LightCounts);
    createBuffer(countBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 app_state.lightCountBuffer, app_state.lightCountBufferMemory);

    
    TextureData texData;
    try {
        if (std::filesystem::exists(kDefaultTexturePath)) {
            texData = loadPPM(kDefaultTexturePath);
        } else if (std::filesystem::exists("textures/checker.ppm")) {
            texData = loadPPM("textures/checker.ppm");
        } else {
            texData = makeFallbackChecker();
        }
    } catch (...) {
        texData = makeFallbackChecker();
    }
    app_state.texture = new veekay::graphics::Texture(
        cmd,
        texData.width,
        texData.height,
        VK_FORMAT_R8G8B8A8_UNORM,
        texData.pixels.data()
    );

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    if (vkCreateSampler(veekay::app.vk_device, &samplerInfo, nullptr, &app_state.textureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }

    createDepthImage(kShadowMapSize, kShadowMapSize,
                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                     app_state.shadowImage,
                     app_state.shadowImageMemory,
                     app_state.shadowImageView);

    VkSamplerCreateInfo shadowSamplerInfo{};
    shadowSamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    shadowSamplerInfo.magFilter = VK_FILTER_LINEAR;
    shadowSamplerInfo.minFilter = VK_FILTER_LINEAR;
    shadowSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shadowSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shadowSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shadowSamplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    shadowSamplerInfo.compareEnable = VK_TRUE;
    shadowSamplerInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    shadowSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    shadowSamplerInfo.minLod = 0.0f;
    shadowSamplerInfo.maxLod = 0.0f;
    shadowSamplerInfo.unnormalizedCoordinates = VK_FALSE;
    if (vkCreateSampler(veekay::app.vk_device, &shadowSamplerInfo, nullptr, &app_state.shadowSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shadow sampler!");
    }
    
    app_state.vertexShaderModule = loadShaderModule("shaders/vert.spv");
    app_state.fragmentShaderModule = loadShaderModule("shaders/frag.spv");
    app_state.shadowVertexShaderModule = loadShaderModule("shaders/shadow_vert.spv");
    app_state.shadowFragmentShaderModule = loadShaderModule("shaders/shadow_frag.spv");
    
    if (!app_state.vertexShaderModule || !app_state.fragmentShaderModule ||
        !app_state.shadowVertexShaderModule || !app_state.shadowFragmentShaderModule) {
        std::cerr << "Failed to load shaders!" << std::endl;
        veekay::app.running = false;
        return;
    }
    
    std::array<VkDescriptorSetLayoutBinding, 8> layoutBindings{};
    
    layoutBindings[0].binding = 0;
    layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBindings[0].descriptorCount = 1;
    layoutBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    
    layoutBindings[1].binding = 1;
    layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBindings[1].descriptorCount = 1;
    layoutBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    
    layoutBindings[2].binding = 2;
    layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBindings[2].descriptorCount = 1;
    layoutBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    
    layoutBindings[3].binding = 3;
    layoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings[3].descriptorCount = 1;
    layoutBindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    
    layoutBindings[4].binding = 4;
    layoutBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings[4].descriptorCount = 1;
    layoutBindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    
    layoutBindings[5].binding = 5;
    layoutBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBindings[5].descriptorCount = 1;
    layoutBindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    
    layoutBindings[6].binding = 6;
    layoutBindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    layoutBindings[6].descriptorCount = 1;
    layoutBindings[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    layoutBindings[7].binding = 7;
    layoutBindings[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    layoutBindings[7].descriptorCount = 1;
    layoutBindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    layoutInfo.pBindings = layoutBindings.data();
    layoutInfo.pBindings = layoutBindings.data();
    
    if (vkCreateDescriptorSetLayout(veekay::app.vk_device, &layoutInfo, nullptr, &app_state.descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
    
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &app_state.descriptorSetLayout;
    
    if (vkCreatePipelineLayout(veekay::app.vk_device, &pipelineLayoutInfo, nullptr, &app_state.pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }
    
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = app_state.vertexShaderModule;
    vertShaderStageInfo.pName = "main";
    
    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = app_state.fragmentShaderModule;
    fragShaderStageInfo.pName = "main";
    
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
    
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    
    std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, position);
    
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, normal);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Vertex, color);
    
    attributeDescriptions[3].binding = 0;
    attributeDescriptions[3].location = 3;
    attributeDescriptions[3].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[3].offset = offsetof(Vertex, texCoord);
    
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
    
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = 1.0f;
    viewport.height = 1.0f;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {1, 1};
    
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;
    
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;
    
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    
    VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;
    
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = app_state.pipelineLayout;
    pipelineInfo.renderPass = veekay::app.vk_render_pass;
    pipelineInfo.subpass = 0;
    
    if (vkCreateGraphicsPipelines(veekay::app.vk_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &app_state.graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }
    
    rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
    rasterizer.lineWidth = 1.5f;
    
    if (vkCreateGraphicsPipelines(veekay::app.vk_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &app_state.wireframePipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create wireframe pipeline!");
    }

    VkPipelineShaderStageCreateInfo shadowStages[2]{};
    shadowStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shadowStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shadowStages[0].module = app_state.shadowVertexShaderModule;
    shadowStages[0].pName = "main";
    shadowStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shadowStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shadowStages[1].module = app_state.shadowFragmentShaderModule;
    shadowStages[1].pName = "main";

    VkVertexInputBindingDescription shadowBinding{};
    shadowBinding.binding = 0;
    shadowBinding.stride = sizeof(Vertex);
    shadowBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription shadowAttr{};
    shadowAttr.binding = 0;
    shadowAttr.location = 0;
    shadowAttr.format = VK_FORMAT_R32G32B32_SFLOAT;
    shadowAttr.offset = offsetof(Vertex, position);

    VkPipelineVertexInputStateCreateInfo shadowVertexInput{};
    shadowVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    shadowVertexInput.vertexBindingDescriptionCount = 1;
    shadowVertexInput.pVertexBindingDescriptions = &shadowBinding;
    shadowVertexInput.vertexAttributeDescriptionCount = 1;
    shadowVertexInput.pVertexAttributeDescriptions = &shadowAttr;

    VkPipelineInputAssemblyStateCreateInfo shadowInputAssembly{};
    shadowInputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    shadowInputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    shadowInputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport shadowViewport{};
    shadowViewport.x = 0.0f;
    shadowViewport.y = 0.0f;
    shadowViewport.width = static_cast<float>(kShadowMapSize);
    shadowViewport.height = static_cast<float>(kShadowMapSize);
    shadowViewport.minDepth = 0.0f;
    shadowViewport.maxDepth = 1.0f;

    VkRect2D shadowScissor{};
    shadowScissor.offset = {0, 0};
    shadowScissor.extent = {kShadowMapSize, kShadowMapSize};

    VkPipelineViewportStateCreateInfo shadowViewportState{};
    shadowViewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    shadowViewportState.viewportCount = 1;
    shadowViewportState.pViewports = &shadowViewport;
    shadowViewportState.scissorCount = 1;
    shadowViewportState.pScissors = &shadowScissor;

    VkPipelineRasterizationStateCreateInfo shadowRaster{};
    shadowRaster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    shadowRaster.depthClampEnable = VK_FALSE;
    shadowRaster.rasterizerDiscardEnable = VK_FALSE;
    shadowRaster.polygonMode = VK_POLYGON_MODE_FILL;
    shadowRaster.lineWidth = 1.0f;
    shadowRaster.cullMode = VK_CULL_MODE_BACK_BIT;
    shadowRaster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    shadowRaster.depthBiasEnable = VK_TRUE;
    shadowRaster.depthBiasConstantFactor = 1.25f;
    shadowRaster.depthBiasClamp = 0.0f;
    shadowRaster.depthBiasSlopeFactor = 1.75f;

    VkPipelineMultisampleStateCreateInfo shadowMs{};
    shadowMs.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    shadowMs.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    shadowMs.sampleShadingEnable = VK_FALSE;

    VkPipelineDepthStencilStateCreateInfo shadowDepth{};
    shadowDepth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    shadowDepth.depthTestEnable = VK_TRUE;
    shadowDepth.depthWriteEnable = VK_TRUE;
    shadowDepth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    shadowDepth.depthBoundsTestEnable = VK_FALSE;
    shadowDepth.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo shadowBlend{};
    shadowBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    shadowBlend.attachmentCount = 0;

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 0;
    renderingInfo.pColorAttachmentFormats = nullptr;
    renderingInfo.depthAttachmentFormat = getShadowDepthFormat();
    renderingInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

    VkGraphicsPipelineCreateInfo shadowPipelineInfo{};
    shadowPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    shadowPipelineInfo.stageCount = 2;
    shadowPipelineInfo.pStages = shadowStages;
    shadowPipelineInfo.pVertexInputState = &shadowVertexInput;
    shadowPipelineInfo.pInputAssemblyState = &shadowInputAssembly;
    shadowPipelineInfo.pViewportState = &shadowViewportState;
    shadowPipelineInfo.pRasterizationState = &shadowRaster;
    shadowPipelineInfo.pMultisampleState = &shadowMs;
    shadowPipelineInfo.pDepthStencilState = &shadowDepth;
    shadowPipelineInfo.pColorBlendState = &shadowBlend;
    shadowPipelineInfo.layout = app_state.pipelineLayout;
    shadowPipelineInfo.renderPass = VK_NULL_HANDLE;
    shadowPipelineInfo.subpass = 0;
    shadowPipelineInfo.pNext = &renderingInfo;

    if (vkCreateGraphicsPipelines(veekay::app.vk_device, VK_NULL_HANDLE, 1, &shadowPipelineInfo, nullptr, &app_state.shadowPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shadow pipeline!");
    }
    
    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = 8; 
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 4; 
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[2].descriptorCount = 4;
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 2;
    
    if (vkCreateDescriptorPool(veekay::app.vk_device, &poolInfo, nullptr, &app_state.descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
    
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = app_state.descriptorPool;
    std::array<VkDescriptorSetLayout, 2> setLayouts = {app_state.descriptorSetLayout, app_state.descriptorSetLayout};
    allocInfo.descriptorSetCount = static_cast<uint32_t>(setLayouts.size());
    allocInfo.pSetLayouts = setLayouts.data();
    
    std::array<VkDescriptorSet, 2> descriptorSets{};
    if (vkAllocateDescriptorSets(veekay::app.vk_device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }
    app_state.descriptorSetSphere = descriptorSets[0];
    app_state.descriptorSetPlane = descriptorSets[1];
    
    auto writeDescriptorSet = [&](VkDescriptorSet dstSet, VkBuffer uboBuffer) {
        VkDescriptorBufferInfo uboInfo{};
        uboInfo.buffer = uboBuffer;
        uboInfo.offset = 0;
        uboInfo.range = sizeof(UniformBufferObject);

        VkDescriptorBufferInfo materialInfo{};
        materialInfo.buffer = app_state.materialBuffer;
        materialInfo.offset = 0;
        materialInfo.range = sizeof(MaterialData);

        VkDescriptorBufferInfo dirLightInfo{};
        dirLightInfo.buffer = app_state.directionalLightBuffer;
        dirLightInfo.offset = 0;
        dirLightInfo.range = sizeof(DirectionalLightData);

        VkDescriptorBufferInfo pointInfo{};
        pointInfo.buffer = app_state.pointLightBuffer;
        pointInfo.offset = 0;
        pointInfo.range = sizeof(PointLightData) * kMaxPointLights;

        VkDescriptorBufferInfo spotInfo{};
        spotInfo.buffer = app_state.spotLightBuffer;
        spotInfo.offset = 0;
        spotInfo.range = sizeof(SpotLightData) * kMaxSpotLights;

        VkDescriptorBufferInfo countsInfo{};
        countsInfo.buffer = app_state.lightCountBuffer;
        countsInfo.offset = 0;
        countsInfo.range = sizeof(LightCounts);

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = app_state.texture ? app_state.texture->view : VK_NULL_HANDLE;
        imageInfo.sampler = app_state.textureSampler;

        VkDescriptorImageInfo shadowInfo{};
        shadowInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
        shadowInfo.imageView = app_state.shadowImageView;
        shadowInfo.sampler = app_state.shadowSampler;

        std::array<VkWriteDescriptorSet, 8> descriptorWrites{};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = dstSet;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &uboInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = dstSet;
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &materialInfo;

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = dstSet;
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pBufferInfo = &dirLightInfo;

        descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[3].dstSet = dstSet;
        descriptorWrites[3].dstBinding = 3;
        descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[3].descriptorCount = 1;
        descriptorWrites[3].pBufferInfo = &pointInfo;

        descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[4].dstSet = dstSet;
        descriptorWrites[4].dstBinding = 4;
        descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[4].descriptorCount = 1;
        descriptorWrites[4].pBufferInfo = &spotInfo;

        descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[5].dstSet = dstSet;
        descriptorWrites[5].dstBinding = 5;
        descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[5].descriptorCount = 1;
        descriptorWrites[5].pBufferInfo = &countsInfo;

        descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[6].dstSet = dstSet;
        descriptorWrites[6].dstBinding = 6;
        descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[6].descriptorCount = 1;
        descriptorWrites[6].pImageInfo = &imageInfo;

        descriptorWrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[7].dstSet = dstSet;
        descriptorWrites[7].dstBinding = 7;
        descriptorWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[7].descriptorCount = 1;
        descriptorWrites[7].pImageInfo = &shadowInfo;

        vkUpdateDescriptorSets(veekay::app.vk_device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    };

    writeDescriptorSet(app_state.descriptorSetSphere, app_state.uniformBuffer);
    writeDescriptorSet(app_state.descriptorSetPlane, app_state.planeUniformBuffer);
    
    std::cout << "Initialization complete!" << std::endl;
}

void shutdown() {
    vkDestroyDescriptorPool(veekay::app.vk_device, app_state.descriptorPool, nullptr);
    vkDestroyPipeline(veekay::app.vk_device, app_state.graphicsPipeline, nullptr);
    vkDestroyPipeline(veekay::app.vk_device, app_state.wireframePipeline, nullptr);
    vkDestroyPipeline(veekay::app.vk_device, app_state.shadowPipeline, nullptr);
    vkDestroyPipelineLayout(veekay::app.vk_device, app_state.pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(veekay::app.vk_device, app_state.descriptorSetLayout, nullptr);
    vkDestroyShaderModule(veekay::app.vk_device, app_state.fragmentShaderModule, nullptr);
    vkDestroyShaderModule(veekay::app.vk_device, app_state.vertexShaderModule, nullptr);
    vkDestroyShaderModule(veekay::app.vk_device, app_state.shadowFragmentShaderModule, nullptr);
    vkDestroyShaderModule(veekay::app.vk_device, app_state.shadowVertexShaderModule, nullptr);
    if (app_state.texture) {
        delete app_state.texture;
        app_state.texture = nullptr;
    }
    if (app_state.textureSampler != VK_NULL_HANDLE) {
        vkDestroySampler(veekay::app.vk_device, app_state.textureSampler, nullptr);
    }
    if (app_state.shadowSampler != VK_NULL_HANDLE) {
        vkDestroySampler(veekay::app.vk_device, app_state.shadowSampler, nullptr);
    }
    if (app_state.shadowImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(veekay::app.vk_device, app_state.shadowImageView, nullptr);
    }
    if (app_state.shadowImage != VK_NULL_HANDLE) {
        vkDestroyImage(veekay::app.vk_device, app_state.shadowImage, nullptr);
    }
    if (app_state.shadowImageMemory != VK_NULL_HANDLE) {
        vkFreeMemory(veekay::app.vk_device, app_state.shadowImageMemory, nullptr);
    }
    vkDestroyBuffer(veekay::app.vk_device, app_state.lightCountBuffer, nullptr);
    vkFreeMemory(veekay::app.vk_device, app_state.lightCountBufferMemory, nullptr);
    vkDestroyBuffer(veekay::app.vk_device, app_state.spotLightBuffer, nullptr);
    vkFreeMemory(veekay::app.vk_device, app_state.spotLightBufferMemory, nullptr);
    vkDestroyBuffer(veekay::app.vk_device, app_state.pointLightBuffer, nullptr);
    vkFreeMemory(veekay::app.vk_device, app_state.pointLightBufferMemory, nullptr);
    vkDestroyBuffer(veekay::app.vk_device, app_state.directionalLightBuffer, nullptr);
    vkFreeMemory(veekay::app.vk_device, app_state.directionalLightBufferMemory, nullptr);
    vkDestroyBuffer(veekay::app.vk_device, app_state.materialBuffer, nullptr);
    vkFreeMemory(veekay::app.vk_device, app_state.materialBufferMemory, nullptr);
    vkDestroyBuffer(veekay::app.vk_device, app_state.uniformBuffer, nullptr);
    vkFreeMemory(veekay::app.vk_device, app_state.uniformBufferMemory, nullptr);
    vkDestroyBuffer(veekay::app.vk_device, app_state.planeUniformBuffer, nullptr);
    vkFreeMemory(veekay::app.vk_device, app_state.planeUniformBufferMemory, nullptr);
    vkDestroyBuffer(veekay::app.vk_device, app_state.indexBuffer, nullptr);
    vkFreeMemory(veekay::app.vk_device, app_state.indexBufferMemory, nullptr);
    vkDestroyBuffer(veekay::app.vk_device, app_state.vertexBuffer, nullptr);
    vkFreeMemory(veekay::app.vk_device, app_state.vertexBufferMemory, nullptr);
    vkDestroyBuffer(veekay::app.vk_device, app_state.planeIndexBuffer, nullptr);
    vkFreeMemory(veekay::app.vk_device, app_state.planeIndexBufferMemory, nullptr);
    vkDestroyBuffer(veekay::app.vk_device, app_state.planeVertexBuffer, nullptr);
    vkFreeMemory(veekay::app.vk_device, app_state.planeVertexBufferMemory, nullptr);
}

void update(double time) {
    
    float deltaTime = 0.0f;
    if (app_state.lastTime > 0.0) {
        deltaTime = static_cast<float>(time - app_state.lastTime);
    }
    app_state.lastTime = time;
    deltaTime = std::clamp(deltaTime, 0.0f, 0.1f); 

    using namespace veekay::input;

    bool rmbDown = mouse::isButtonDown(mouse::Button::right);
    if (rmbDown && !app_state.mouseCaptured) {
        mouse::setCaptured(true);
        app_state.mouseCaptured = true;
    } else if (!rmbDown && app_state.mouseCaptured) {
        mouse::setCaptured(false);
        app_state.mouseCaptured = false;
    }

    if (app_state.mouseCaptured) {
        auto md = mouse::cursorDelta();
        app_state.camera.rotate(md.x * app_state.mouseSensitivity, -md.y * app_state.mouseSensitivity);
    }

    glm::vec3 moveDir(0.0f);
    if (keyboard::isKeyDown(keyboard::Key::w)) moveDir += app_state.camera.getForward();
    if (keyboard::isKeyDown(keyboard::Key::s)) moveDir -= app_state.camera.getForward();
    if (keyboard::isKeyDown(keyboard::Key::d)) moveDir += app_state.camera.getRight();
    if (keyboard::isKeyDown(keyboard::Key::a)) moveDir -= app_state.camera.getRight();
    if (keyboard::isKeyDown(keyboard::Key::space)) moveDir += glm::vec3(0.0f, 1.0f, 0.0f);
    if (keyboard::isKeyDown(keyboard::Key::left_control)) moveDir += glm::vec3(0.0f, -1.0f, 0.0f);

    float speed = app_state.baseMoveSpeed * (keyboard::isKeyDown(keyboard::Key::left_shift) ? 2.0f : 1.0f);
    if (glm::length(moveDir) > 0.0001f && deltaTime > 0.0f) {
        moveDir = glm::normalize(moveDir);
        app_state.camera.move(moveDir * speed * deltaTime);
    }

    
    auto scroll = mouse::scrollDelta();
    app_state.fov = std::clamp(app_state.fov - scroll.y * 1.2f, 30.0f, 90.0f);

    ImGui::Begin("Controls");
    ImGui::Text("=== Camera (WASD/Space/Ctrl + RMB look) ===");
    ImGui::SliderFloat("Move speed", &app_state.baseMoveSpeed, 0.5f, 20.0f, "%.2f");
    ImGui::SliderFloat("Mouse sensitivity", &app_state.mouseSensitivity, 0.02f, 0.6f, "%.2f");
    ImGui::SliderFloat("FOV", &app_state.fov, 30.0f, 90.0f);
    glm::vec3 camPos = app_state.camera.getPosition();
    ImGui::Text("Cam pos: (%.2f, %.2f, %.2f)", camPos.x, camPos.y, camPos.z);
    
    ImGui::Separator();
    ImGui::Text("=== Sphere Rotation ===");
    ImGui::Checkbox("Auto Rotate Y", &app_state.autoRotate);
    if (!app_state.autoRotate) {
        ImGui::SliderFloat("Sphere Rotate Y", &app_state.sphereRotationY, 0.0f, 360.0f);
    } else {
        ImGui::Text("Auto rotating: %.1f deg", app_state.sphereRotationY);
    }
    ImGui::SliderFloat("Sphere Rotate X", &app_state.sphereRotationX, 0.0f, 360.0f);
    if (ImGui::Button("Reset Sphere Rotation")) {
        app_state.sphereRotationY = 0.0f;
        app_state.sphereRotationX = 0.0f;
    }
    
    ImGui::Separator();
    ImGui::Text("=== Rendering ===");
    ImGui::Checkbox("Wireframe Mode", &app_state.wireframeMode);
    ImGui::Text("(Show edges/faces)");

    ImGui::Separator();
    ImGui::Text("=== Ambient & Material ===");
    ImGui::ColorEdit3("Ambient color", &app_state.ambient.x);
    ImGui::SliderFloat("Ambient intensity", &app_state.ambient.w, 0.0f, 2.0f);
    ImGui::ColorEdit3("Albedo", &app_state.material.albedo.x);
    ImGui::ColorEdit3("Specular", &app_state.material.specularShininess.x);
    ImGui::SliderFloat("Shininess", &app_state.material.specularShininess.w, 2.0f, 128.0f);

    ImGui::Separator();
    ImGui::Text("=== Directional Light ===");
    ImGui::DragFloat3("Dir (towards scene)", &app_state.dirLight.directionIntensity.x, 0.01f, -1.0f, 1.0f);
    ImGui::ColorEdit3("Dir color", &app_state.dirLight.color.x);
    ImGui::SliderFloat("Dir intensity", &app_state.dirLight.directionIntensity.w, 0.0f, 50.0f);

    ImGui::Separator();
    ImGui::Text("=== Point Lights ===");
    int desiredPoint = static_cast<int>(app_state.pointLights.size());
    if (ImGui::SliderInt("Point count", &desiredPoint, 0, static_cast<int>(kMaxPointLights))) {
        desiredPoint = std::clamp(desiredPoint, 0, static_cast<int>(kMaxPointLights));
        PointLightData def{glm::vec4(0.0f, 1.0f, 0.0f, 15.0f), glm::vec4(1.0f, 1.0f, 1.0f, 5.0f)};
        app_state.pointLights.resize(desiredPoint, def);
    }
    for (size_t i = 0; i < app_state.pointLights.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        ImGui::Text("Point #%zu", i);
        ImGui::DragFloat3("Position", &app_state.pointLights[i].positionIntensity.x, 0.05f);
        ImGui::ColorEdit3("Color", &app_state.pointLights[i].colorRange.x);
        ImGui::SliderFloat("Intensity", &app_state.pointLights[i].positionIntensity.w, 0.0f, 50.0f);
        ImGui::SliderFloat("Range", &app_state.pointLights[i].colorRange.w, 0.0f, 20.0f);
        ImGui::Separator();
        ImGui::PopID();
    }

    ImGui::Text("=== Spot Lights ===");
    int desiredSpot = static_cast<int>(app_state.spotLights.size());
    if (ImGui::SliderInt("Spot count", &desiredSpot, 0, static_cast<int>(kMaxSpotLights))) {
        desiredSpot = std::clamp(desiredSpot, 0, static_cast<int>(kMaxSpotLights));
        SpotLightData def{
            glm::vec4(0.0f, 2.0f, 2.0f, 20.0f),
            glm::vec4(0.0f, -1.0f, 0.0f, glm::cos(glm::radians(12.5f))),
            glm::vec4(1.0f, 1.0f, 1.0f, glm::cos(glm::radians(17.5f)))
        };
        app_state.spotLights.resize(desiredSpot, def);
    }
    for (size_t i = 0; i < app_state.spotLights.size(); ++i) {
        ImGui::PushID(static_cast<int>(100 + i));
        ImGui::Text("Spot #%zu", i);
        ImGui::DragFloat3("Position", &app_state.spotLights[i].positionIntensity.x, 0.05f);
        ImGui::DragFloat3("Direction", &app_state.spotLights[i].directionInnerCos.x, 0.02f, -1.0f, 1.0f);
        ImGui::ColorEdit3("Color", &app_state.spotLights[i].colorOuterCos.x);
        float innerDeg = glm::degrees(acosf(app_state.spotLights[i].directionInnerCos.w));
        float outerDeg = glm::degrees(acosf(app_state.spotLights[i].colorOuterCos.w));
        if (ImGui::SliderFloat("Inner angle", &innerDeg, 1.0f, 45.0f)) {
            app_state.spotLights[i].directionInnerCos.w = glm::cos(glm::radians(innerDeg));
        }
        if (ImGui::SliderFloat("Outer angle", &outerDeg, 1.0f, 60.0f)) {
            outerDeg = std::max(outerDeg, innerDeg + 0.5f);
            app_state.spotLights[i].colorOuterCos.w = glm::cos(glm::radians(outerDeg));
        }
        ImGui::SliderFloat("Intensity", &app_state.spotLights[i].positionIntensity.w, 0.0f, 60.0f);
        ImGui::Separator();
        ImGui::PopID();
    }

    ImGui::Separator();
    ImGui::Text("=== Animation Info ===");
    float scale = 1.0f + 0.3f * std::sin(static_cast<float>(time));
    ImGui::Text("Scale: %.2f", scale);
    ImGui::Text("Time: %.2f", static_cast<float>(time));
    
    ImGui::End();
    
    if (app_state.autoRotate) {
        app_state.sphereRotationY = static_cast<float>(time) * 30.0f;
    }
    
    glm::mat4 sphereModel = glm::translate(glm::mat4(1.0f), app_state.modelPosition);
    sphereModel = glm::rotate(sphereModel, glm::radians(app_state.sphereRotationY), glm::vec3(0.0f, 1.0f, 0.0f));
    sphereModel = glm::rotate(sphereModel, glm::radians(app_state.sphereRotationX), glm::vec3(1.0f, 0.0f, 0.0f));
    sphereModel = glm::scale(sphereModel, glm::vec3(scale));

    glm::mat4 planeModel = glm::translate(glm::mat4(1.0f), app_state.planePosition);
    planeModel = glm::scale(planeModel, glm::vec3(1.0f));

    glm::mat4 viewMatrix = app_state.camera.getViewMatrix();

    float aspect = static_cast<float>(veekay::app.window_width) / static_cast<float>(veekay::app.window_height);
    if (aspect <= 0.0f) aspect = 1.0f;
    glm::mat4 projectionMatrix = glm::perspectiveRH_ZO(glm::radians(app_state.fov), aspect, 0.1f, 100.0f);
    projectionMatrix[1][1] *= -1.0f; // flip Y for Vulkan

    glm::vec3 lightDir = glm::normalize(-glm::vec3(app_state.dirLight.directionIntensity));
    glm::vec3 lightPos = lightDir * -8.0f + glm::vec3(0.0f, 6.0f, 0.0f);
    glm::mat4 lightView = glm::lookAt(lightPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 lightProj = glm::orthoRH_ZO(-12.0f, 12.0f, -12.0f, 12.0f, 0.1f, 30.0f);
    lightProj[1][1] *= -1.0f; // flip Y for Vulkan
    glm::mat4 lightSpaceMatrix = lightProj * lightView;

    auto writeUbo = [&](const glm::mat4& model, VkDeviceMemory memory) {
        glm::mat3 normal3 = glm::transpose(glm::inverse(glm::mat3(model)));
        glm::mat4 normalMatrix = glm::mat4(1.0f);
        normalMatrix[0] = glm::vec4(normal3[0], 0.0f);
        normalMatrix[1] = glm::vec4(normal3[1], 0.0f);
        normalMatrix[2] = glm::vec4(normal3[2], 0.0f);

        UniformBufferObject ubo{};
        ubo.model = model;
        ubo.view = viewMatrix;
        ubo.projection = projectionMatrix;
        ubo.normalMatrix = normalMatrix;
        ubo.lightSpaceMatrix = lightSpaceMatrix;
        ubo.cameraPos = glm::vec4(app_state.camera.getPosition(), 1.0f);
        ubo.ambientColor = app_state.ambient;

        void* data;
        vkMapMemory(veekay::app.vk_device, memory, 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(veekay::app.vk_device, memory);
    };

    writeUbo(sphereModel, app_state.uniformBufferMemory);
    writeUbo(planeModel, app_state.planeUniformBufferMemory);

    void* data = nullptr;
    vkMapMemory(veekay::app.vk_device, app_state.materialBufferMemory, 0, sizeof(app_state.material), 0, &data);
    memcpy(data, &app_state.material, sizeof(app_state.material));
    vkUnmapMemory(veekay::app.vk_device, app_state.materialBufferMemory);

    
    glm::vec3 dir = glm::normalize(glm::vec3(app_state.dirLight.directionIntensity));
    app_state.dirLight.directionIntensity.x = dir.x;
    app_state.dirLight.directionIntensity.y = dir.y;
    app_state.dirLight.directionIntensity.z = dir.z;
    vkMapMemory(veekay::app.vk_device, app_state.directionalLightBufferMemory, 0, sizeof(app_state.dirLight), 0, &data);
    memcpy(data, &app_state.dirLight, sizeof(app_state.dirLight));
    vkUnmapMemory(veekay::app.vk_device, app_state.directionalLightBufferMemory);

    
    std::vector<PointLightData> pointStorage(kMaxPointLights);
    size_t pCount = std::min(app_state.pointLights.size(), static_cast<size_t>(kMaxPointLights));
    for (size_t i = 0; i < pCount; ++i) {
        pointStorage[i] = app_state.pointLights[i];
    }
    vkMapMemory(veekay::app.vk_device, app_state.pointLightBufferMemory, 0,
                sizeof(PointLightData) * kMaxPointLights, 0, &data);
    memcpy(data, pointStorage.data(), sizeof(PointLightData) * kMaxPointLights);
    vkUnmapMemory(veekay::app.vk_device, app_state.pointLightBufferMemory);

    
    std::vector<SpotLightData> spotStorage(kMaxSpotLights);
    size_t sCount = std::min(app_state.spotLights.size(), static_cast<size_t>(kMaxSpotLights));
    for (size_t i = 0; i < sCount; ++i) {
        
        glm::vec3 n = glm::normalize(glm::vec3(app_state.spotLights[i].directionInnerCos));
        app_state.spotLights[i].directionInnerCos.x = n.x;
        app_state.spotLights[i].directionInnerCos.y = n.y;
        app_state.spotLights[i].directionInnerCos.z = n.z;
        spotStorage[i] = app_state.spotLights[i];
    }
    vkMapMemory(veekay::app.vk_device, app_state.spotLightBufferMemory, 0,
                sizeof(SpotLightData) * kMaxSpotLights, 0, &data);
    memcpy(data, spotStorage.data(), sizeof(SpotLightData) * kMaxSpotLights);
    vkUnmapMemory(veekay::app.vk_device, app_state.spotLightBufferMemory);

    
    app_state.lightCounts.counts = glm::ivec4(static_cast<int>(pCount), static_cast<int>(sCount), 0, 0);
    vkMapMemory(veekay::app.vk_device, app_state.lightCountBufferMemory, 0, sizeof(app_state.lightCounts), 0, &data);
    memcpy(data, &app_state.lightCounts, sizeof(app_state.lightCounts));
    vkUnmapMemory(veekay::app.vk_device, app_state.lightCountBufferMemory);
}

void render(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer) {
    vkResetCommandBuffer(commandBuffer, 0);
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkImageLayout shadowOldLayout = app_state.shadowInitialized ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags shadowSrcStage = app_state.shadowInitialized ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkAccessFlags shadowSrcAccess = app_state.shadowInitialized ? VK_ACCESS_SHADER_READ_BIT : 0;
    transitionDepthImage(
        commandBuffer,
        app_state.shadowImage,
        shadowOldLayout,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        shadowSrcStage,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        shadowSrcAccess,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    VkClearValue shadowClear{};
    shadowClear.depthStencil = {1.0f, 0};

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = app_state.shadowImageView;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue = shadowClear;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = {{0, 0}, {kShadowMapSize, kShadowMapSize}};
    renderingInfo.layerCount = 1;
    renderingInfo.viewMask = 0;
    renderingInfo.colorAttachmentCount = 0;
    renderingInfo.pColorAttachments = nullptr;
    renderingInfo.pDepthAttachment = &depthAttachment;
    renderingInfo.pStencilAttachment = nullptr;

    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    VkViewport shadowViewport{};
    shadowViewport.x = 0.0f;
    shadowViewport.y = 0.0f;
    shadowViewport.width = static_cast<float>(kShadowMapSize);
    shadowViewport.height = static_cast<float>(kShadowMapSize);
    shadowViewport.minDepth = 0.0f;
    shadowViewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &shadowViewport);

    VkRect2D shadowScissor{};
    shadowScissor.offset = {0, 0};
    shadowScissor.extent = {kShadowMapSize, kShadowMapSize};
    vkCmdSetScissor(commandBuffer, 0, 1, &shadowScissor);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app_state.shadowPipeline);

    VkBuffer shadowVb[] = {app_state.vertexBuffer};
    VkDeviceSize shadowOffsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, shadowVb, shadowOffsets);
    vkCmdBindIndexBuffer(commandBuffer, app_state.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app_state.pipelineLayout, 0, 1, &app_state.descriptorSetSphere, 0, nullptr);
    if (app_state.indexCount > 0) {
        vkCmdDrawIndexed(commandBuffer, app_state.indexCount, 1, 0, 0, 0);
    }

    VkBuffer shadowPlaneVb[] = {app_state.planeVertexBuffer};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, shadowPlaneVb, shadowOffsets);
    vkCmdBindIndexBuffer(commandBuffer, app_state.planeIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app_state.pipelineLayout, 0, 1, &app_state.descriptorSetPlane, 0, nullptr);
    if (app_state.planeIndexCount > 0) {
        vkCmdDrawIndexed(commandBuffer, app_state.planeIndexCount, 1, 0, 0, 0);
    }

    vkCmdEndRendering(commandBuffer);

    transitionDepthImage(
        commandBuffer,
        app_state.shadowImage,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT);
    app_state.shadowInitialized = true;
    
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = veekay::app.vk_render_pass;
    renderPassInfo.framebuffer = framebuffer;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = {veekay::app.window_width, veekay::app.window_height};
    
    VkClearValue clearValues[2];
    clearValues[0].color = {{0.1f, 0.1f, 0.1f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};
    renderPassInfo.clearValueCount = 2;
    renderPassInfo.pClearValues = clearValues;
    
    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(veekay::app.window_width);
    viewport.height = static_cast<float>(veekay::app.window_height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {veekay::app.window_width, veekay::app.window_height};
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    
    VkPipeline currentPipeline = app_state.wireframeMode ? app_state.wireframePipeline : app_state.graphicsPipeline;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, currentPipeline);
    
    VkBuffer vertexBuffers[] = {app_state.vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, app_state.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app_state.pipelineLayout, 0, 1, &app_state.descriptorSetSphere, 0, nullptr);
    if (app_state.indexCount > 0) {
        vkCmdDrawIndexed(commandBuffer, app_state.indexCount, 1, 0, 0, 0);
    }

    VkBuffer planeVb[] = {app_state.planeVertexBuffer};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, planeVb, offsets);
    vkCmdBindIndexBuffer(commandBuffer, app_state.planeIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app_state.pipelineLayout, 0, 1, &app_state.descriptorSetPlane, 0, nullptr);
    if (app_state.planeIndexCount > 0) {
        vkCmdDrawIndexed(commandBuffer, app_state.planeIndexCount, 1, 0, 0, 0);
    }
    
    vkCmdEndRenderPass(commandBuffer);
    vkEndCommandBuffer(commandBuffer);
}

int main() {
    veekay::ApplicationInfo appInfo{
        .init = init,
        .shutdown = shutdown,
        .update = update,
        .render = render
    };
    
    return veekay::run(appInfo);
}
