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

static PFN_vkCmdBeginRendering pfnCmdBeginRendering = nullptr;
static PFN_vkCmdEndRendering pfnCmdEndRendering = nullptr;
struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
    alignas(16) glm::mat4 normalMatrix;
    alignas(16) glm::mat4 lightViewProj;
    alignas(16) glm::vec4 cameraPos;       // xyz: position, w: unused
    alignas(16) glm::vec4 ambientColor;    // rgb: ambient color, w: intensity
};

struct MaterialData {
    alignas(16) glm::vec4 albedo;              // rgb, w unused
    alignas(16) glm::vec4 specularShininess;   // rgb specular, w shininess
};

struct DirectionalLightData {
    alignas(16) glm::vec4 directionIntensity;  // xyz dir (towards light), w intensity
    alignas(16) glm::vec4 color;               // rgb color, w unused
};

struct PointLightData {
    alignas(16) glm::vec4 positionIntensity;   // xyz position, w intensity
    alignas(16) glm::vec4 colorRange;          // rgb color, w range (optional falloff clamp)
};

struct SpotLightData {
    alignas(16) glm::vec4 positionIntensity;   // xyz position, w intensity
    alignas(16) glm::vec4 directionInnerCos;   // xyz direction, w inner cos
    alignas(16) glm::vec4 colorOuterCos;       // rgb color, w outer cos
};

struct LightCounts {
    alignas(16) glm::ivec4 counts;             // x: point count, y: spot count
};

constexpr uint32_t kMaxPointLights = 8;
constexpr uint32_t kMaxSpotLights = 4;
constexpr const char* kDefaultTexturePath = "textures/owl.ppm";

struct TextureData {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint32_t> pixels; // RGBA8
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
    // Skip comments
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
        file.get(c); // consume single whitespace
        size_t count = rgb.size();
        file.read(reinterpret_cast<char*>(rgb.data()), count);
    } else { // P3 ascii
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

static struct {
    std::vector<Vertex> sphereVertices;
    std::vector<uint32_t> sphereIndices;
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;
    VkBuffer uniformBuffer = VK_NULL_HANDLE;
    VkDeviceMemory uniformBufferMemory = VK_NULL_HANDLE;
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
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    veekay::graphics::Texture* texture = nullptr;
    VkSampler textureSampler = VK_NULL_HANDLE;
    VkImage shadowImage = VK_NULL_HANDLE;
    VkDeviceMemory shadowImageMemory = VK_NULL_HANDLE;
    VkImageView shadowImageView = VK_NULL_HANDLE;
    VkSampler shadowSampler = VK_NULL_HANDLE;
    VkFormat shadowFormat = VK_FORMAT_D32_SFLOAT;
    VkExtent2D shadowExtent{1024, 1024};
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
    MaterialData material{
        .albedo = glm::vec4(0.7f, 0.7f, 0.9f, 1.0f),
        .specularShininess = glm::vec4(0.9f, 0.9f, 0.9f, 32.0f)
    };
    DirectionalLightData dirLight{
        .directionIntensity = glm::vec4(-0.2f, -1.0f, -0.3f, 1.0f),
        .color = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f)
    };
    std::vector<PointLightData> pointLights{
        {glm::vec4(2.0f, 1.5f, 2.0f, 20.0f), glm::vec4(1.0f, 0.8f, 0.7f, 6.0f)},
        {glm::vec4(-2.0f, 1.0f, -1.5f, 16.0f), glm::vec4(0.6f, 0.8f, 1.0f, 5.0f)},
    };
    std::vector<SpotLightData> spotLights{
        {
            glm::vec4(0.0f, 2.5f, 0.0f, 24.0f),
            glm::vec4(0.0f, -1.0f, 0.0f, glm::cos(glm::radians(12.5f))),
            glm::vec4(1.0f, 1.0f, 0.9f, glm::cos(glm::radians(17.5f)))
        }
    };
    LightCounts lightCounts{};
    glm::vec4 ambient{0.08f, 0.08f, 0.1f, 1.0f};
    glm::mat4 cachedView{1.0f};
    glm::mat4 cachedProj{1.0f};
    glm::mat4 cachedLightVP{1.0f};
    float cachedScale = 1.0f;
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

void init(VkCommandBuffer cmd) {
    std::cout << "Initializing application..." << std::endl;
    
    // Генерация сферы (возвращаем как в исходной версии)
    const int segments = 10;
    app_state.sphereVertices = SphereGenerator::generateSphere(1.0f, segments, glm::vec3(0.5f, 0.8f, 1.0f));
    app_state.sphereIndices = SphereGenerator::generateIndices(segments);
    app_state.indexCount = static_cast<uint32_t>(app_state.sphereIndices.size());

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

    
    VkDeviceSize uniformBufferSize = sizeof(UniformBufferObject);
    createBuffer(uniformBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 app_state.uniformBuffer, app_state.uniformBufferMemory);

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

    // Shadow depth image
    {
        VkImageCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = app_state.shadowFormat,
            .extent = {app_state.shadowExtent.width, app_state.shadowExtent.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        if (vkCreateImage(veekay::app.vk_device, &info, nullptr, &app_state.shadowImage) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shadow image");
        }

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(veekay::app.vk_device, app_state.shadowImage, &req);

        VkPhysicalDeviceMemoryProperties props{};
        vkGetPhysicalDeviceMemoryProperties(veekay::app.vk_physical_device, &props);
        uint32_t idx = UINT32_MAX;
        for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
            if ((req.memoryTypeBits & (1u << i)) &&
                (props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                idx = i;
                break;
            }
        }
        if (idx == UINT32_MAX) throw std::runtime_error("no mem type for shadow image");

        VkMemoryAllocateInfo ai{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = req.size,
            .memoryTypeIndex = idx,
        };
        if (vkAllocateMemory(veekay::app.vk_device, &ai, nullptr, &app_state.shadowImageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to alloc shadow image mem");
        }
        vkBindImageMemory(veekay::app.vk_device, app_state.shadowImage, app_state.shadowImageMemory, 0);

        VkImageViewCreateInfo vi{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = app_state.shadowImage,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = app_state.shadowFormat,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        if (vkCreateImageView(veekay::app.vk_device, &vi, nullptr, &app_state.shadowImageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shadow view");
        }

        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        si.unnormalizedCoordinates = VK_FALSE;
        si.compareEnable = VK_FALSE; // manual compare
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        if (vkCreateSampler(veekay::app.vk_device, &si, nullptr, &app_state.shadowSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shadow sampler");
        }

    VkImageMemoryBarrier toDepth{};
    toDepth.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toDepth.srcAccessMask = 0;
    toDepth.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    toDepth.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toDepth.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    toDepth.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDepth.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDepth.image = app_state.shadowImage;
    toDepth.subresourceRange = vi.subresourceRange;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                             0,
                             0, nullptr,
                             0, nullptr,
                             1, &toDepth);
    }

    // === Texture ===
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
    
    pfnCmdBeginRendering = reinterpret_cast<PFN_vkCmdBeginRendering>(vkGetDeviceProcAddr(veekay::app.vk_device, "vkCmdBeginRendering"));
    pfnCmdEndRendering = reinterpret_cast<PFN_vkCmdEndRendering>(vkGetDeviceProcAddr(veekay::app.vk_device, "vkCmdEndRendering"));
    if (!pfnCmdBeginRendering || !pfnCmdEndRendering) {
        std::cerr << "Dynamic rendering not supported (vkCmdBeginRendering missing)" << std::endl;
        veekay::app.running = false;
        return;
    }
    
    std::array<VkDescriptorSetLayoutBinding, 8> layoutBindings{};
    // Binding 0: matrices + camera/ambient
    layoutBindings[0].binding = 0;
    layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBindings[0].descriptorCount = 1;
    layoutBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    // Binding 1: material
    layoutBindings[1].binding = 1;
    layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBindings[1].descriptorCount = 1;
    layoutBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Binding 2: directional light
    layoutBindings[2].binding = 2;
    layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBindings[2].descriptorCount = 1;
    layoutBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Binding 3: point lights SSBO
    layoutBindings[3].binding = 3;
    layoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings[3].descriptorCount = 1;
    layoutBindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Binding 4: spot lights SSBO
    layoutBindings[4].binding = 4;
    layoutBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings[4].descriptorCount = 1;
    layoutBindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Binding 5: light counts
    layoutBindings[5].binding = 5;
    layoutBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBindings[5].descriptorCount = 1;
    layoutBindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Binding 6: texture sampler
    layoutBindings[6].binding = 6;
    layoutBindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    layoutBindings[6].descriptorCount = 1;
    layoutBindings[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Binding 7: shadow map
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
    
    // Shadow pipeline (depth-only, dynamic rendering)
    VkPipelineShaderStageCreateInfo shadowStages[2]{};
    shadowStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shadowStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shadowStages[0].module = app_state.shadowVertexShaderModule;
    shadowStages[0].pName = "main";
    shadowStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shadowStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shadowStages[1].module = app_state.shadowFragmentShaderModule;
    shadowStages[1].pName = "main";

    VkPipelineRasterizationStateCreateInfo shadowRaster = rasterizer;
    shadowRaster.polygonMode = VK_POLYGON_MODE_FILL;
    shadowRaster.cullMode = VK_CULL_MODE_BACK_BIT;
    shadowRaster.depthBiasEnable = VK_TRUE;
    shadowRaster.depthBiasConstantFactor = 1.25f;
    shadowRaster.depthBiasSlopeFactor = 1.75f;

    VkPipelineDepthStencilStateCreateInfo shadowDepth = depthStencil;
    shadowDepth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendStateCreateInfo shadowBlend{};
    shadowBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    shadowBlend.attachmentCount = 0;

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 0;
    renderingInfo.depthAttachmentFormat = app_state.shadowFormat;

    VkGraphicsPipelineCreateInfo shadowPipelineInfo{};
    shadowPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    shadowPipelineInfo.pNext = &renderingInfo;
    shadowPipelineInfo.stageCount = 2;
    shadowPipelineInfo.pStages = shadowStages;
    shadowPipelineInfo.pVertexInputState = &vertexInputInfo;
    shadowPipelineInfo.pInputAssemblyState = &inputAssembly;
    shadowPipelineInfo.pViewportState = &viewportState;
    shadowPipelineInfo.pRasterizationState = &shadowRaster;
    shadowPipelineInfo.pMultisampleState = &multisampling;
    shadowPipelineInfo.pDepthStencilState = &shadowDepth;
    shadowPipelineInfo.pColorBlendState = &shadowBlend;
    shadowPipelineInfo.pDynamicState = &dynamicState;
    shadowPipelineInfo.layout = app_state.pipelineLayout;
    shadowPipelineInfo.renderPass = VK_NULL_HANDLE;

    if (vkCreateGraphicsPipelines(veekay::app.vk_device, VK_NULL_HANDLE, 1, &shadowPipelineInfo, nullptr, &app_state.shadowPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shadow pipeline!");
    }
    
    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = 4; // ubo + material + dir light + counts
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 2; // point + spot
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[2].descriptorCount = 2; // texture + shadow map
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;
    
    if (vkCreateDescriptorPool(veekay::app.vk_device, &poolInfo, nullptr, &app_state.descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
    
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = app_state.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &app_state.descriptorSetLayout;
    
    if (vkAllocateDescriptorSets(veekay::app.vk_device, &allocInfo, &app_state.descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }
    
    VkDescriptorBufferInfo uboInfo{};
    uboInfo.buffer = app_state.uniformBuffer;
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

    std::array<VkWriteDescriptorSet, 8> descriptorWrites{};

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = app_state.descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &uboInfo;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = app_state.descriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &materialInfo;

    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = app_state.descriptorSet;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &dirLightInfo;

    descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[3].dstSet = app_state.descriptorSet;
    descriptorWrites[3].dstBinding = 3;
    descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[3].descriptorCount = 1;
    descriptorWrites[3].pBufferInfo = &pointInfo;
    
    descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[4].dstSet = app_state.descriptorSet;
    descriptorWrites[4].dstBinding = 4;
    descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[4].descriptorCount = 1;
    descriptorWrites[4].pBufferInfo = &spotInfo;

    descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[5].dstSet = app_state.descriptorSet;
    descriptorWrites[5].dstBinding = 5;
    descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[5].descriptorCount = 1;
    descriptorWrites[5].pBufferInfo = &countsInfo;

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = app_state.texture ? app_state.texture->view : VK_NULL_HANDLE;
    imageInfo.sampler = app_state.textureSampler;

    descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[6].dstSet = app_state.descriptorSet;
    descriptorWrites[6].dstBinding = 6;
    descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[6].descriptorCount = 1;
    descriptorWrites[6].pImageInfo = &imageInfo;

    VkDescriptorImageInfo shadowImageInfo{};
    shadowImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    shadowImageInfo.imageView = app_state.shadowImageView;
    shadowImageInfo.sampler = app_state.shadowSampler;

    descriptorWrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[7].dstSet = app_state.descriptorSet;
    descriptorWrites[7].dstBinding = 7;
    descriptorWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[7].descriptorCount = 1;
    descriptorWrites[7].pImageInfo = &shadowImageInfo;

    vkUpdateDescriptorSets(veekay::app.vk_device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    
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
    vkDestroyBuffer(veekay::app.vk_device, app_state.indexBuffer, nullptr);
    vkFreeMemory(veekay::app.vk_device, app_state.indexBufferMemory, nullptr);
    vkDestroyBuffer(veekay::app.vk_device, app_state.vertexBuffer, nullptr);
    vkFreeMemory(veekay::app.vk_device, app_state.vertexBufferMemory, nullptr);
}

static void writeModelUbo(const glm::mat4& modelMatrix) {
    glm::mat3 normal3 = glm::transpose(glm::inverse(glm::mat3(modelMatrix)));
    glm::mat4 normalMatrix = glm::mat4(1.0f);
    normalMatrix[0] = glm::vec4(normal3[0], 0.0f);
    normalMatrix[1] = glm::vec4(normal3[1], 0.0f);
    normalMatrix[2] = glm::vec4(normal3[2], 0.0f);

    UniformBufferObject ubo{};
    ubo.model = modelMatrix;
    ubo.view = app_state.cachedView;
    ubo.projection = app_state.cachedProj;
    ubo.normalMatrix = normalMatrix;
    ubo.lightViewProj = app_state.cachedLightVP;
    ubo.cameraPos = glm::vec4(app_state.camera.getPosition(), 1.0f);
    ubo.ambientColor = app_state.ambient;

    void* data = nullptr;
    vkMapMemory(veekay::app.vk_device, app_state.uniformBufferMemory, 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(veekay::app.vk_device, app_state.uniformBufferMemory);
}

void update(double time) {
    // === Input: camera controls (KB+mouse, no UI needed) ===
    float deltaTime = 0.0f;
    if (app_state.lastTime > 0.0) {
        deltaTime = static_cast<float>(time - app_state.lastTime);
    }
    app_state.lastTime = time;
    deltaTime = std::clamp(deltaTime, 0.0f, 0.1f); // clamp to avoid huge jumps

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

    // Scroll to adjust FOV
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
    
    glm::mat4 modelMatrix = glm::translate(glm::mat4(1.0f), app_state.modelPosition);
    modelMatrix = glm::rotate(modelMatrix, glm::radians(app_state.sphereRotationY), glm::vec3(0.0f, 1.0f, 0.0f));
    modelMatrix = glm::rotate(modelMatrix, glm::radians(app_state.sphereRotationX), glm::vec3(1.0f, 0.0f, 0.0f));
    modelMatrix = glm::scale(modelMatrix, glm::vec3(scale));
    
    glm::mat4 viewMatrix = app_state.camera.getViewMatrix();
    
    float aspect = static_cast<float>(veekay::app.window_width) / static_cast<float>(veekay::app.window_height);
    if (aspect <= 0.0f) aspect = 1.0f;
    glm::mat4 projectionMatrix = MathUtils::perspective(app_state.fov, aspect, 0.1f, 100.0f);
    
    // Light (directional) view-projection for shadows
    glm::vec3 lightDir = glm::normalize(glm::vec3(app_state.dirLight.directionIntensity));
    glm::vec3 lightPos = -lightDir * 10.0f;
    glm::mat4 lightView = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    float orthoRange = 6.0f;
    glm::mat4 lightProj = glm::ortho(-orthoRange, orthoRange, -orthoRange, orthoRange, 0.1f, 30.0f);
    glm::mat4 lightViewProj = lightProj * lightView;

    app_state.cachedView = viewMatrix;
    app_state.cachedProj = projectionMatrix;
    app_state.cachedLightVP = lightViewProj;
    app_state.cachedScale = scale;

    void* data = nullptr;
    // Material
    vkMapMemory(veekay::app.vk_device, app_state.materialBufferMemory, 0, sizeof(app_state.material), 0, &data);
    memcpy(data, &app_state.material, sizeof(app_state.material));
    vkUnmapMemory(veekay::app.vk_device, app_state.materialBufferMemory);

    // Directional light (normalize direction)
    glm::vec3 dir = glm::normalize(glm::vec3(app_state.dirLight.directionIntensity));
    app_state.dirLight.directionIntensity.x = dir.x;
    app_state.dirLight.directionIntensity.y = dir.y;
    app_state.dirLight.directionIntensity.z = dir.z;
    vkMapMemory(veekay::app.vk_device, app_state.directionalLightBufferMemory, 0, sizeof(app_state.dirLight), 0, &data);
    memcpy(data, &app_state.dirLight, sizeof(app_state.dirLight));
    vkUnmapMemory(veekay::app.vk_device, app_state.directionalLightBufferMemory);

    // Point lights
    std::vector<PointLightData> pointStorage(kMaxPointLights);
    size_t pCount = std::min(app_state.pointLights.size(), static_cast<size_t>(kMaxPointLights));
    for (size_t i = 0; i < pCount; ++i) {
        pointStorage[i] = app_state.pointLights[i];
    }
    vkMapMemory(veekay::app.vk_device, app_state.pointLightBufferMemory, 0,
                sizeof(PointLightData) * kMaxPointLights, 0, &data);
    memcpy(data, pointStorage.data(), sizeof(PointLightData) * kMaxPointLights);
    vkUnmapMemory(veekay::app.vk_device, app_state.pointLightBufferMemory);

    // Spot lights
    std::vector<SpotLightData> spotStorage(kMaxSpotLights);
    size_t sCount = std::min(app_state.spotLights.size(), static_cast<size_t>(kMaxSpotLights));
    for (size_t i = 0; i < sCount; ++i) {
        // нормализуем направление
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

    // Light counts
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
    
    // Transition shadow map for depth write
    VkImageMemoryBarrier toDepth{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = app_state.shadowImage,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &toDepth);

    // Shadow pass (dynamic rendering depth-only)
    VkRenderingAttachmentInfo depthAttach{};
    depthAttach.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttach.imageView = app_state.shadowImageView;
    depthAttach.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttach.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo shadowRender{};
    shadowRender.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    shadowRender.renderArea.offset = {0, 0};
    shadowRender.renderArea.extent = app_state.shadowExtent;
    shadowRender.layerCount = 1;
    shadowRender.colorAttachmentCount = 0;
    shadowRender.pDepthAttachment = &depthAttach;

    pfnCmdBeginRendering(commandBuffer, &shadowRender);

    VkViewport shadowVP{};
    shadowVP.x = 0.0f;
    shadowVP.y = 0.0f;
    shadowVP.width = static_cast<float>(app_state.shadowExtent.width);
    shadowVP.height = static_cast<float>(app_state.shadowExtent.height);
    shadowVP.minDepth = 0.0f;
    shadowVP.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &shadowVP);

    VkRect2D shadowScissor{};
    shadowScissor.offset = {0, 0};
    shadowScissor.extent = app_state.shadowExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &shadowScissor);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app_state.shadowPipeline);
    // Sphere shadow
    glm::mat4 sphereModel = glm::translate(glm::mat4(1.0f), app_state.modelPosition);
    sphereModel = glm::rotate(sphereModel, glm::radians(app_state.sphereRotationY), glm::vec3(0.0f, 1.0f, 0.0f));
    sphereModel = glm::rotate(sphereModel, glm::radians(app_state.sphereRotationX), glm::vec3(1.0f, 0.0f, 0.0f));
    sphereModel = glm::scale(sphereModel, glm::vec3(app_state.cachedScale));
    writeModelUbo(sphereModel);
    VkBuffer vertexBuffersShadow[] = {app_state.vertexBuffer};
    VkDeviceSize offsetsShadow[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffersShadow, offsetsShadow);
    vkCmdBindIndexBuffer(commandBuffer, app_state.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app_state.pipelineLayout, 0, 1, &app_state.descriptorSet, 0, nullptr);
    if (app_state.indexCount > 0) {
        vkCmdDrawIndexed(commandBuffer, app_state.indexCount, 1, 0, 0, 0);
    }

    pfnCmdEndRendering(commandBuffer);

    // Transition shadow map to shader-read for main pass
    VkImageMemoryBarrier toSample = toDepth;
    toSample.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    toSample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    toSample.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    toSample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &toSample);

    // Main render pass
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
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app_state.pipelineLayout, 0, 1, &app_state.descriptorSet, 0, nullptr);
    
    if (app_state.indexCount > 0) {
        vkCmdDrawIndexed(commandBuffer, app_state.indexCount, 1, 0, 0, 0);
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
