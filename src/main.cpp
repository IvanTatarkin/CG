#include <veekay/veekay.hpp>
#include "sphere_generator.h"
#include "camera.h"
#include "math_utils.h"
#include "vertex.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <imgui.h>

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
};

static struct {
    std::vector<Vertex> sphereVertices;
    std::vector<uint32_t> sphereIndices;
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;
    VkBuffer uniformBuffer = VK_NULL_HANDLE;
    VkDeviceMemory uniformBufferMemory = VK_NULL_HANDLE;
    VkShaderModule vertexShaderModule = VK_NULL_HANDLE;
    VkShaderModule fragmentShaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkPipeline wireframePipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    Camera camera;
    float sphereRotationY = 0.0f;
    float sphereRotationX = 0.0f;
    bool autoRotate = true;
    bool wireframeMode = false;
    uint32_t indexCount = 0;
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

void init(VkCommandBuffer /*cmd*/) {
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
    
    app_state.vertexShaderModule = loadShaderModule("shaders/vert.spv");
    app_state.fragmentShaderModule = loadShaderModule("shaders/frag.spv");
    
    if (!app_state.vertexShaderModule || !app_state.fragmentShaderModule) {
        std::cerr << "Failed to load shaders!" << std::endl;
        veekay::app.running = false;
        return;
    }
    
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboLayoutBinding;
    
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
    
    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, position);
    
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);
    
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
    
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = 1;
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
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
    
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = app_state.uniformBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(UniformBufferObject);
    
    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = app_state.descriptorSet;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &bufferInfo;
    
    vkUpdateDescriptorSets(veekay::app.vk_device, 1, &descriptorWrite, 0, nullptr);
    
    std::cout << "Initialization complete!" << std::endl;
}

void shutdown() {
    vkDestroyDescriptorPool(veekay::app.vk_device, app_state.descriptorPool, nullptr);
    vkDestroyPipeline(veekay::app.vk_device, app_state.graphicsPipeline, nullptr);
    vkDestroyPipeline(veekay::app.vk_device, app_state.wireframePipeline, nullptr);
    vkDestroyPipelineLayout(veekay::app.vk_device, app_state.pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(veekay::app.vk_device, app_state.descriptorSetLayout, nullptr);
    vkDestroyShaderModule(veekay::app.vk_device, app_state.fragmentShaderModule, nullptr);
    vkDestroyShaderModule(veekay::app.vk_device, app_state.vertexShaderModule, nullptr);
    vkDestroyBuffer(veekay::app.vk_device, app_state.uniformBuffer, nullptr);
    vkFreeMemory(veekay::app.vk_device, app_state.uniformBufferMemory, nullptr);
    vkDestroyBuffer(veekay::app.vk_device, app_state.indexBuffer, nullptr);
    vkFreeMemory(veekay::app.vk_device, app_state.indexBufferMemory, nullptr);
    vkDestroyBuffer(veekay::app.vk_device, app_state.vertexBuffer, nullptr);
    vkFreeMemory(veekay::app.vk_device, app_state.vertexBufferMemory, nullptr);
}

void update(double time) {
    ImGui::Begin("Controls");
    
    ImGui::Text("=== Camera (Orbital) ===");
    float yaw = app_state.camera.getYaw();
    float pitch = app_state.camera.getPitch();
    float distance = app_state.camera.getDistance();
    
    if (ImGui::SliderFloat("Camera Yaw", &yaw, -180.0f, 180.0f)) {
        app_state.camera.setRotation(yaw, pitch);
    }
    if (ImGui::SliderFloat("Camera Pitch", &pitch, -89.0f, 89.0f)) {
        app_state.camera.setRotation(yaw, pitch);
    }
    if (ImGui::SliderFloat("Camera Distance", &distance, 1.0f, 10.0f)) {
        app_state.camera.setDistance(distance);
    }
    
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
    ImGui::Text("=== Animation Info ===");
    float scale = 1.0f + 0.3f * std::sin(static_cast<float>(time));
    ImGui::Text("Scale: %.2f", scale);
    ImGui::Text("Time: %.2f", static_cast<float>(time));
    
    ImGui::End();
    
    if (app_state.autoRotate) {
        app_state.sphereRotationY = static_cast<float>(time) * 30.0f;
    }
    
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    modelMatrix = glm::rotate(modelMatrix, glm::radians(app_state.sphereRotationY), glm::vec3(0.0f, 1.0f, 0.0f));
    modelMatrix = glm::rotate(modelMatrix, glm::radians(app_state.sphereRotationX), glm::vec3(1.0f, 0.0f, 0.0f));
    modelMatrix = glm::scale(modelMatrix, glm::vec3(scale));
    
    glm::mat4 viewMatrix = app_state.camera.getViewMatrix();
    
    float aspect = static_cast<float>(veekay::app.window_width) / static_cast<float>(veekay::app.window_height);
    if (aspect <= 0.0f) aspect = 1.0f;
    glm::mat4 projectionMatrix = MathUtils::perspective(45.0f, aspect, 0.1f, 100.0f);
    
    UniformBufferObject ubo{};
    ubo.model = modelMatrix;
    ubo.view = viewMatrix;
    ubo.projection = projectionMatrix;
    
    void* data;
    vkMapMemory(veekay::app.vk_device, app_state.uniformBufferMemory, 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(veekay::app.vk_device, app_state.uniformBufferMemory);
}

void render(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer) {
    vkResetCommandBuffer(commandBuffer, 0);
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = veekay::app.vk_render_pass;
    renderPassInfo.framebuffer = framebuffer;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = {veekay::app.window_width, veekay::app.window_height};
    
    VkClearValue clearColor = {{{0.1f, 0.1f, 0.1f, 1.0f}}};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;
    
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
