#include "cylinder_generator.h"
#include <cmath>

std::vector<Vertex> CylinderGenerator::generateCylinder(float radius, float height, int segments, const glm::vec3& color) {
    std::vector<Vertex> vertices;
    
    float halfHeight = height / 2.0f;
    

    Vertex bottomCenter;
    bottomCenter.position = glm::vec3(0.0f, -halfHeight, 0.0f);
    bottomCenter.normal = glm::vec3(0.0f, -1.0f, 0.0f);
    bottomCenter.color = color;
    bottomCenter.texCoord = glm::vec2(0.5f, 0.5f);
    vertices.push_back(bottomCenter);
    
    for (int i = 0; i <= segments; ++i) {
        float theta = 2.0f * M_PI * i / segments;
        Vertex vertex;
        vertex.position.x = radius * cos(theta);
        vertex.position.y = -halfHeight;
        vertex.position.z = radius * sin(theta);
        vertex.normal = glm::vec3(0.0f, -1.0f, 0.0f);
        vertex.color = color;
        vertex.texCoord = glm::vec2(0.5f + 0.5f * cos(theta), 0.5f + 0.5f * sin(theta));
        vertices.push_back(vertex);
    }
    
    Vertex topCenter;
    topCenter.position = glm::vec3(0.0f, halfHeight, 0.0f);
    topCenter.normal = glm::vec3(0.0f, 1.0f, 0.0f);
    topCenter.color = color;
    topCenter.texCoord = glm::vec2(0.5f, 0.5f);
    vertices.push_back(topCenter);
    
    for (int i = 0; i <= segments; ++i) {
        float theta = 2.0f * M_PI * i / segments;
        Vertex vertex;
        vertex.position.x = radius * cos(theta);
        vertex.position.y = halfHeight;
        vertex.position.z = radius * sin(theta);
        vertex.normal = glm::vec3(0.0f, 1.0f, 0.0f);
        vertex.color = color;
        vertex.texCoord = glm::vec2(0.5f + 0.5f * cos(theta), 0.5f + 0.5f * sin(theta));
        vertices.push_back(vertex);
    }
    
    for (int i = 0; i <= segments; ++i) {
        float theta = 2.0f * M_PI * i / segments;
        
        Vertex vertexBottom;
        vertexBottom.position.x = radius * cos(theta);
        vertexBottom.position.y = -halfHeight;
        vertexBottom.position.z = radius * sin(theta);
        vertexBottom.normal = glm::normalize(glm::vec3(cos(theta), 0.0f, sin(theta)));
        vertexBottom.color = color;
        vertexBottom.texCoord = glm::vec2(theta / (2.0f * M_PI), 0.0f);
        vertices.push_back(vertexBottom);
        
        Vertex vertexTop;
        vertexTop.position.x = radius * cos(theta);
        vertexTop.position.y = halfHeight;
        vertexTop.position.z = radius * sin(theta);
        vertexTop.normal = glm::normalize(glm::vec3(cos(theta), 0.0f, sin(theta)));
        vertexTop.color = color;
        vertexTop.texCoord = glm::vec2(theta / (2.0f * M_PI), 1.0f);
        vertices.push_back(vertexTop);
    }
    
    return vertices;
}

std::vector<uint32_t> CylinderGenerator::generateIndices(int segments) {
    std::vector<uint32_t> indices;
    
    int bottomCenter = 0;
    int bottomCircleStart = 1;
    int topCenter = bottomCircleStart + segments + 1;
    int topCircleStart = topCenter + 1;
    int sideStart = topCircleStart + segments + 1;
    
    for (int i = 0; i < segments; ++i) {
        indices.push_back(bottomCenter);
        indices.push_back(bottomCircleStart + i);
        indices.push_back(bottomCircleStart + i + 1);
    }
    
    for (int i = 0; i < segments; ++i) {
        indices.push_back(topCenter);
        indices.push_back(topCircleStart + i + 1);
        indices.push_back(topCircleStart + i);
    }
    
    for (int i = 0; i < segments; ++i) {
        int bottomIdx = sideStart + 2 * i;
        int topIdx = sideStart + 2 * i + 1;
        int nextBottomIdx = sideStart + 2 * ((i + 1) % (segments + 1));
        int nextTopIdx = sideStart + 2 * ((i + 1) % (segments + 1)) + 1;
        
        indices.push_back(bottomIdx);
        indices.push_back(topIdx);
        indices.push_back(nextBottomIdx);
        
        indices.push_back(nextBottomIdx);
        indices.push_back(topIdx);
        indices.push_back(nextTopIdx);
    }
    
    return indices;
}

