#include "sphere_generator.h"
#include <cmath>

std::vector<Vertex> SphereGenerator::generateSphere(float radius, int segments, const glm::vec3& color) {
    std::vector<Vertex> vertices;
    
    // Генерируем вершины сферы используя параметрические уравнения
    // x = r * sin(phi) * cos(theta)
    // y = r * cos(phi)
    // z = r * sin(phi) * sin(theta)
    // где phi от 0 до PI, theta от 0 до 2*PI
    
    for (int i = 0; i <= segments; ++i) {
        float phi = M_PI * i / segments;  // От 0 до PI
        
        for (int j = 0; j <= segments; ++j) {
            float theta = 2.0f * M_PI * j / segments;  // От 0 до 2*PI
            
            Vertex vertex;
            vertex.position.x = radius * sin(phi) * cos(theta);
            vertex.position.y = radius * cos(phi);
            vertex.position.z = radius * sin(phi) * sin(theta);
            vertex.normal = glm::normalize(vertex.position); // нормаль направлена от центра
            vertex.normal = glm::normalize(vertex.position);
            vertex.color = color;
            vertex.texCoord = glm::vec2(theta / (2.0f * M_PI), phi / M_PI);
            
            vertices.push_back(vertex);
        }
    }
    
    return vertices;
}

std::vector<uint32_t> SphereGenerator::generateIndices(int segments) {
    std::vector<uint32_t> indices;
    
    // Генерируем индексы для треугольников
    for (int i = 0; i < segments; ++i) {
        for (int j = 0; j < segments; ++j) {
            int first = i * (segments + 1) + j;
            int second = first + segments + 1;
            
            // Первый треугольник
            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);
            
            // Второй треугольник
            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }
    
    return indices;
}

