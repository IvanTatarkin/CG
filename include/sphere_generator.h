#pragma once

#include "vertex.h"
#include <vector>
#include <glm/glm.hpp>

class SphereGenerator {
public:
    // Генерирует сферу с заданным количеством сегментов
    // Примерно ~100 вершин при segments = 10
    static std::vector<Vertex> generateSphere(float radius, int segments = 10, const glm::vec3& color = glm::vec3(0.5f, 0.8f, 1.0f));
    
    // Возвращает индексы для отрисовки
    static std::vector<uint32_t> generateIndices(int segments);
};

