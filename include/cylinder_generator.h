#pragma once

#include "vertex.h"
#include <vector>
#include <glm/glm.hpp>

class CylinderGenerator {
public:
    // Генерирует цилиндр с заданным радиусом, высотой и количеством сегментов
    static std::vector<Vertex> generateCylinder(float radius, float height, int segments = 20, const glm::vec3& color = glm::vec3(0.5f, 0.8f, 1.0f));
    
    // Возвращает индексы для отрисовки
    static std::vector<uint32_t> generateIndices(int segments);
};

