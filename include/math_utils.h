#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>

namespace MathUtils {
    // Создание матрицы перспективной проекции
    inline glm::mat4 perspective(float fov, float aspect, float near, float far) {
        return glm::perspective(glm::radians(fov), aspect, near, far);
    }

    // Создание матрицы ортографической проекции
    inline glm::mat4 orthographic(float left, float right, float bottom, float top, float near, float far) {
        return glm::ortho(left, right, bottom, top, near, far);
    }

    // Создание матрицы вида из камеры
    inline glm::mat4 lookAt(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up) {
        return glm::lookAt(eye, center, up);
    }
}

