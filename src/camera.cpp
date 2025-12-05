#include "camera.h"
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

Camera::Camera() 
    : distance_(3.0f)
    , yaw_(0.0f)
    , pitch_(0.0f)
{
}

void Camera::setDistance(float distance) {
    distance_ = std::clamp(distance, MIN_DISTANCE, MAX_DISTANCE);
}

void Camera::setRotation(float yaw, float pitch) {
    yaw_ = yaw;
    pitch_ = std::clamp(pitch, MIN_PITCH, MAX_PITCH);
}

void Camera::rotate(float deltaYaw, float deltaPitch) {
    yaw_ += deltaYaw;
    pitch_ = std::clamp(pitch_ + deltaPitch, MIN_PITCH, MAX_PITCH);
}

glm::vec3 Camera::getPosition() const {
    // Вычисляем позицию камеры на сфере вокруг цели (начала координат)
    // Используем стандартную сферическую систему координат для орбитальной камеры:
    // При yaw=0, pitch=0 камера должна быть в (0, 0, distance) и смотреть на (0, 0, 0)
    float yawRad = glm::radians(yaw_);
    float pitchRad = glm::radians(pitch_);
    
    glm::vec3 position;
    // Стандартная формула для орбитальной камеры
    position.x = distance_ * cos(pitchRad) * sin(yawRad);
    position.y = distance_ * sin(pitchRad);
    position.z = distance_ * cos(pitchRad) * cos(yawRad);
    
    return position;
}

glm::mat4 Camera::getViewMatrix() const {
    // Орбитальная камера: вращается вокруг начала координат (где находится сфера)
    glm::vec3 target = glm::vec3(0.0f, 0.0f, 0.0f); // Сфера в начале координат
    glm::vec3 position = getPosition();
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    
    return glm::lookAt(position, target, up);
}

