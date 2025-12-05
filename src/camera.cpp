#include "camera.h"
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

Camera::Camera() 
    : position_(0.0f, 0.0f, 3.0f)
    , distance_(3.0f)
    , yaw_(0.0f)
    , pitch_(0.0f) {
}

void Camera::setDistance(float distance) {
    distance_ = std::clamp(distance, MIN_DISTANCE, MAX_DISTANCE);
    // Сохраняем орбитальное поведение: позиция лежит на сфере радиусом distance вокруг (0,0,0)
    glm::vec3 forward = getForward();
    position_ = -forward * distance_;
}

void Camera::setRotation(float yaw, float pitch) {
    yaw_ = yaw;
    pitch_ = std::clamp(pitch, MIN_PITCH, MAX_PITCH);
    // Пересчитываем позицию, чтобы остаться на текущей орбите
    glm::vec3 forward = getForward();
    position_ = -forward * distance_;
}

void Camera::rotate(float deltaYaw, float deltaPitch) {
    yaw_ += deltaYaw;
    pitch_ = std::clamp(pitch_ + deltaPitch, MIN_PITCH, MAX_PITCH);
    glm::vec3 forward = getForward();
    position_ = -forward * distance_;
}

void Camera::move(const glm::vec3& delta) {
    position_ += delta;
    // Обновляем distance_, чтобы совместить свободное перемещение и орбиту/слайдер
    distance_ = std::clamp(glm::length(position_), MIN_DISTANCE, MAX_DISTANCE);
}

void Camera::moveRelative(float forward, float right, float up) {
    glm::vec3 f = getForward();
    glm::vec3 r = getRight();
    glm::vec3 u = getUp();
    move(f * forward + r * right + u * up);
}

glm::vec3 Camera::getPosition() const {
    return position_;
}

glm::mat4 Camera::getViewMatrix() const {
    glm::vec3 forward = getForward();
    glm::vec3 up = getUp();
    return glm::lookAt(position_, position_ + forward, up);
}

glm::vec3 Camera::getForward() const {
    float yawRad = glm::radians(yaw_);
    float pitchRad = glm::radians(pitch_);
    
    glm::vec3 forward;
    forward.x = cos(pitchRad) * sin(yawRad);
    forward.y = sin(pitchRad);
    forward.z = -cos(pitchRad) * cos(yawRad); // направо-handed: при yaw=0 смотрим вдоль -Z
    return glm::normalize(forward);
}

glm::vec3 Camera::getRight() const {
    return glm::normalize(glm::cross(getForward(), glm::vec3(0.0f, 1.0f, 0.0f)));
}

glm::vec3 Camera::getUp() const {
    return glm::normalize(glm::cross(getRight(), getForward()));
}
