#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
    Camera();
    
    void setDistance(float distance); // Расстояние от цели (сферы)
    void setRotation(float yaw, float pitch);
    void rotate(float deltaYaw, float deltaPitch);
    void move(const glm::vec3& delta);
    void moveRelative(float forward, float right, float up);
    
    glm::mat4 getViewMatrix() const;
    glm::vec3 getPosition() const;
    glm::vec3 getForward() const;
    glm::vec3 getRight() const;
    glm::vec3 getUp() const;
    float getYaw() const { return yaw_; }
    float getPitch() const { return pitch_; }
    float getDistance() const { return distance_; }

private:
    glm::vec3 position_; // Позиция камеры в мировых координатах
    float distance_;  // Расстояние от цели (сферы в начале координат)
    float yaw_;       // Горизонтальный поворот вокруг сферы (в градусах)
    float pitch_;     // Вертикальный поворот вокруг сферы (в градусах)
    
    static constexpr float MAX_PITCH = 89.0f;
    static constexpr float MIN_PITCH = -89.0f;
    static constexpr float MIN_DISTANCE = 1.0f;
    static constexpr float MAX_DISTANCE = 20.0f;
};

