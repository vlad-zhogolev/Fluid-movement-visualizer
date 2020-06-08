#pragma once

#include <rendering/shader.h>
#include <glm/glm.hpp>

struct ProjectionInfo
{
    float l, r;
    float t, b;
    float n, f;

    float projectionXX;
    float projectionYY;
    float projectionZZ;
    float projectionZW;
};

class Camera
{
public:
    Camera(const glm::vec3 &pos, const glm::vec3 &focus, float aspect);
    Camera(const glm::vec3 &pos, const glm::vec3 front, const glm::vec3 up, float fov, float aspect);
    ~Camera();

    void use(const Shader &shader, bool translate_invariant = false) const;
    inline void setAspect(float aspect) { m_aspect = aspect;}
    inline void setPos(const glm::vec3& position) { m_position = position; }
    inline void setFront(const glm::vec3& front) { m_front = front; }
    inline void setUp(const glm::vec3& up) { m_up = up; }

    void rotate(const glm::vec2 dxy);
    void pan(const glm::vec2 dxy);
    void zoom(float dy);

    const glm::vec3& getPos() const { return m_position; }
    const glm::vec3& getUp() const { return m_up;  }

    const glm::vec3& getFront() const { return m_front;  }

    ProjectionInfo getProjectionInfo() const;
    glm::mat4 getInverseView() const;

private:
    static const float SCREEN_ROTATE_RATE;
    static const float SCREEN_PAN_RATE;
    static const float SCREEN_SCROLL_RATE;
    static const float MIN_SCROLL_DISTANCE;
    static const float MAX_SCROLL_DISTANCE;
    static const float NEAR_PLANE_DISTANCE;
    static const float FAR_PLANE_DISTANCE;

private:
    glm::vec3 m_position;
    glm::vec3 m_up;
    glm::vec3 m_front;

    glm::vec3 m_rotx;
    glm::vec3 m_roty;

    float m_fov;
    float m_aspect;
};

