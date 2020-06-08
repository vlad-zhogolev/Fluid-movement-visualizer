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
    Camera();

    Camera(const glm::vec3 &pos, const glm::vec3 &focus, float aspect);
    Camera(const glm::vec3 &pos, const glm::vec3 front, const glm::vec3 up, float fov, float aspect);
    ~Camera();

    void use(const Shader &shader, bool translate_invariant = false) const;
    void setAspect(float aspect);
    void setPos(const glm::vec3 &pos);
    void setFront(const glm::vec3 &front);
    void setUp(const glm::vec3 &up);

    void rotate(const glm::vec2 dxy);
    void pan(const glm::vec2 dxy);
    void zoom(float dy);

    const glm::vec3& getPos() const { return pos; }
    const glm::vec3& getUp() const { return up;  }

    const glm::vec3& getFront() const { return front;  }

    ProjectionInfo getProjectionInfo() const;
    glm::mat4 getInverseView() const;

private:
    static const float SCREEN_ROTATE_RATE;
    static const float SCREEN_PAN_RATE;
    static const float SCREEN_SCROLL_RATE;

private:
    glm::vec3 pos;
    glm::vec3 up;
    glm::vec3 front;


    glm::vec3 rotx, roty;

    float fov;
    float aspect;
};

