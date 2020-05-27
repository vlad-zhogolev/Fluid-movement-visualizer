#pragma once

#include <glad/glad.h>
#include <glm/gtx/common.hpp>

struct IParticlesProvider
{
    virtual void Provide(GLuint positions, GLuint velocities, int particlesNumber) = 0;
    virtual void SetPosition(const glm::vec3& position) = 0;
    virtual void SetSize(float size) = 0;
};