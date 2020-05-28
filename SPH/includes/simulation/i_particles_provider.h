#pragma once

#include <glad/glad.h>
#include <helper_math.h>

struct IParticlesProvider
{
    virtual void SetTargets(GLuint positions, GLuint velocities) = 0;
    virtual void Provide(int& particlesNumber) = 0;
    virtual bool SetPosition(const float3& position) = 0;
    virtual bool SetSize(int particlesNumber) = 0;
    virtual bool SetDensity(float density) = 0;
    virtual bool IsInsideBoundaries(const float3& upperBoundary, const float3& lowerBoundary) = 0;
};