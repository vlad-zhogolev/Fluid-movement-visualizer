#pragma once

#include <simulation/providers/cube_provider.h>

class SphereProvider : public CubeProvider
{
public:
    SphereProvider(const float3& position, int sizeInParticles)
        : CubeProvider(position, sizeInParticles) {}

    void Provide() override;
    int GetParticlesNumber() override;

private:

};