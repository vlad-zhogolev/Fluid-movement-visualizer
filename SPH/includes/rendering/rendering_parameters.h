#pragma once

#include <glm/common.hpp>


class RenderingParameters
{
public:
    static const int SMOOTH_STEPS_NUMBER_MIN = 0;
    static const int SMOOTH_STEPS_NUMBER_MAX = 25;

    static const float ATTENUATION_COEFFICIENT_MIN;
    static const float ATTENUATION_COEFFICIENT_MAX;

public:
    int fps;
    int smoothStepsNumber;
    float fluidRefractionIndex;
    float particleRadius;
    
    glm::vec3 fluidColor;
    glm::vec3 attenuationCoefficients;

    static RenderingParameters& GetInstance();
    static RenderingParameters* GetInstancePtr();
};