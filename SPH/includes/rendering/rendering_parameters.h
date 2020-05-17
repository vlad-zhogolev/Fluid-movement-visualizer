#pragma once

#include <glm/common.hpp>


class RenderingParameters
{
public:
    static const int SMOOTH_STEPS_NUMBER_MIN = 0;
    static const int SMOOTH_STEPS_NUMBER_MAX = 25;

public:
    int fps;
    int smoothStepsNumber;
    float fluidRefractionIndex;
    float particleRadius;
    
    glm::vec3 fluidColor;

    static RenderingParameters& GetInstance();
};