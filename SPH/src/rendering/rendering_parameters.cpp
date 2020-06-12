#include <rendering/rendering_parameters.h>
#include <glm/geometric.hpp>
#include <helper_math.h>

const float RenderingParameters::ATTENUATION_COEFFICIENT_MIN = 0.0f;
const float RenderingParameters::ATTENUATION_COEFFICIENT_MAX = 1.0f;

RenderingParameters& RenderingParameters::GetInstance()
{
    static RenderingParameters instance;
    
    static bool isInitialized = false;
    if (isInitialized)
    {
        return instance;
    }
    isInitialized = true;

    instance.fps = 0;
    instance.smoothStepsNumber = 3;
    instance.fluidRefractionIndex = 1.333f;
    instance.particleRadius = 0.04f;
    //instance.fluidColor = glm::normalize(glm::vec3(15, 94, 156));
    instance.fluidColor = glm::normalize(glm::vec3(0.08, 0.71, 0.85));
    instance.attenuationCoefficients = glm::vec3(0.22f, 0.18f, 0.2f);

    return instance;
}

RenderingParameters* RenderingParameters::GetInstancePtr()
{
    return &GetInstance();
}

void RenderingParameters::SetSmoothingIter(int number)
{
    smoothStepsNumber = clamp(number, SMOOTH_STEPS_NUMBER_MIN, SMOOTH_STEPS_NUMBER_MAX);
}

void RenderingParameters::SetAttenuationRed(float value)
{
    attenuationCoefficients.r = clamp(value, ATTENUATION_COEFFICIENT_MIN, ATTENUATION_COEFFICIENT_MAX);
}

void RenderingParameters::SetAttenuationBlue(float value)
{
    attenuationCoefficients.b = clamp(value, ATTENUATION_COEFFICIENT_MIN, ATTENUATION_COEFFICIENT_MAX);
}

void RenderingParameters::SetAttenuationGreen(float value)
{
    attenuationCoefficients.g = clamp(value, ATTENUATION_COEFFICIENT_MIN, ATTENUATION_COEFFICIENT_MAX);
}