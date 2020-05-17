#include <rendering/rendering_parameters.h>
#include <glm/geometric.hpp>

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
    instance.particleRadius = 0.06f;
    instance.fluidColor = glm::normalize(glm::vec3(15, 94, 156));
    instance.attenuationCoefficients = glm::vec3(0.05f);

    return instance;
}