#include <rendering/rendering_parameters.h>
#include <glm/geometric.hpp>

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