#pragma once

#include <glad/glad.h>
#include <rendering/shader.h>
#include <rendering/camera.h>
#include <simulation/simulation_parameters.h>
#include <memory>

namespace rendering {

class SmoothRenderer
{
public:
    explicit SmoothRenderer(int windowWidth, int windowHeight, std::shared_ptr<Camera> camera, GLuint skyboxTexture);

    void Render(GLuint particlesVAO, int particlesNumber);

    void HandleWindowResolutionChange(int newWindowWidth, int newWindowHeight);

private:

    void RenderDepthTexture(GLuint particlesVAO, int particlesNumber);
    void SmoothDepthTexture();
    void ExtractNormalsFromDepth();
    void RenderThicknessTexture(GLuint particlesVAO, int particlesNumber);
    void RenderFluid();

    float GetBaseReflectance();

    GLuint GetSmoothingSourceDepthTexture();
    GLuint GetSmoothingTargetDepthTexture();
    GLuint GetSmoothingTargetColorAttachment();
    
    void GenerateFramebufferAndTextures();
    void ConfigureFramebuffer();

    void UpdateParameters();

private:
    int m_windowWidth;
    int m_windowHeight;
    std::shared_ptr<Camera> m_camera = nullptr;
    GLuint m_skyboxTexture;

    int m_smoothingIterations;
    float m_fluidRefractionIndex = 1.333f; // water refraction index, TODO: move to UI
    float m_particleRadius = 0.06f; // TODO: move to UI
    glm::vec3 m_fluidColor;
    glm::vec3 m_attenuationCoefficients;

    // Framebuffer and it's components
    GLuint m_FBO;
    GLuint m_defaultDepthTexture;
    GLuint m_depthTexture1;
    GLuint m_depthTexture2;
    GLuint m_normalsTexture;

    GLuint m_thicknessFBO;
    GLuint m_thicknessDepthTexture;
    GLuint m_thicknessTexture;

    bool m_isFirstDepthTextureSource = true;

    std::unique_ptr<Shader> m_depthShader = nullptr;
    std::unique_ptr<Shader> m_textureRenderShader = nullptr;
    std::unique_ptr<Shader> m_depthSmoothingShader = nullptr;
    std::unique_ptr<Shader> m_normalsExtractionShader = nullptr;
    std::unique_ptr<Shader> m_thicknessShader = nullptr;
    std::unique_ptr<Shader> m_combinedRenderingShader = nullptr;
};

} // namespace rendering