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
    explicit SmoothRenderer(int windowWidth, int windowHeight, Camera* camera, GLuint skyboxTexture);

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

    
    void GenerateFramebufferAndTextures();
    void ConfigureFramebuffer();

private:
    int m_windowWidth;
    int m_windowHeight;
    Camera* m_camera = nullptr;
    GLuint m_skyboxTexture;

    float m_fluidRefractionIndex = 1.333f; // water refraction index
    float m_particleRadius = 0.06f; // TODO: find out how to set this parameter (maybe from UI or implicitly?)

    // Framebuffer and it's components
    GLuint m_FBO;
    GLuint m_defaultDepthTexture;
    GLuint m_depthTexture1;
    GLuint m_depthTexture2;
    GLuint m_normalsTexture;
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