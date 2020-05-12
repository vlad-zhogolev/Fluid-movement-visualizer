#pragma once

#include <glad/glad.h>
#include <rendering/shader.h>
#include <rendering/camera.h>
#include <memory>

namespace rendering {

class SmoothRenderer
{
public:
    explicit SmoothRenderer(int windowWidth, int windowHeight, Camera* camera);

    void Render(GLuint particlesVAO, int particlesNumber);

private:

    void RenderDepthTexture(GLuint particlesVAO, int particlesNumber);
    void SmoothDepthTexture();
    void ExtractNormalsFromDepth();
    void RenderThicknessTexture(GLuint particlesVAO, int particlesNumber);

private:
    int m_windowWidth;
    int m_windowHeight;
    Camera* m_camera = nullptr;

    float m_particleRadius = 0.04f; // TODO: find out how to set this parameter (maybe from UI or implicitly?)

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
};

} // namespace rendering