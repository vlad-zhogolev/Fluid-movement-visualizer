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

    void render(GLuint particlesVAO, int particlesNumber);

private:

    void renderDepthTexture(GLuint particlesVAO, int particlesNumber);

private:
    int m_windowWidth;
    int m_windowHeight;
    Camera* m_camera = nullptr;

    GLuint m_FBO;
    GLuint m_depthTexture;
    GLuint m_depthTexture1;

    std::unique_ptr<Shader> m_depthShader = nullptr;
    std::unique_ptr<Shader> m_textureRenderShader = nullptr;
};

} // namespace rendering