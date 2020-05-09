#include <rendering/smooth_renderer.h>
#include <rendering/camera.h>
#include <helper.h>
#include <iostream>

namespace {

GLuint screenQuadVAO = 0;
GLuint screenQuadVBO = 0;

void renderScreenQuad()
{
    if (screenQuadVAO == 0)
    {
        float quadVertices[] = 
        {   // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
            // positions   // texCoords
            -1.0f,  1.0f,  0.0f, 1.0f,
            -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f, 0.0f,

            -1.0f,  1.0f,  0.0f, 1.0f,
             1.0f, -1.0f,  1.0f, 0.0f,
             1.0f,  1.0f,  1.0f, 1.0f
        };
        // Create buffer objects for quad
        glGenVertexArrays(1, &screenQuadVAO);
        glGenBuffers(1, &screenQuadVBO);

        // Setup buffer layout
        glBindVertexArray(screenQuadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, screenQuadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0); // positions
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float))); // texture coordinates

        // Cleanup
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    // Draw quad in screen space
    glBindVertexArray(screenQuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

}

namespace rendering {

SmoothRenderer::SmoothRenderer(int windowWidth, int windowHeight, Camera* camera)
    : m_windowWidth(windowWidth)
    , m_windowHeight(windowHeight)
    , m_camera(camera)
{
    // Compile and load shaders
    const std::string shadersFolderPath = "shaders/smooth_renderer";
    m_depthShader = std::make_unique<Shader>(
        Path(shadersFolderPath + "/depth_shader.vert"),
        Path(shadersFolderPath + "/depth_shader.frag"));

    m_textureRenderShader = std::make_unique<Shader>(
        Path(shadersFolderPath + "/render_texture.vert"),
        Path(shadersFolderPath + "/render_texture.frag"));

    m_depthSmoothingShader = std::make_unique<Shader>(
        Path(shadersFolderPath + "/depth_smoothing_shader.vert"),
        Path(shadersFolderPath + "/depth_smoothing_shader.frag"));

    // Create and setup framebuffer textures
    // Texture for default depth, neccessary for all framebuffers
    glGenTextures(1, &m_defaultDepthTexture);
    glBindTexture(GL_TEXTURE_2D, m_defaultDepthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, m_windowWidth, m_windowHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenTextures(1, &m_depthTexture1);
    glBindTexture(GL_TEXTURE_2D, m_depthTexture1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, m_windowWidth, m_windowHeight, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenTextures(1, &m_depthTexture2);
    glBindTexture(GL_TEXTURE_2D, m_depthTexture2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, m_windowWidth, m_windowHeight, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Create and setup framebuffer
    glGenFramebuffers(1, &m_FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_defaultDepthTexture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_depthTexture1, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, m_depthTexture2, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "Framebuffer not complete!" << std::endl;
    }

    // Cleanup
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SmoothRenderer::render(GLuint particlesVAO, int particlesNumber)
{
    renderDepthTexture(particlesVAO, particlesNumber);

    smoothDepthTexture();

    // Configure
    m_textureRenderShader->use();
    m_textureRenderShader->setUnif("sourceTexture", 0);
    glActiveTexture(GL_TEXTURE0);
    //glBindTexture(GL_TEXTURE_2D, m_isFirstDepthTextureSource ? m_depthTexture2 : m_depthTexture1);
    glBindTexture(GL_TEXTURE_2D, m_depthTexture2);

    // Draw
    renderScreenQuad();

    // Cleanup
    glBindTexture(GL_TEXTURE_2D, 0);
}

void SmoothRenderer::renderDepthTexture(GLuint particlesVAO, int particlesNumber)
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glDisable(GL_BLEND);

    GLfloat inf[] = { 100.f };
    glClearTexImage(m_depthTexture1, 0, GL_RED, GL_FLOAT, inf);
    //glClearTexImage(m_depthTexture2, 0, GL_RED, GL_FLOAT, inf);

    GLenum bufs[] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, bufs);

    m_depthShader->use();
    m_camera->use(Shader::now());

    ProjectionInfo i = m_camera->getProjectionInfo();
    m_depthShader->setUnif("windowHeight", m_windowHeight);
    m_depthShader->setUnif("projectionTop", i.t);
    m_depthShader->setUnif("projectionNear", i.n);
    m_depthShader->setUnif("particleRadius", 0.04f); // TODO: find out how to set this parameter (maybe from UI or implicitly?)

    glEnable(GL_DEPTH_TEST);
    // glDisable(GL_BLEND);
    glBindVertexArray(particlesVAO);

    // Draw
    glClear(GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_POINTS, 0, particlesNumber);

    // Cleanup
    glBindVertexArray(0);
    glEnable(GL_BLEND);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SmoothRenderer::smoothDepthTexture()
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);

    m_depthSmoothingShader->use();
    m_camera->use(Shader::now());

    // m_depthSmoothingShader->setUnif("filterRadius", 10);
    // m_depthSmoothingShader->setUnif("blurScale", 6.f);
    // m_depthSmoothingShader->setUnif("blurDepthFalloff", 0.1f);

    m_depthSmoothingShader->setUnif("sourceDepthTexture", 0);
    glActiveTexture(GL_TEXTURE0);

    m_isFirstDepthTextureSource = false;
    for (int i = 0; i < 10; ++i)
    {
        m_isFirstDepthTextureSource = !m_isFirstDepthTextureSource;
        if (m_isFirstDepthTextureSource)
        {
            glBindTexture(GL_TEXTURE_2D, m_depthTexture1);
            GLenum bufs[] = { GL_COLOR_ATTACHMENT1 };
            glDrawBuffers(1, bufs);
        }
        else
        {
            glBindTexture(GL_TEXTURE_2D, m_depthTexture2);
            GLenum bufs[] = { GL_COLOR_ATTACHMENT0 };
            glDrawBuffers(1, bufs);
        }
        // Draw
        glDisable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
        renderScreenQuad();
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
    }

    // Cleanup
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

} // namespace rendering