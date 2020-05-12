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

} // namespace

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

    m_normalsExtractionShader = std::make_unique<Shader>(
        Path(shadersFolderPath + "/extract_normals_from_depth.vert"),
        Path(shadersFolderPath + "/extract_normals_from_depth.frag"));

    m_thicknessShader = std::make_unique<Shader>(
        Path(shadersFolderPath + "/thickness_shader.vert"),
        Path(shadersFolderPath + "/thickness_shader.frag"));

    // Create and setup framebuffer textures
    // Texture for default depth, neccessary for all framebuffers
    glGenTextures(1, &m_defaultDepthTexture);
    glBindTexture(GL_TEXTURE_2D, m_defaultDepthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, m_windowWidth, m_windowHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Texture for linear depth in view-space and further smoothing. Used for initial depth extraction.
    glGenTextures(1, &m_depthTexture1);
    glBindTexture(GL_TEXTURE_2D, m_depthTexture1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, m_windowWidth, m_windowHeight, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Texture for linear depth in view-space and further smoothing.
    glGenTextures(1, &m_depthTexture2);
    glBindTexture(GL_TEXTURE_2D, m_depthTexture2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, m_windowWidth, m_windowHeight, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Texture for fluid surface normals extracted from depth.
    glGenTextures(1, &m_normalsTexture);
    glBindTexture(GL_TEXTURE_2D, m_normalsTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_windowWidth, m_windowHeight, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Texture for fluid thickness
    glGenTextures(1, &m_thicknessTexture);
    glBindTexture(GL_TEXTURE_2D, m_thicknessTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, m_windowWidth, m_windowHeight, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Create and setup framebuffer
    glGenFramebuffers(1, &m_FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_defaultDepthTexture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_depthTexture1, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, m_depthTexture2, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, m_normalsTexture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, m_thicknessTexture, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "Framebuffer not complete!" << std::endl;
    }

    // Cleanup
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SmoothRenderer::Render(GLuint particlesVAO, int particlesNumber)
{
    //RenderDepthTexture(particlesVAO, particlesNumber);
    RenderThicknessTexture(particlesVAO, particlesNumber);
    //SmoothDepthTexture();
    //ExtractNormalsFromDepth();
    

    // Configure
    m_textureRenderShader->use();
    m_textureRenderShader->setUnif("sourceTexture", 0);
    glActiveTexture(GL_TEXTURE0);
    //glBindTexture(GL_TEXTURE_2D, m_depthTexture1);
    //glBindTexture(GL_TEXTURE_2D, m_isFirstDepthTextureSource ? m_depthTexture2 : m_depthTexture1);
    //glBindTexture(GL_TEXTURE_2D, m_normalsTexture);
    glBindTexture(GL_TEXTURE_2D, m_thicknessTexture);

    // Draw
    renderScreenQuad();

    // Cleanup
    glBindTexture(GL_TEXTURE_2D, 0);
}

void SmoothRenderer::RenderDepthTexture(GLuint particlesVAO, int particlesNumber)
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);

    GLenum drawBuffer[] = { GL_COLOR_ATTACHMENT0 }; // m_depthTexture1
    glDrawBuffers(1, drawBuffer);

    // Clear linear depth texture
    GLfloat largeDepth = 1000.f;
    glClearBufferfv(GL_COLOR, 0, &largeDepth);
    // Clear z-buffer for m_FBO
    glClear(GL_DEPTH_BUFFER_BIT);

    m_depthShader->use();
    m_camera->use(Shader::now());

    ProjectionInfo projectionInfo = m_camera->getProjectionInfo();
    m_depthShader->setUnif("windowHeight", m_windowHeight);
    m_depthShader->setUnif("projectionTop", projectionInfo.t);
    m_depthShader->setUnif("projectionNear", projectionInfo.n);
    m_depthShader->setUnif("particleRadius", m_particleRadius);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    // Draw
    glBindVertexArray(particlesVAO);
    glDrawArrays(GL_POINTS, 0, particlesNumber);

    // Cleanup
    glBindVertexArray(0);
    glDisable(GL_DEPTH_TEST);
    //glEnable(GL_BLEND);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SmoothRenderer::SmoothDepthTexture()
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);

    m_depthSmoothingShader->use();
    m_camera->use(Shader::now());

    m_depthSmoothingShader->setUnif("filterRadius", 10);
    m_depthSmoothingShader->setUnif("blurScale", 0.2f);
    m_depthSmoothingShader->setUnif("blurDepthFalloff", 10.f);

    m_depthSmoothingShader->setUnif("sourceDepthTexture", 0);
    glActiveTexture(GL_TEXTURE0);
    // TODO: move texture binding here (to glActiveTexture)

    m_isFirstDepthTextureSource = false;
    const int smoothingIterations = 3; // TODO: add UI parameter
    for (int i = 0; i < smoothingIterations; ++i)
    {
        m_isFirstDepthTextureSource = !m_isFirstDepthTextureSource;
        if (m_isFirstDepthTextureSource)
        {
            glBindTexture(GL_TEXTURE_2D, m_depthTexture1);
            GLenum drawBuffer[] = { GL_COLOR_ATTACHMENT1 }; // m_depthTexture2
            glDrawBuffers(1, drawBuffer);
        }
        else
        {
            glBindTexture(GL_TEXTURE_2D, m_depthTexture2);
            GLenum drawBuffer[] = { GL_COLOR_ATTACHMENT0 }; // m_depthTexture1
            glDrawBuffers(1, drawBuffer);
        }

        glDisable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
        
        // Draw (smooth depth)
        renderScreenQuad();

        //glEnable(GL_DEPTH_TEST);
        //glEnable(GL_BLEND);
    }

    // Cleanup
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SmoothRenderer::ExtractNormalsFromDepth()
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);

    m_normalsExtractionShader->use();

    ProjectionInfo projectionInfo = m_camera->getProjectionInfo();
    m_normalsExtractionShader->setUnif("f_x", projectionInfo.n / projectionInfo.r);
    m_normalsExtractionShader->setUnif("f_y", projectionInfo.n / projectionInfo.t);
    m_normalsExtractionShader->setUnif("windowWidth", static_cast<float>(m_windowWidth));
    m_normalsExtractionShader->setUnif("windowHeight", static_cast<float>(m_windowHeight));
    m_normalsExtractionShader->setUnif("depthTexture", 0);

    GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT2 }; // m_normalsTexture
    glDrawBuffers(1, drawBuffers);

    glActiveTexture(GL_TEXTURE0);
    if (m_isFirstDepthTextureSource)
    {
        glBindTexture(GL_TEXTURE_2D, m_depthTexture2);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, m_depthTexture1);
    }

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    // Draw (extract normals from depth)
    renderScreenQuad();

    //glEnable(GL_DEPTH_TEST);
    //glEnable(GL_BLEND);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SmoothRenderer::RenderThicknessTexture(GLuint particlesVAO, int particlesNumber)
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);

    GLenum drawBuffer[] = { GL_COLOR_ATTACHMENT3 }; // m_thicknessTexture
    glDrawBuffers(1, drawBuffer);

    // Clear linear depth texture
    GLfloat zero = 0.f;
    glClearBufferfv(GL_COLOR, 0, &zero);    // Clear z-buffer for m_FBO
    glClear(GL_DEPTH_BUFFER_BIT);

    m_thicknessShader->use();
    m_camera->use(Shader::now());

    ProjectionInfo projectionInfo = m_camera->getProjectionInfo();
    m_thicknessShader->setUnif("windowHeight", m_windowHeight);
    m_thicknessShader->setUnif("projectionTop", projectionInfo.t);
    m_thicknessShader->setUnif("projectionNear", projectionInfo.n);
    m_thicknessShader->setUnif("particleRadius", m_particleRadius);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    // Draw
    glBindVertexArray(particlesVAO);
    glDrawArrays(GL_POINTS, 0, particlesNumber);

    // Cleanup
    glBindVertexArray(0);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

} // namespace rendering