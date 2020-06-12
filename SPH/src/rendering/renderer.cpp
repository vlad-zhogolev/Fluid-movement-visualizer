#include <rendering/renderer.h>
#include <rendering/rendering_parameters.h>
#include <input.h>

#include <cstdlib>

#include <GLFW/glfw3.h>
#include <nanogui/nanogui.h>
#include <nanogui/colorwheel.h>
#include <nanogui/combobox.h>

#include <glm/common.hpp>
#include <glm/gtx/rotate_vector.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb.h>

namespace {

float SKYBOX_VERTICES[] =
{
    -1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,

    -1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,

    1.0f, -1.0f, -1.0f,
    1.0f, -1.0f,  1.0f,
    1.0f,  1.0f,  1.0f,
    1.0f,  1.0f,  1.0f,
    1.0f,  1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,

    -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,
    1.0f,  1.0f,  1.0f,
    1.0f,  1.0f,  1.0f,
    1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,

    -1.0f,  1.0f, -1.0f,
    1.0f,  1.0f, -1.0f,
    1.0f,  1.0f,  1.0f,
    1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,

    -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,
    1.0f, -1.0f,  1.0f
};

unsigned int loadCubemap(char **faces)
{
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrChannels;
    for (unsigned int i = 0; i < 6; i++)
    {
        unsigned char *data = stbi_load(faces[i], &width, &height, &nrChannels, 0);
        if (data)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data
            );
            stbi_image_free(data);
        }
        else
        {
            std::cout << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
            stbi_image_free(data);
        }
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return textureID;
}

} // namespace

Renderer::Renderer(std::unique_ptr<GLFWwindow> glfwWindow)
    : m_glfwWindow(std::move(glfwWindow))
{
    glfwSetWindowSizeLimits(m_glfwWindow.get(), 800, 600, GLFW_DONT_CARE, GLFW_DONT_CARE);
    Init();
}

void Renderer::Init()
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    // NanoGUI initializtion
    m_nanoguiScreen =new nanogui::Screen();
    m_nanoguiScreen->initialize(m_glfwWindow.get(), true);
    m_nanoguiScreen->setSize(Eigen::Vector2i(1000, 750));

    glfwGetFramebufferSize(m_glfwWindow.get(), &m_width, &m_height);
    glViewport(0, 0, m_width, m_height);
    glfwSwapInterval(0);
    glfwSwapBuffers(m_glfwWindow.get());

    m_controlsFormHelper = new nanogui::FormHelper(m_nanoguiScreen);
    const int initialCoordinate = 5;
    m_controlsWindow = m_controlsFormHelper->addWindow(
        Eigen::Vector2i(initialCoordinate, initialCoordinate), 
        "Simulation controls and parameters");  

    //TODO: fix it. Temporarily set fixed width.
    m_controlsWindow->setFixedWidth(274);

    m_controlsFormHelper->addGroup("Simulation indicators");
    m_controlsFormHelper->setFixedSize({ 80, 20 });
    m_controlsFormHelper->addVariable("FPS", m_renderingParams->fps)->setEditable(false);
    m_controlsFormHelper->addVariable("Current frame number", m_input->frameCount)->setEditable(false);
    m_controlsFormHelper->setFixedSize({ 0, 20 });
    
    m_controlsFormHelper->addGroup("Simulation controls");
    auto simulationControl = new nanogui::Widget(m_controlsWindow);
    m_controlsFormHelper->addWidget("", simulationControl);
    simulationControl->setLayout(
        new nanogui::BoxLayout(nanogui::Orientation::Horizontal, nanogui::Alignment::Middle, 2, 8));
    
    const int controlButtonSize = 30;
    auto nextFrameButton = new nanogui::Button(simulationControl, "", ENTYPO_ICON_CONTROLLER_NEXT);
    nextFrameButton->setFixedSize(nanogui::Vector2i(controlButtonSize, controlButtonSize));
    nextFrameButton->setFlags(nanogui::Button::NormalButton);

    auto testRunOrStopButton = new nanogui::Button(simulationControl, "", ENTYPO_ICON_CONTROLLER_PLAY);
    testRunOrStopButton->setFlags(nanogui::Button::ToggleButton);
    testRunOrStopButton->setFixedSize(nanogui::Vector2i(controlButtonSize, controlButtonSize));
    
    auto restartButton = new nanogui::Button(simulationControl, "", ENTYPO_ICON_CCW);
    restartButton->setFlags(nanogui::Button::NormalButton);

    auto particleSourceSelector = new nanogui::Widget(m_controlsWindow);
    particleSourceSelector->setLayout(
        new nanogui::BoxLayout(nanogui::Orientation::Horizontal, nanogui::Alignment::Middle, 2, 8));
    m_controlsFormHelper->addWidget("Source", particleSourceSelector);
    auto particleSourceComboBox = new nanogui::ComboBox(particleSourceSelector, { "Cube", "Sphere" });

    particleSourceComboBox->setCallback([this](int index) {
        ParticleSource sourceType;
        if (index == 0)
            sourceType = ParticleSource::Cube;
        else if (index == 1)
            sourceType = ParticleSource::Sphere;

        m_simulationParams->SetParticlesSource(sourceType);
    });

    auto domainSelector = new nanogui::Widget(m_controlsWindow);
    domainSelector->setLayout(
        new nanogui::BoxLayout(nanogui::Orientation::Horizontal, nanogui::Alignment::Middle, 2, 8));
    m_controlsFormHelper->addWidget("Domain", domainSelector);
    auto domainComboBox = new nanogui::ComboBox(domainSelector, { "Small", "Medium", "Large", "Stretched" });

    domainComboBox->setCallback([this](int index) {
        SimulationDomainSize size;
        if (index == 0)
            size = SimulationDomainSize::Small;
        else if (index == 1)
            size = SimulationDomainSize::Medium;
        else if (index == 2)
            size = SimulationDomainSize::Large;
        else if (index == 3)
            size = SimulationDomainSize::Stretched;

        m_simulationParams->SetDomainSize(size);
    });
    
    auto xSetter = [this](const float& value) {
        m_simulationParams->SetStartX(value);
    };
    auto xGetter = [this]() -> float {
        return m_simulationParams->fluidStartPosition.x;
    };
    auto startPositionX = m_controlsFormHelper->addVariable<float>("Start position, x", xSetter, xGetter);
    startPositionX->setSpinnable(true);

    auto ySetter = [this](const float& value) {
        m_simulationParams->SetStartY(value);
    };
    auto yGetter = [this]() -> float {
        return m_simulationParams->fluidStartPosition.y;
    };
    auto startPositionY = m_controlsFormHelper->addVariable<float>("Start position, y", ySetter, yGetter);
    startPositionY->setSpinnable(true);

    auto zSetter = [this](const float& value) {
        m_simulationParams->SetStartZ(value);
    };
    auto zGetter = [this]() -> float {
        return m_simulationParams->fluidStartPosition.z;
    };
    auto startPositionZ = m_controlsFormHelper->addVariable<float>("Start position, z", zSetter, zGetter);
    startPositionZ->setSpinnable(true);

    auto fluidSizeSetter = [this](const int& value) {
        m_simulationParams->SetFluidSize(value);
    };
    auto fluidSizeGetter = [this]() -> int {
        return m_simulationParams->GetFluidSize();
    };
    auto fluidSizeVariable = m_controlsFormHelper->addVariable<int>("Fluid size", fluidSizeSetter, fluidSizeGetter);
    fluidSizeVariable->setMinMaxValues(SimulationParameters::FLUID_SIZE_MIN, SimulationParameters::FLUID_SIZE_MAX);
    fluidSizeVariable->setSpinnable(true);

    m_positionVariables = {
        startPositionX,
        startPositionY,
        startPositionZ,
        fluidSizeVariable
    };

    m_switchOffRestart = {
        startPositionX,
        startPositionY,
        startPositionZ,
        fluidSizeVariable,
        domainComboBox,
        particleSourceComboBox
    };

    nextFrameButton->setCallback([this, domainComboBox]() {
        domainComboBox->setEnabled(false);
        m_simulationParams->SetCommand(SimulationCommand::StepOneFrame);
    });

    testRunOrStopButton->setChangeCallback(
        [this, 
        testRunOrStopButton, 
        startPositionX, 
        startPositionY,
        startPositionZ,
        fluidSizeVariable, 
        domainComboBox]
        (bool isPressed)
    {
        if (isPressed)
        {
            testRunOrStopButton->setIcon(ENTYPO_ICON_CONTROLLER_STOP);
            SetStartSettingsEnabled(false);
            m_simulationParams->SetCommand(SimulationCommand::Run);
        }
        else
        {
            testRunOrStopButton->setIcon(ENTYPO_ICON_CONTROLLER_PLAY);
            m_simulationParams->SetCommand(SimulationCommand::Pause);
        }
    });

    restartButton->setCallback(
        [this,
        testRunOrStopButton,
        startPositionX,
        startPositionY,
        startPositionZ,
        fluidSizeVariable,
        domainComboBox]()
    {
        m_simulationParams->SetCommand(SimulationCommand::Restart);

        // Enable start button
        testRunOrStopButton->setPushed(false);
        testRunOrStopButton->setIcon(ENTYPO_ICON_CONTROLLER_PLAY);

        SetStartSettingsEnabled(true);
    });

    m_scrollFormHelper = new nanogui::ScrollFormHelper(m_nanoguiScreen);
    auto scrollWindowInitialCoordinates = Eigen::Vector2i(0, 0);
    m_scrollWindow = m_scrollFormHelper->addWindow(
        scrollWindowInitialCoordinates, "Simulation and rendering parameters");

    m_scrollFormHelper->addGroup("Gravity acceleration");

    auto gravityXSetter = [this](const float& value) {
        m_simulationParams->SetGravityX(value);
    };
    auto gravityXGetter = [this]() -> float {
        return m_simulationParams->GetGravityX();
    };
    auto* gravityX = m_scrollFormHelper->addVariable<float>("Gravity, x", gravityXSetter, gravityXGetter);
    gravityX->setMinMaxValues(SimulationParameters::GRAVITY_MIN, SimulationParameters::GRAVITY_MAX);

    auto gravityYSetter = [this](const float& value) {
        m_simulationParams->SetGravityY(value);
    };
    auto gravityYGetter = [this]() -> float {
        return m_simulationParams->GetGravityY();
    };
    auto* gravityY = m_scrollFormHelper->addVariable<float>("Gravity, y", gravityYSetter, gravityYGetter);
    gravityY->setMinMaxValues(SimulationParameters::GRAVITY_MIN, SimulationParameters::GRAVITY_MAX);

    auto gravityZSetter = [this](const float& value) {
        m_simulationParams->SetGravityZ(value);
    };
    auto gravityZGetter = [this]() -> float {
        return m_simulationParams->GetGravityZ();
    };
    auto* gravityZ = m_scrollFormHelper->addVariable<float>("Gravity, z", gravityZSetter, gravityZGetter);
    gravityZ->setMinMaxValues(SimulationParameters::GRAVITY_MIN, SimulationParameters::GRAVITY_MAX);

    m_scrollFormHelper->addGroup("Fluid parameters");

    //m_scrollFormHelper->addVariable("Change", m_simulationParams->change);
    auto& setSubstepsNumberCallback = [this](const int& value) {
        m_simulationParams->SetSubstepsNumber(value);
    };
    auto& getSubstepsNumberCallback = [this]() -> int {
        return m_simulationParams->GetDensity();
    };
    auto* substepsNumberVar = m_scrollFormHelper->addVariable<int>("Substeps number", setSubstepsNumberCallback, getSubstepsNumberCallback);
    substepsNumberVar->setMinMaxValues(SimulationParameters::SUBSTEPS_NUMBER_MIN, SimulationParameters::SUBSTEPS_NUMBER_MAX);

    auto& setRestDensityCallback = [this](const float& value) {
        m_simulationParams->SetDensity(value);
    };
    auto& getDensityCallback = [this]() -> float { 
        return m_simulationParams->GetDensity(); 
    };
    auto* densityVar = m_scrollFormHelper->addVariable<float>("Rest density", setRestDensityCallback, getDensityCallback);
    densityVar->setMinMaxValues(SimulationParameters::DENSITY_MIN, SimulationParameters::DENSITY_MAX);

    auto& setKernelRadiusCallback = [this](const float& value) {
        m_simulationParams->SetKernelRadius(value);
    };
    auto& getKernelRadiusCallback = [this]() -> float {
        return m_simulationParams->GetKernelRadius();
    };
    auto* kernelRadiusVar = m_scrollFormHelper->addVariable<float>("Kernel radius", setKernelRadiusCallback, getKernelRadiusCallback);
    kernelRadiusVar->setMinMaxValues(SimulationParameters::KERNEL_RADIUS_MIN, SimulationParameters::KERNEL_RADIUS_MAX);

    auto& setDeltaTimeCallback = [this](const float& value) {
        m_simulationParams->SetDeltaTime(value);
    };
    auto& getDeltaTimeCallback = [this]() -> float {
        return m_simulationParams->GetDeltaTime();
    };
    auto* deltaTimeVar = m_scrollFormHelper->addVariable<float>("Delta time", setDeltaTimeCallback, getDeltaTimeCallback);
    deltaTimeVar->setMinMaxValues(SimulationParameters::DELTA_TIME_MIN, SimulationParameters::DELTA_TIME_MAX);

    auto& setLambdaEpsilonCallback = [this](const float& value) {
        m_simulationParams->SetLambdaEpsilon(value);
    };
    auto& getLambdaEpsilonCallback = [this]() -> float {
        return m_simulationParams->GetLambdaEpsilon();
    };
    auto* lambdaEpsilonVar = m_scrollFormHelper->addVariable("Lambda epsilon", m_simulationParams->relaxationParameter);
    lambdaEpsilonVar->setMinMaxValues(SimulationParameters::RELAXATION_PARAM_MIN, SimulationParameters::RELAXATION_PARAM_MAX);

    auto& setDeltaQCallback = [this](const float& value) {
        m_simulationParams->SetDeltaQ(value);
    };
    auto& getDeltaQCallback = [this]() -> float {
        return m_simulationParams->GetDeltaQ();
    };
    auto* deltaQVar = m_scrollFormHelper->addVariable<float>("DeltaQ", setDeltaQCallback, getDeltaQCallback);
    deltaQVar->setMinMaxValues(SimulationParameters::DELTA_Q_MIN, SimulationParameters::DELTA_Q_MAX);

    auto& setCorrectionCoefCallback = [this](const float& value) {
        m_simulationParams->SetCorrectionCoefficient(value);
    };
    auto& getCorrectionCoefCallback = [this]() -> float {
        return m_simulationParams->GetCorrectionCoefficient();
    };
    auto* correctionCoefVar = m_scrollFormHelper->addVariable<float>("Correction coefficient", setCorrectionCoefCallback, getCorrectionCoefCallback);
    correctionCoefVar->setMinMaxValues(SimulationParameters::CORRECTION_COEF_MIN, SimulationParameters::CORRECTION_COEF_MAX);

    auto& setCorrectionPowerCallback = [this](const float& value) {
        m_simulationParams->SetCorrectionPower(value);
    };
    auto& getCorrectionPowerCallback = [this]() -> float {
        return m_simulationParams->GetCorrectionPower();
    };
    auto* correctionPowerVar = m_scrollFormHelper->addVariable<float>("Correction power", setCorrectionPowerCallback, getCorrectionPowerCallback);
    correctionPowerVar->setMinMaxValues(SimulationParameters::CORRECTION_POWER_MIN, SimulationParameters::CORRECTION_POWER_MAX);

    auto& setXSPHCoefCallback = [this](const float& value) {
        m_simulationParams->SetXSPHCoefficient(value);
    };
    auto& getXSPHCoefCallback = [this]() -> float {
        return m_simulationParams->GetXSPHCoefficient();
    };
    auto* xsphCoefVar = m_scrollFormHelper->addVariable<float>("XSPH coefficient", setXSPHCoefCallback, getXSPHCoefCallback);
    xsphCoefVar->setMinMaxValues(SimulationParameters::XSPH_COEF_MIN, SimulationParameters::XSPH_COEF_MAX);

    auto& setViscosityIterCallback = [this](const int& value) {
        m_simulationParams->SetViscosityIter(value);
    };
    auto& getViscosityIterCallback = [this]() -> int {
        return m_simulationParams->GetViscosityIter();
    };
    auto* viscosityIterationsVar = m_scrollFormHelper->addVariable<int>("Viscosity iterations", setViscosityIterCallback, getViscosityIterCallback);
    viscosityIterationsVar->setMinMaxValues(SimulationParameters::XSPH_ITERATIONS_MIN, SimulationParameters::XSPH_ITERATIONS_MAX);

    auto& setVorticityCallback = [this](const float& value) {
        m_simulationParams->SetVorticity(value);
    };
    auto& getVorticityCallback = [this]() -> float {
        return m_simulationParams->GetVorticity();
    };
    auto* vorticityCoefVar = m_scrollFormHelper->addVariable<float>("Vorticity coefficient", setVorticityCallback, getVorticityCallback);
    vorticityCoefVar->setMinMaxValues(SimulationParameters::VORTICITY_MIN, SimulationParameters::VORTICITY_MAX);

    m_scrollFormHelper->addGroup("Rendering parameters");

    auto& setSmoothingIterCallback = [this](const int& value) {
        m_renderingParams->SetSmoothingIter(value);
    };
    auto& getSmoothingIterCallback = [this]() -> int {
        return m_renderingParams->GetSmoothingIter();
    };
    auto* smoothingIterations = m_scrollFormHelper->addVariable<int>(
        "Smoothing iterations", setSmoothingIterCallback, getSmoothingIterCallback);
    smoothingIterations->setMinMaxValues(
        RenderingParameters::SMOOTH_STEPS_NUMBER_MIN, RenderingParameters::SMOOTH_STEPS_NUMBER_MAX);
    auto* fluidRefractionIndex = m_scrollFormHelper->addVariable(
        "Refraction index", m_renderingParams->fluidRefractionIndex);
    auto* particleRadius = m_scrollFormHelper->addVariable("Particle radius", m_renderingParams->particleRadius);

    auto* colorWheel = new nanogui::ColorWheel(m_scrollFormHelper->wrapper());
    m_scrollFormHelper->addWidget("Fluid color", colorWheel);
    colorWheel->setColor(nanogui::Color(
        m_renderingParams->fluidColor.r, m_renderingParams->fluidColor.g, m_renderingParams->fluidColor.b, 1.0f));
    colorWheel->setCallback([this](const nanogui::Color& color) {
        m_renderingParams->fluidColor.r = color.r();
        m_renderingParams->fluidColor.g = color.g();
        m_renderingParams->fluidColor.b = color.b();
        std::cout 
            << "Fluid color,"
            << " r: " << m_renderingParams->fluidColor.r 
            << " g: " << m_renderingParams->fluidColor.g 
            << " b: " << m_renderingParams->fluidColor.b 
            << std::endl;
    });

    auto& setAttenuationRed = [this](const float& value) {
        m_renderingParams->SetAttenuationRed(value);
    };
    auto& getAttenuationRed = [this]() -> float {
        return m_renderingParams->GetAttenuationRed();
    };
    auto* attenuationRed = m_scrollFormHelper->addVariable<float>("Attenuation, red", setAttenuationRed, getAttenuationRed);
    attenuationRed->setMinMaxValues(
        RenderingParameters::ATTENUATION_COEFFICIENT_MIN, RenderingParameters::ATTENUATION_COEFFICIENT_MAX);

    auto& setAttenuationGreen = [this](const float& value) {
        m_renderingParams->SetAttenuationGreen(value);
    };
    auto& getAttenuationGreen = [this]() -> float {
        return m_renderingParams->GetAttenuationGreen();
    };
    auto* attenuationGreen = m_scrollFormHelper->addVariable<float>("Attenuation, green", setAttenuationGreen, getAttenuationGreen);
    attenuationGreen->setMinMaxValues(
        RenderingParameters::ATTENUATION_COEFFICIENT_MIN, RenderingParameters::ATTENUATION_COEFFICIENT_MAX);

    auto& setAttenuationBlue = [this](const float& value) {
        m_renderingParams->SetAttenuationBlue(value);
    };
    auto& getAttenuationBlue = [this]() -> float {
        return m_renderingParams->GetAttenuationBlue();
    };
    auto* attenuationBlue = m_scrollFormHelper->addVariable<float>("Attenuation, blue", setAttenuationBlue, getAttenuationBlue);
    attenuationBlue->setMinMaxValues(
        RenderingParameters::ATTENUATION_COEFFICIENT_MIN, RenderingParameters::ATTENUATION_COEFFICIENT_MAX);

    m_nanoguiScreen->performLayout();
    m_nanoguiScreen->setVisible(true);
    

    m_scrollWindow->setPosition({ 5, 2 * initialCoordinate + m_controlsWindow->height() });

    BindGLFWCallbacks();

    const glm::vec3 cameraPosition{ 1.f, -5.f, 2.f };
    const glm::vec3 cameraFocus{ 0, 0, 1.5f };
    float aspect = static_cast<float>(m_width) / m_height;
    m_camera = std::make_shared<Camera>(cameraPosition, cameraFocus, aspect);

    m_boundaryShader = std::make_unique<Shader>(Path("shaders/boundary.vert"), Path("shaders/boundary.frag"));
    m_particlesShader = std::make_unique<Shader>(Path("shaders/particle.vert"), Path("shaders/particle.frag"));
    m_skyboxShader = std::make_unique<Shader>(Path("shaders/skybox.vert"), Path("shaders/skybox.frag"));

    char *sky_faces[] =
    {
        "skybox/checkerboard/checkerboard.jpg",
        "skybox/checkerboard/checkerboard.jpg",
        "skybox/checkerboard/checkerboard.jpg",
        "skybox/checkerboard/checkerboard.jpg",
        "skybox/checkerboard/checkerboard.jpg",
        "skybox/checkerboard/checkerboard.jpg"
    };
    m_skyboxTexture = loadCubemap(sky_faces);
    
    glGenVertexArrays(1, &m_particlesVAO);

    glGenVertexArrays(1, &m_boundariesVAO);
    glGenBuffers(1, &m_boundariesVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_boundariesVBO);
    glBufferData(GL_ARRAY_BUFFER, 12 * 2 * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glBindVertexArray(m_boundariesVAO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glGenVertexArrays(1, &m_skyboxVAO);
    glGenBuffers(1, &m_skyboxVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, 6 * 2 * 3 * 3 * sizeof(float), SKYBOX_VERTICES, GL_STATIC_DRAW);
    glBindVertexArray(m_skyboxVAO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    m_smoothRenderer = std::make_unique<rendering::SmoothRenderer>(m_width, m_height, m_camera, m_skyboxTexture);
}

void Renderer::SetStartSettingsEnabled(bool isEnabled)
{
    for (auto widget : m_switchOffRestart)
    {
        widget->setEnabled(isEnabled);
    }
    for (auto textBox : m_positionVariables)
    {
        textBox->setEditable(isEnabled);
    }
}


void Renderer::WindowSizeCallback(GLFWwindow* window, int width, int height)
{
    const int minWidth = 800;
    const int minHeight = 600;
    width = std::max(width, minWidth);
    height = std::max(height, minHeight);

    const float widthChangeRatio = static_cast<float>(width) / m_width;
    const float heightChangeRatio = static_cast<float>(height) / m_height;

    nanogui::Vector2i oldPosition = m_controlsWindow->position();

    m_width = width;
    m_height = height;
    glViewport(0, 0, width, height);
    m_camera->setAspect((float)width / height);
    m_nanoguiScreen->resizeCallbackEvent(width, height);

    nanogui::Vector2i newPosition = { oldPosition[0] * widthChangeRatio, oldPosition[1] * heightChangeRatio };
    const int margin = 0;

    // std::cout << "Window width: " << m_controlsWindow->width() << " height: " << m_controlsWindow->height() << std::endl;
    // std::cout << "Screen width: " << m_nanoguiScreen->width() << " height: " << m_nanoguiScreen->height() << std::endl;
    // std::cout << std::endl;

    if (newPosition[0] + m_controlsWindow->width() > m_nanoguiScreen->width())
    {
        newPosition[0] = m_nanoguiScreen->width() - m_controlsWindow->width() - margin;
    }
    if (newPosition[1] + m_controlsWindow->height() > m_nanoguiScreen->height())
    {
        newPosition[1] = m_nanoguiScreen->height() - m_controlsWindow->height() - margin;
    }
    m_controlsWindow->setPosition(newPosition);

    m_smoothRenderer->HandleWindowResolutionChange(width, height);
}

void Renderer::MouseButtonCallback(GLFWwindow *w, int button, int action, int mods)
{
    if (m_nanoguiScreen->mouseButtonCallbackEvent(button, action, mods)) return;

    Input::Pressed buttonState = action == GLFW_PRESS ? Input::DOWN : Input::UP;
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        m_input->left_mouse = buttonState;
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        m_input->right_mouse = buttonState;
    }
    if (button == GLFW_MOUSE_BUTTON_MIDDLE)
    {
        m_input->mid_mouse = buttonState;
    }
}

void Renderer::MouseMoveCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (m_nanoguiScreen->cursorPosCallbackEvent(xpos, ypos)) return;

    m_input->UpdateMousePosition(glm::vec2(xpos, ypos));

    // Camera rotation
    glm::vec2 mouseDiff = m_input->getMouseDiff();

    if (m_input->left_mouse == Input::DOWN)
    {
        m_camera->rotate(mouseDiff);
    }

    /* Panning */
    if (m_input->right_mouse == Input::DOWN)
    {
        m_camera->pan(mouseDiff);
    }
}

void Renderer::KeyCallback(GLFWwindow *w, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_V && action == GLFW_RELEASE)
    {
        m_controlsWindow->setVisible(!m_controlsWindow->visible());
        m_scrollWindow->setVisible(!m_scrollWindow->visible());
    }
    else if (key == GLFW_KEY_N && action == GLFW_RELEASE)
    {
        auto pos = m_camera->getPos();
        auto front = m_camera->getFront();
        auto up = m_camera->getUp();
        std::cout << "camera pos: " << pos.x << " " << pos.y << " " << pos.z << std::endl;
        std::cout << "camera front: " << front.x << " " << front.y << " " << front.z << std::endl;
        std::cout << "camera pos: " << up.x << " " << up.y << " " << up.z << std::endl;
        m_camera->setPos(glm::vec3(-0.789017, 1.1729, 0.948009));
        m_camera->setFront(glm::vec3(0.536186, -0.802564, -0.35533));
        m_camera->setUp(glm::vec3(0.191918, -0.287263, 0.938425));
    }
    else
    {
        m_nanoguiScreen->keyCallbackEvent(key, scancode, action, mods);
    }
}

void Renderer::MouseScrollCallback(GLFWwindow* w, float dx, float dy)
{
    if (m_nanoguiScreen->scrollCallbackEvent(dx, dy))
    {
        return;
    }
    m_camera->zoom(dy);
}

void Renderer::CharCallback(GLFWwindow *w, unsigned int codepoint)
{
    m_nanoguiScreen->charCallbackEvent(codepoint);
}

void Renderer::BindGLFWCallbacks()
{

    glfwSetWindowUserPointer(m_glfwWindow.get(), this);

    glfwSetWindowSizeCallback(m_glfwWindow.get(), [](GLFWwindow *win, int width, int height) {
        ((Renderer*)(glfwGetWindowUserPointer(win)))->WindowSizeCallback(win, width, height);
    });

    glfwSetCursorPosCallback(m_glfwWindow.get(), [](GLFWwindow *w, double xpos, double ypos) {
        ((Renderer*)(glfwGetWindowUserPointer(w)))->MouseMoveCallback(w, xpos, ypos);
    });

    glfwSetMouseButtonCallback(m_glfwWindow.get(), [](GLFWwindow* w, int button, int action, int mods) {
        ((Renderer*)(glfwGetWindowUserPointer(w)))->MouseButtonCallback(w, button, action, mods);
    });

    glfwSetScrollCallback(m_glfwWindow.get(), [](GLFWwindow *w, double dx, double dy) {
        ((Renderer*)(glfwGetWindowUserPointer(w)))->MouseScrollCallback(w, dx, dy);
    });

    glfwSetKeyCallback(m_glfwWindow.get(),
        [](GLFWwindow *w, int key, int scancode, int action, int mods) {
        ((Renderer*)(glfwGetWindowUserPointer(w)))->KeyCallback(w, key, scancode, action, mods);
    });

    glfwSetCharCallback(m_glfwWindow.get(),
        [](GLFWwindow *w, unsigned int codepoint) {
        ((Renderer*)(glfwGetWindowUserPointer(w)))->CharCallback(w, codepoint);
    });
}

void Renderer::RenderImpl()
{
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (m_skyboxShader->loaded())
    {
        glDepthMask(GL_FALSE);
        m_skyboxShader->use();
        m_camera->use(Shader::now(), true);
        glBindVertexArray(m_skyboxVAO);
        glBindTexture(GL_TEXTURE_CUBE_MAP, m_skyboxTexture);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glDepthMask(GL_TRUE);
    }
    
    const bool smoothFluid = true;
    if (m_particlesShader->loaded() && !smoothFluid)
    {
        m_particlesShader->use();
        m_camera->use(Shader::now());
        m_particlesShader->setUnif("color", glm::vec4(1.f, 0.f, 0.f, .1f));
        m_particlesShader->setUnif("pointRadius", SimulationParameters::GetInstance().kernelRadius);
        m_particlesShader->setUnif("pointScale", 500.f);
        glBindVertexArray(m_particlesVAO);
        glDrawArrays(GL_POINTS, 0, m_nparticle);
    }
    else if (smoothFluid)
    {
        m_smoothRenderer->Render(m_particlesVAO, m_nparticle);
    }

    if (m_boundaryShader->loaded() && m_simulationParams->change)
    {
        m_boundaryShader->use();
        m_camera->use(Shader::now());
        glBindVertexArray(m_boundariesVAO);
        m_boundaryShader->setUnif("color", glm::vec4(1.f, 1.f, 1.f, 1.f));
        glDrawArrays(GL_LINES, 0, 12 * 2);
    }
}

Renderer::~Renderer() {}

void Renderer::Render(unsigned int positions, unsigned int iid, int nparticle)
{
    d_iid = iid;
    m_particlesPositions = positions;
    m_nparticle = nparticle;

    m_upperBoundary = m_simulationParams->GetUpperBoundary();
    m_lowerBoundary = m_simulationParams->GetLowerBoundary();

    glBindVertexArray(m_particlesVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_particlesPositions);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, d_iid);
    glVertexAttribIPointer(1, 1, GL_UNSIGNED_INT, 0, (void*)0);
    glEnableVertexAttribArray(1);

    float particleRadius = m_simulationParams->GetParticleRadius();
    float3 up = m_upperBoundary + particleRadius;
    float3 low = m_lowerBoundary - particleRadius;
    float x1 = fmin(up.x, low.x);
    float x2 = fmax(up.x, low.x);
    float y1 = fmin(up.y, low.y);
    float y2 = fmax(up.y, low.y);
    float z1 = fmin(up.z, low.z);
    float z2 = fmax(up.z, low.z);

    glm::vec3 lines[][2] = 
    {
        { glm::vec3(x1, y1, z1), glm::vec3(x2, y1, z1) },
        { glm::vec3(x1, y1, z2), glm::vec3(x2, y1, z2) },
        { glm::vec3(x1, y2, z1), glm::vec3(x2, y2, z1) },
        { glm::vec3(x1, y2, z2), glm::vec3(x2, y2, z2) },

        { glm::vec3(x1, y1, z1), glm::vec3(x1, y2, z1) },
        { glm::vec3(x1, y1, z2), glm::vec3(x1, y2, z2) },
        { glm::vec3(x2, y1, z1), glm::vec3(x2, y2, z1) },
        { glm::vec3(x2, y1, z2), glm::vec3(x2, y2, z2) },

        { glm::vec3(x1, y1, z1), glm::vec3(x1, y1, z2) },
        { glm::vec3(x1, y2, z1), glm::vec3(x1, y2, z2) },
        { glm::vec3(x2, y1, z1), glm::vec3(x2, y1, z2) },
        { glm::vec3(x2, y2, z1), glm::vec3(x2, y2, z2) } 
    };

    glBindBuffer(GL_ARRAY_BUFFER, m_boundariesVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(lines), lines);

    m_controlsFormHelper->refresh();
    m_scrollFormHelper->refresh();

    if (!glfwWindowShouldClose(m_glfwWindow.get()))
    {
        glfwPollEvents();
        RenderImpl();
        m_nanoguiScreen->drawContents();
        m_nanoguiScreen->drawWidgets();
        glfwSwapBuffers(m_glfwWindow.get());
    }
    else fexit(0);
}

void Renderer::SetBoundaries(const float3 & upperBoundary, const float3 & lowerBoundary)
{
    m_upperBoundary = upperBoundary;
    m_lowerBoundary = lowerBoundary;
}
