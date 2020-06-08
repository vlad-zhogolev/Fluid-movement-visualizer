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

#include <thread>

extern float SKYBOX_VERTICES[];
extern float GROUND_VERTICES[];
static unsigned int loadCubemap(char **faces);

void Renderer::Init()
{
    const glm::vec3 cameraPosition{ 1.f, -5.f, 2.f };
    const glm::vec3 cameraFocus{ 0, 0, 1.5f };
    init(cameraPosition, cameraFocus);
}

void Renderer::init(const glm::vec3 &cam_pos, const glm::vec3 &cam_focus)
{
    SimulationParameters& simulationParameters = SimulationParameters::GetInstance();
    RenderingParameters& renderingParameters = RenderingParameters::GetInstance();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    // NanoGUI initializtion
	m_nanoguiScreen =new nanogui::Screen();
	m_nanoguiScreen->initialize(m_glfwWindow.get(), true);
	m_nanoguiScreen->setSize(Eigen::Vector2i(1000, 750));

	int width_, height_;
	glfwGetFramebufferSize(m_glfwWindow.get(), &width_, &height_);
	m_width = width_; m_height = height_;
	glViewport(0, 0, width_, height_);
	glfwSwapInterval(0);
	glfwSwapBuffers(m_glfwWindow.get());

	m_formHelper = new nanogui::FormHelper(m_nanoguiScreen);
    const int initialCoordinate = 5;
	m_nanoguiWindow = m_formHelper->addWindow(
        Eigen::Vector2i(initialCoordinate, initialCoordinate), 
        "Simulation controls and parameters");  

    //TODO: fix it. Temporarily set fixed width.
    m_nanoguiWindow->setFixedWidth(274);

    m_formHelper->addGroup("Simulation indicators");
    m_formHelper->setFixedSize({ 80, 20 });
    m_formHelper->addVariable("FPS", renderingParameters.fps)->setEditable(false);
    m_formHelper->addVariable("Current frame number", m_input->frameCount)->setEditable(false);
    m_formHelper->setFixedSize({ 0, 20 });
    
    m_formHelper->addGroup("Simulation controls");
    auto simulationControl = new nanogui::Widget(m_nanoguiWindow);
    m_formHelper->addWidget("", simulationControl);
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

    auto particleSourceSelector = new nanogui::Widget(m_nanoguiWindow);
    particleSourceSelector->setLayout(
        new nanogui::BoxLayout(nanogui::Orientation::Horizontal, nanogui::Alignment::Middle, 2, 8));
    m_formHelper->addWidget("Source", particleSourceSelector);
    auto particleSourceComboBox = new nanogui::ComboBox(particleSourceSelector, { "Cube", "Sphere" });

    particleSourceComboBox->setCallback([this](int index) {
        ParticleSource sourceType;
        if (index == 0)
            sourceType = ParticleSource::Cube;
        else if (index == 1)
            sourceType = ParticleSource::Sphere;

        m_simulationParams->SetParticlesSource(sourceType);
    });

    auto domainSelector = new nanogui::Widget(m_nanoguiWindow);
    domainSelector->setLayout(
        new nanogui::BoxLayout(nanogui::Orientation::Horizontal, nanogui::Alignment::Middle, 2, 8));
    m_formHelper->addWidget("Domain", domainSelector);
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
    auto startPositionX = m_formHelper->addVariable<float>("Start position, x", xSetter, xGetter);
    startPositionX->setSpinnable(true);

    auto ySetter = [this](const float& value) {
        m_simulationParams->SetStartY(value);
    };
    auto yGetter = [this]() -> float {
        return m_simulationParams->fluidStartPosition.y;
    };
    auto startPositionY = m_formHelper->addVariable<float>("Start position, y", ySetter, yGetter);
    startPositionY->setSpinnable(true);

    auto zSetter = [this](const float& value) {
        m_simulationParams->SetStartZ(value);
    };
    auto zGetter = [this]() -> float {
        return m_simulationParams->fluidStartPosition.z;
    };
    auto startPositionZ = m_formHelper->addVariable<float>("Start position, z", zSetter, zGetter);
    startPositionZ->setSpinnable(true);

    auto fluidSizeSetter = [this](const int& value) {
        m_simulationParams->SetFluidSize(value);
    };
    auto fluidSizeGetter = [this]() -> int {
        return m_simulationParams->GetFluidSize();
    };
    auto fluidSizeVariable = m_formHelper->addVariable<int>("Fluid size", fluidSizeSetter, fluidSizeGetter);
    fluidSizeVariable->setMinMaxValues(1, 50);
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

    m_scrollFormHelper->addVariable("Gravity, x", simulationParameters.gravity.x);
    m_scrollFormHelper->addVariable("Gravity, y", simulationParameters.gravity.y);
    m_scrollFormHelper->addVariable("Gravity, z", simulationParameters.gravity.z);

    m_scrollFormHelper->addGroup("Fluid parameters");

    //m_scrollFormHelper->addVariable("Change", simulationParameters.change);
    m_scrollFormHelper->addVariable("Substeps number", simulationParameters.substepsNumber)->setSpinnable(true);

    auto& setRestDensityCallback = [this](const float& value) {
        m_simulationParams->SetDensity(value);
    };
    auto& getDensityCallback = [this]() -> float { 
        return m_simulationParams->GetDensity(); 
    };
    m_scrollFormHelper->addVariable<float>("Rest density", setRestDensityCallback, getDensityCallback);
    m_scrollFormHelper->addVariable("Kernel radius", simulationParameters.kernelRadius);
    m_scrollFormHelper->addVariable("Delta time", simulationParameters.deltaTime);
    m_scrollFormHelper->addVariable("Lambda epsilon", simulationParameters.relaxationParameter);
    m_scrollFormHelper->addVariable("DeltaQ", simulationParameters.deltaQ);
    m_scrollFormHelper->addVariable("Correction coefficient", simulationParameters.correctionCoefficient);
    m_scrollFormHelper->addVariable("Correction power", simulationParameters.correctionPower);
    m_scrollFormHelper->addVariable("XSPH coefficient", simulationParameters.c_XSPH);
    m_scrollFormHelper->addVariable("Viscosity iterations", simulationParameters.viscosityIterations);
    m_scrollFormHelper->addVariable("Vorticity coefficient", simulationParameters.vorticityEpsilon);

    m_scrollFormHelper->addGroup("Rendering parameters");
    auto* smoothingIterations = m_scrollFormHelper->addVariable(
        "Smoothing iterations", renderingParameters.smoothStepsNumber);
    smoothingIterations->setMinMaxValues(
        RenderingParameters::SMOOTH_STEPS_NUMBER_MIN, RenderingParameters::SMOOTH_STEPS_NUMBER_MAX);
    auto* fluidRefractionIndex = m_scrollFormHelper->addVariable(
        "Refraction index", renderingParameters.fluidRefractionIndex);
    auto* particleRadius = m_scrollFormHelper->addVariable("Particle radius", renderingParameters.particleRadius);

    auto* colorWheel = new nanogui::ColorWheel(m_scrollFormHelper->wrapper());
    m_scrollFormHelper->addWidget("Fluid color", colorWheel);
    colorWheel->setColor(nanogui::Color(
        renderingParameters.fluidColor.r, renderingParameters.fluidColor.g, renderingParameters.fluidColor.b, 1.0f));
    colorWheel->setCallback([](const nanogui::Color& color) {
        RenderingParameters& renderingParameters = RenderingParameters::GetInstance();
        renderingParameters.fluidColor.r = color.r();
        renderingParameters.fluidColor.g = color.g();
        renderingParameters.fluidColor.b = color.b();
        std::cout 
            << "Fluid color,"
            << " r: " << renderingParameters.fluidColor.r 
            << " g: " << renderingParameters.fluidColor.g 
            << " b: " << renderingParameters.fluidColor.b 
            << std::endl;
    });

    auto* attenuationRed = m_scrollFormHelper->addVariable("Attenuation, red", renderingParameters.attenuationCoefficients.r);
    attenuationRed->setMinMaxValues(
        RenderingParameters::ATTENUATION_COEFFICIENT_MIN, RenderingParameters::ATTENUATION_COEFFICIENT_MAX);
    auto* attenuationGreen = m_scrollFormHelper->addVariable("Attenuation, green", renderingParameters.attenuationCoefficients.g);
    attenuationGreen->setMinMaxValues(
        RenderingParameters::ATTENUATION_COEFFICIENT_MIN, RenderingParameters::ATTENUATION_COEFFICIENT_MAX);
    auto* attenuationBlue = m_scrollFormHelper->addVariable("Attenuation, blue", renderingParameters.attenuationCoefficients.b);
    attenuationBlue->setMinMaxValues(
        RenderingParameters::ATTENUATION_COEFFICIENT_MIN, RenderingParameters::ATTENUATION_COEFFICIENT_MAX);

    m_nanoguiScreen->performLayout();
	m_nanoguiScreen->setVisible(true);
	

    m_scrollWindow->setPosition({ 5, 2 * initialCoordinate + m_nanoguiWindow->height() });

	__binding();

    float aspect = (float) width_ / height_;
	m_camera = std::make_shared<Camera>(cam_pos, cam_focus, aspect);
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
	
	glGenVertexArrays(1, &d_vao);

    glGenVertexArrays(1, &d_bbox_vao);
	glGenBuffers(1, &d_bbox_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, d_bbox_vbo);
	glBufferData(GL_ARRAY_BUFFER, 12 * 2 * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glBindVertexArray(d_bbox_vao);
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


void Renderer::__window_size_callback(GLFWwindow* window, int width, int height)
{
    const int minWidth = 800;
    const int minHeight = 600;
    width = std::max(width, minWidth);
    height = std::max(height, minHeight);

    const float widthChangeRatio = static_cast<float>(width) / m_width;
    const float heightChangeRatio = static_cast<float>(height) / m_height;

    nanogui::Vector2i oldPosition = m_nanoguiWindow->position();

	m_width = width;
	m_height = height;
	glViewport(0, 0, width, height);
	m_camera->setAspect((float)width / height);
	m_nanoguiScreen->resizeCallbackEvent(width, height);

    nanogui::Vector2i newPosition = { oldPosition[0] * widthChangeRatio, oldPosition[1] * heightChangeRatio };
    const int margin = 0;

    // std::cout << "Window width: " << m_nanoguiWindow->width() << " height: " << m_nanoguiWindow->height() << std::endl;
    // std::cout << "Screen width: " << m_nanoguiScreen->width() << " height: " << m_nanoguiScreen->height() << std::endl;
    // std::cout << std::endl;

    if (newPosition[0] + m_nanoguiWindow->width() > m_nanoguiScreen->width())
    {
        newPosition[0] = m_nanoguiScreen->width() - m_nanoguiWindow->width() - margin;
    }
    if (newPosition[1] + m_nanoguiWindow->height() > m_nanoguiScreen->height())
    {
        newPosition[1] = m_nanoguiScreen->height() - m_nanoguiWindow->height() - margin;
    }
    m_nanoguiWindow->setPosition(newPosition);

    m_smoothRenderer->HandleWindowResolutionChange(width, height);
}

void Renderer::__mouse_button_callback(GLFWwindow *w, int button, int action, int mods) {
	if (m_nanoguiScreen->mouseButtonCallbackEvent(button, action, mods)) return;

	Input::Pressed updown = action == GLFW_PRESS ? Input::DOWN : Input::UP;
	if (button == GLFW_MOUSE_BUTTON_LEFT)
		m_input->left_mouse = updown;
	if (button == GLFW_MOUSE_BUTTON_RIGHT)
		m_input->right_mouse = updown;
	if (button == GLFW_MOUSE_BUTTON_MIDDLE)
		m_input->mid_mouse = updown;
}

void Renderer::__mouse_move_callback(GLFWwindow* window, double xpos, double ypos) {
	if (m_nanoguiScreen->cursorPosCallbackEvent(xpos, ypos)) return;

	m_input->updateMousePos(glm::vec2(xpos, ypos));

	/* -- Camera control -- */

	/* Rotating */
	glm::vec2 scr_d = m_input->getMouseDiff();
	glm::vec3 pos = m_camera->getPos(), front = m_camera->getFront(), center = pos + front, up = m_camera->getUp();
	glm::vec3 cam_d = scr_d.x * -glm::normalize(glm::cross(front, up)) + scr_d.y * glm::normalize(up);

	if (m_input->left_mouse == Input::DOWN)
		m_camera->rotate(scr_d);

	/* Panning */
	if (m_input->right_mouse == Input::DOWN)
		m_camera->pan(scr_d);
	
}

void Renderer::__key_callback(GLFWwindow *w, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_V && action == GLFW_RELEASE)
    {
		m_nanoguiWindow->setVisible(!m_nanoguiWindow->visible());
        m_scrollWindow->setVisible(!m_scrollWindow->visible());
	}
    else if (key == GLFW_KEY_R && action == GLFW_RELEASE)
    {
        Input::getInstance().running = !Input::getInstance().running;
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

void Renderer::__mouse_scroll_callback(GLFWwindow *w, float dx, float dy) {
	if(m_nanoguiScreen->scrollCallbackEvent(dx, dy)) return;
	m_camera->zoom(dy);
}

void Renderer::__char_callback(GLFWwindow *w, unsigned int codepoint) {
	m_nanoguiScreen->charCallbackEvent(codepoint);
}

void Renderer::__binding() {

	glfwSetWindowUserPointer(m_glfwWindow.get(), this);

	glfwSetWindowSizeCallback(m_glfwWindow.get(), [](GLFWwindow *win, int width, int height) {
		((Renderer*)(glfwGetWindowUserPointer(win)))->__window_size_callback(win, width, height);
	});

	glfwSetCursorPosCallback(m_glfwWindow.get(), [](GLFWwindow *w, double xpos, double ypos) {
		((Renderer*)(glfwGetWindowUserPointer(w)))->__mouse_move_callback(w, xpos, ypos);
	});

	glfwSetMouseButtonCallback(m_glfwWindow.get(), [](GLFWwindow* w, int button, int action, int mods) {
		((Renderer*)(glfwGetWindowUserPointer(w)))->__mouse_button_callback(w, button, action, mods);
	});

	glfwSetScrollCallback(m_glfwWindow.get(), [](GLFWwindow *w, double dx, double dy) {
		((Renderer*)(glfwGetWindowUserPointer(w)))->__mouse_scroll_callback(w, dx, dy);
	});

	glfwSetKeyCallback(m_glfwWindow.get(),
		[](GLFWwindow *w, int key, int scancode, int action, int mods) {
		((Renderer*)(glfwGetWindowUserPointer(w)))->__key_callback(w, key, scancode, action, mods);
	});

	glfwSetCharCallback(m_glfwWindow.get(),
		[](GLFWwindow *w, unsigned int codepoint) {
		((Renderer*)(glfwGetWindowUserPointer(w)))->__char_callback(w, codepoint);
	});
}

void Renderer::__render() {
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
		m_particlesShader->setUnif("hlIndex", m_input->hlIndex);
		glBindVertexArray(d_vao);
		glDrawArrays(GL_POINTS, 0, m_nparticle);
	}
    else if (smoothFluid)
    {
        m_smoothRenderer->Render(d_vao, m_nparticle);
    }

	if (m_boundaryShader->loaded() && m_simulationParams->change)
    {
		m_boundaryShader->use();
		m_camera->use(Shader::now());
		glBindVertexArray(d_bbox_vao);
		m_boundaryShader->setUnif("color", glm::vec4(1.f, 1.f, 1.f, 1.f));
		glDrawArrays(GL_LINES, 0, 12 * 2);
	}
}

Renderer::~Renderer() {}

void Renderer::render(unsigned int pos, unsigned int iid, int nparticle)
{
	d_iid = iid;
	d_pos = pos;
	m_nparticle = nparticle;

    m_upperBoundary = m_simulationParams->GetUpperBoundary();
    m_lowerBoundary = m_simulationParams->GetLowerBoundary();

	glBindVertexArray(d_vao);
	glBindBuffer(GL_ARRAY_BUFFER, d_pos);
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
    //float x1 = fmin(m_upperBoundary.x, m_lowerBoundary.x);
    //float x2 = fmax(m_upperBoundary.x, m_lowerBoundary.x);
    //float y1 = fmin(m_upperBoundary.y, m_lowerBoundary.y);
    //float y2 = fmax(m_upperBoundary.y, m_lowerBoundary.y);
    //float z1 = fmin(m_upperBoundary.z, m_lowerBoundary.z);
    //float z2 = fmax(m_upperBoundary.z, m_lowerBoundary.z);

	glm::vec3 lines[][2] = {
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

	glBindBuffer(GL_ARRAY_BUFFER, d_bbox_vbo);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(lines), lines);

	m_formHelper->refresh();
    m_scrollFormHelper->refresh();

    if (!glfwWindowShouldClose(m_glfwWindow.get())) {
		glfwPollEvents();
		__render();
		m_nanoguiScreen->drawContents();
		m_nanoguiScreen->drawWidgets();
		glfwSwapBuffers(m_glfwWindow.get());
	}
	else fexit(0);
}

static unsigned int loadCubemap(char **faces) {
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

void Renderer::SetBoundaries(const float3 & upperBoundary, const float3 & lowerBoundary)
{
    m_upperBoundary = upperBoundary;
	m_lowerBoundary = lowerBoundary;
}
