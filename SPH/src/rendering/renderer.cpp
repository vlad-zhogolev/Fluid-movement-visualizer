#include <rendering/renderer.h>
#include <input.h>

#include <cstdlib>

#include <GLFW\glfw3.h>
#include <nanogui\nanogui.h>
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
    SimulationParameters &params = SimulationParameters::getInstance();

	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    // NanoGUI initializtion
	m_nanoguiScreen =new nanogui::Screen();
	m_nanoguiScreen->initialize(m_glfwWindow.get(), true);
	m_nanoguiScreen->setSize(Eigen::Vector2i(800, 600));

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

    m_formHelper->addGroup("Simulation indicators");
    m_formHelper->setFixedSize({ 80, 20 });
    m_formHelper->addVariable("FPS", params.fps)->setEditable(false);
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
    nextFrameButton->setCallback([this]() { m_input->nextFrame = true; });
    
    auto testRunOrStopButton = new nanogui::Button(simulationControl, "", ENTYPO_ICON_CONTROLLER_PLAY);
    testRunOrStopButton->setFlags(nanogui::Button::ToggleButton);
    testRunOrStopButton->setFixedSize(nanogui::Vector2i(controlButtonSize, controlButtonSize));
    testRunOrStopButton->setChangeCallback([this, testRunOrStopButton](bool isPressed) {
        if (isPressed)
        {
            testRunOrStopButton->setIcon(ENTYPO_ICON_CONTROLLER_STOP);
            m_input->running = true;
        }
        else
        {
            testRunOrStopButton->setIcon(ENTYPO_ICON_CONTROLLER_PLAY);
            m_input->running = false;
        }
    });

    

    // m_scrollPanel = new nanogui::VScrollPanel(m_nanoguiWindow);
    // m_scrollPanel->setFixedSize(nanogui::Vector2i{ 200,200 });
    // m_formHelper->addWidget("", m_scrollPanel);

    // {
    // int width = 30, height = 30;
    // nanogui::Window *window = new nanogui::Window(m_nanoguiScreen, "All Icons");
    // window->setPosition({ 0, 0 });
    // window->setFixedSize({ 200, 200 });
    // window->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Vertical, nanogui::Alignment::Middle, 2, 8));
    // 
    // //nanogui::AdvancedGridLayout* mLayout = new nanogui::AdvancedGridLayout({ 10, 0, 10, 0 }, {});
    // //mLayout->setMargin(10);
    // //mLayout->setColStretch(2, 1);
    // //window->setLayout(mLayout);
    // 
    // 
    // // attach a vertical scroll panel
    // auto vscroll = new nanogui::VScrollPanel(window);
    // vscroll->setFixedSize({ 200, 200 });
    // 
    // // vscroll should only have *ONE* child. this is what `wrapper` is for
    // auto wrapper = new nanogui::Widget(vscroll);
    // wrapper->setFixedSize({ 200, 200 });
    // //wrapper->setLayout(new nanogui::GridLayout());// defaults: 2 columns
    // auto* wrapperLayout = new nanogui::AdvancedGridLayout({ 10, 0, 10, 0 });
    // wrapperLayout->setMargin(10);
    // wrapperLayout->setColStretch(2, 1);
    // wrapper->setLayout(wrapperLayout);
    // }

    // wrapperLayout->appendRow(0);
    // auto *label = new nanogui::Label(wrapper, "A label");
    // wrapperLayout->setAnchor(label, nanogui::AdvancedGridLayout::Anchor(0, 0, 1, 1, nanogui::Alignment::Middle, nanogui::Alignment::Middle));


    // auto testButton = new nanogui::Button(wrapper);     
    // testButton->setIconPosition(nanogui::Button::IconPosition::Left);
    // testButton->setFixedWidth(width);

    // m_widget = new nanogui::Widget(m_scrollPanel.get());
    // m_widget->setFixedSize({ 30, 30 });
    // m_widget->setLayout(new nanogui::GridLayout(nanogui::Orientation::Horizontal, 1));// defaults: 2 columns
    // 
    //nanogui::Button* testButton = nullptr;
    //for (int i = 0; i < 20; ++i)
    //{
    //    testButton = new nanogui::Button(wrapper);
    //    testButton->setIconPosition(nanogui::Button::IconPosition::Left);
    //    testButton->setFixedWidth(m_nanoguiWindow->size()[0] / 2);
    //}

    // m_formHelper->addGroup("Fluid parameters");
    // m_formHelper->addVariable("Change", params.change);
	// m_formHelper->addVariable("Substeps number", params.substepsNumber)->setSpinnable(true);
	// m_formHelper->addVariable("Rest density", params.restDensity)->setSpinnable(true);
	// m_formHelper->addVariable("Gravity acceleration", params.g)->setSpinnable(true);
	// m_formHelper->addVariable("Kernel radius", params.kernelRadius)->setSpinnable(true);
	// m_formHelper->addVariable("Delta time", params.deltaTime)->setSpinnable(true);
	// m_formHelper->addVariable("Lambda epsilon", params.relaxationParameter)->setSpinnable(true);
	// m_formHelper->addVariable("deltaQ", params.deltaQ)->setSpinnable(true);
	// m_formHelper->addVariable("correctionCoefficient", params.correctionCoefficient)->setSpinnable(true);
	// m_formHelper->addVariable("correctionPower", params.correctionPower)->setSpinnable(true);
	// m_formHelper->addVariable("XSPH coef", params.c_XSPH)->setSpinnable(true);
    // m_formHelper->addVariable("Viscosity iterations", params.viscosityIterations)->setSpinnable(true);
    // m_formHelper->addVariable("Vorticity epsilon", params.vorticityEpsilon)->setSpinnable(true);
    // m_formHelper->addVariable("Boundary movement speed", params.boundaryMovementSpeed)->setSpinnable(true);
	// m_formHelper->addVariable("Highlight #", m_input->hlIndex)->setSpinnable(true);


    m_scrollFormHelper = new nanogui::ScrollFormHelper(m_nanoguiScreen);
    auto scrollWindowInitialCoordinates = Eigen::Vector2i(0, 0);
    m_scrollWindow = m_scrollFormHelper->addWindow(
        scrollWindowInitialCoordinates, "Simulation and rendering parameters");
    // for (int i = 0; i < 15; ++i)
    // {
    //     m_scrollFormHelper->addVariable("Change", params.change);
    // }

    // m_scrollFormHelper->addGroup("Simulation indicators");
    // m_scrollFormHelper->addVariable("FPS", params.fps)->setEditable(false);
    // m_scrollFormHelper->addVariable("Current frame number", m_input->frameCount)->setEditable(false);

    // m_scrollFormHelper->addGroup("Simulation controls");
    // auto simulationControl = new nanogui::Widget(m_scrollFormHelper->wrapper());
    // m_scrollFormHelper->addWidget("", simulationControl);
    // simulationControl->setLayout(
    //     new nanogui::BoxLayout(nanogui::Orientation::Horizontal, nanogui::Alignment::Middle, 2, 8));
    // 
    // const int controlButtonSize = 30;
    // auto nextFrameButtonScroll = new nanogui::Button(simulationControl, "", ENTYPO_ICON_CONTROLLER_NEXT);
    // nextFrameButtonScroll->setFixedSize(nanogui::Vector2i(controlButtonSize, controlButtonSize));
    // nextFrameButtonScroll->setFlags(nanogui::Button::NormalButton);
    // nextFrameButtonScroll->setCallback([this]() { m_input->nextFrame = true; });
    // 
    // auto testRunOrStopButtonScroll = new nanogui::Button(simulationControl, "", ENTYPO_ICON_CONTROLLER_PLAY);
    // testRunOrStopButtonScroll->setFlags(nanogui::Button::ToggleButton);
    // testRunOrStopButtonScroll->setFixedSize(nanogui::Vector2i(controlButtonSize, controlButtonSize));
    // testRunOrStopButtonScroll->setChangeCallback([this, testRunOrStopButtonScroll](bool isPressed) {
    //     if (isPressed)
    //     {
    //         testRunOrStopButtonScroll->setIcon(ENTYPO_ICON_CONTROLLER_STOP);
    //         m_input->running = true;
    //     }
    //     else
    //     {
    //         testRunOrStopButtonScroll->setIcon(ENTYPO_ICON_CONTROLLER_PLAY);
    //         m_input->running = false;
    //     }
    // });

    m_scrollFormHelper->addGroup("Fluid parameters");

    m_scrollFormHelper->addVariable("Change", params.change);
    m_scrollFormHelper->addVariable("Substeps number", params.substepsNumber)->setSpinnable(true);
    m_scrollFormHelper->addVariable("Rest density", params.restDensity)->setSpinnable(true);
    m_scrollFormHelper->addVariable("Gravity acceleration", params.g)->setSpinnable(true);
    m_scrollFormHelper->addVariable("Kernel radius", params.kernelRadius)->setSpinnable(true);
    m_scrollFormHelper->addVariable("Delta time", params.deltaTime)->setSpinnable(true);
    m_scrollFormHelper->addVariable("Lambda epsilon", params.relaxationParameter)->setSpinnable(true);
    m_scrollFormHelper->addVariable("deltaQ", params.deltaQ)->setSpinnable(true);
    m_scrollFormHelper->addVariable("correctionCoefficient", params.correctionCoefficient)->setSpinnable(true);
    m_scrollFormHelper->addVariable("correctionPower", params.correctionPower)->setSpinnable(true);
    m_scrollFormHelper->addVariable("XSPH coef", params.c_XSPH)->setSpinnable(true);
    m_scrollFormHelper->addVariable("Viscosity iterations", params.viscosityIterations)->setSpinnable(true);
    m_scrollFormHelper->addVariable("Vorticity epsilon", params.vorticityEpsilon)->setSpinnable(true);

    m_nanoguiScreen->performLayout();
	m_nanoguiScreen->setVisible(true);
	
    m_scrollWindow->setPosition({ 5, 2 * initialCoordinate + m_nanoguiWindow->height() });

	__binding();

    float aspect = (float) width_ / height_;
	m_camera = new Camera(cam_pos, cam_focus, aspect);
	m_box_shader = new Shader(Path("shaders/boundary.vert"), Path("shaders/boundary.frag"));
	m_particle_shader = new Shader(Path("shaders/particle.vert"), Path("shaders/particle.frag"));
    m_sky_shader = new Shader(Path("shaders/skybox.vert"), Path("shaders/skybox.frag"));

	char *sky_faces[] = { 
		"skybox/right.jpg",		
		"skybox/left.jpg",		
		"skybox/front.jpg",		
		"skybox/back.jpg",		
		"skybox/top.jpg",		
		"skybox/bottom.jpg"		
	};
	d_sky_texture = loadCubemap(sky_faces);
	
	glGenVertexArrays(1, &d_vao);

    glGenVertexArrays(1, &d_bbox_vao);
	glGenBuffers(1, &d_bbox_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, d_bbox_vbo);
	glBufferData(GL_ARRAY_BUFFER, 12 * 2 * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glBindVertexArray(d_bbox_vao);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	
    glGenVertexArrays(1, &d_sky_vao);
	glGenBuffers(1, &d_sky_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, d_sky_vbo);
	glBufferData(GL_ARRAY_BUFFER, 6 * 2 * 3 * 3 * sizeof(float), SKYBOX_VERTICES, GL_STATIC_DRAW);
	glBindVertexArray(d_sky_vao);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

    m_smoothRenderer = std::make_unique<rendering::SmoothRenderer>(m_width, m_height, m_camera, d_sky_texture);
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

    std::cout << "Window width: " << m_nanoguiWindow->width() << " height: " << m_nanoguiWindow->height() << std::endl;
    std::cout << "Screen width: " << m_nanoguiScreen->width() << " height: " << m_nanoguiScreen->height() << std::endl;
    std::cout << std::endl;

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

	if (m_sky_shader->loaded())
    {
		glDepthMask(GL_FALSE);
		m_sky_shader->use();
		m_camera->use(Shader::now(), true);
		glBindVertexArray(d_sky_vao);
		glBindTexture(GL_TEXTURE_CUBE_MAP, d_sky_texture);
		glDrawArrays(GL_TRIANGLES, 0, 36);
		glDepthMask(GL_TRUE);
	}
    
    const bool smoothFluid = true;
	if (m_particle_shader->loaded() && !smoothFluid)
    {
		m_particle_shader->use();
		m_camera->use(Shader::now());
		m_particle_shader->setUnif("color", glm::vec4(1.f, 0.f, 0.f, .1f));
		m_particle_shader->setUnif("pointRadius", SimulationParameters::getInstance().kernelRadius);
		m_particle_shader->setUnif("pointScale", 500.f);
		m_particle_shader->setUnif("hlIndex", m_input->hlIndex);
		glBindVertexArray(d_vao);
		glDrawArrays(GL_POINTS, 0, m_nparticle);
	}
    else if (smoothFluid)
    {
        m_smoothRenderer->Render(d_vao, m_nparticle);
    }

	if (m_box_shader->loaded())
    {
		m_box_shader->use();
		m_camera->use(Shader::now());
		glBindVertexArray(d_bbox_vao);
		m_box_shader->setUnif("color", glm::vec4(1.f, 1.f, 1.f, 1.f));
		glDrawArrays(GL_LINES, 0, 12 * 2);
	}
}

Renderer::~Renderer()
{
	if (m_camera) delete m_camera;
	if (m_box_shader) delete m_box_shader;
	if (m_particle_shader) delete m_particle_shader;
	/* TODO: m_window, input */
	if (m_input) delete m_input;
	// _pclose(m_ffmpeg);
}

void Renderer::render(unsigned int pos, unsigned int iid, int nparticle)
{
	d_iid = iid;
	d_pos = pos;
	m_nparticle = nparticle;

	glBindVertexArray(d_vao);
	glBindBuffer(GL_ARRAY_BUFFER, d_pos);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, d_iid);
	glVertexAttribIPointer(1, 1, GL_UNSIGNED_INT, 0, (void*)0);
	glEnableVertexAttribArray(1);

	float x1 = fmin(m_ulim.x, m_llim.x), x2 = fmax(m_ulim.x, m_llim.x),
		y1 = fmin(m_ulim.y, m_llim.y), y2 = fmax(m_ulim.y, m_llim.y),
		z1 = fmin(m_ulim.z, m_llim.z), z2 = fmax(m_ulim.z, m_llim.z);

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
		{ glm::vec3(x2, y2, z1), glm::vec3(x2, y2, z2) } };

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

void Renderer::setLim(const float3 & ulim, const float3 & llim)
{
	m_llim = llim;
	m_ulim = ulim;
}
