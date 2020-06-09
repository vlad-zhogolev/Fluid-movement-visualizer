#pragma once

#include <rendering/camera.h>
#include <rendering/shader.h>
#include <rendering/smooth_renderer.h>
#include <rendering/scroll_form_helper.h>
#include <rendering/rendering_parameters.h>
#include <input.h>
#include <utils.h>
#include <simulation/simulation_parameters.h>
#include <GLFW/glfw3.h>
#include <nanogui/nanogui.h>

#include <functional>
#include <memory>

class Renderer
{
public:

    Renderer(std::unique_ptr<GLFWwindow> glfwWindow);
	~Renderer();

	void Render(unsigned int pos, unsigned int iid, int m_nparticle);
	void SetBoundaries(const float3 &ulim, const float3 &llim);

private:

    void Init();
    
    void SetStartSettingsEnabled(bool isEnabled);

	void BindGLFWCallbacks();
	void RenderImpl();

	void WindowSizeCallback(GLFWwindow* window, int width, int height);
	void MouseMoveCallback(GLFWwindow* window, double xpos, double ypos);
	void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
	void MouseScrollCallback(GLFWwindow* window, float dx, float dy);
	void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	void CharCallback(GLFWwindow* window, unsigned int codepoint);

    int m_width;
    int m_height;
	int m_nparticle;
	
    unsigned int m_particlesVAO;
    unsigned int m_boundariesVAO;
    unsigned int m_boundariesVBO;
    unsigned int m_particlesPositions;
    unsigned int d_iid;
    float3 m_lowerBoundary;
    float3 m_upperBoundary;

	std::shared_ptr<Camera> m_camera = nullptr;

	std::unique_ptr<Shader> m_boundaryShader = nullptr;
	std::unique_ptr<Shader> m_particlesShader = nullptr;
    std::unique_ptr<Shader> m_skyboxShader = nullptr;

    std::unique_ptr<GLFWwindow> m_glfwWindow;

	// NanoGUI
    // No need to manage these pointers, because nanogui does this.
	nanogui::Screen* m_nanoguiScreen = nullptr;
	nanogui::FormHelper* m_controlsFormHelper = nullptr;
    nanogui::ref<nanogui::Window> m_controlsWindow;
    nanogui::ScrollFormHelper* m_scrollFormHelper = nullptr;
    nanogui::ref<nanogui::Window> m_scrollWindow;

    std::vector<nanogui::ref<nanogui::Widget>> m_switchOffRestart;
    std::vector<nanogui::ref<nanogui::TextBox>> m_positionVariables;

	int frameCount = 0;
    
	// Skybox
	unsigned int m_skyboxTexture;
    unsigned int m_skyboxVAO;
    unsigned int m_skyboxVBO;

    std::unique_ptr<rendering::SmoothRenderer> m_smoothRenderer = nullptr;
    Input* m_input = Input::GetInstancePtr();
    RenderingParameters* m_renderingParams = RenderingParameters::GetInstancePtr();
    SimulationParameters* m_simulationParams = SimulationParameters::GetInstancePtr();
};

