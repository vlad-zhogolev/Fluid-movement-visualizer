#pragma once

#include <rendering/camera.h>
#include <rendering/shader.h>
#include <rendering/smooth_renderer.h>
#include <rendering/scroll_form_helper.h>
#include <input.h>
#include <helper.h>
#include <simulation/simulation_parameters.h>>
#include <GLFW/glfw3.h>
#include <nanogui/nanogui.h>


#include <functional>
#include <memory>

class Renderer
{
public:

    Renderer(std::unique_ptr<GLFWwindow> glfwWindow)
        : m_glfwWindow(std::move(glfwWindow))
        , m_input(&Input::getInstance())
    {
        glfwSetWindowSizeLimits(m_glfwWindow.get(), 800, 600, GLFW_DONT_CARE, GLFW_DONT_CARE);
        Init();
    }

	Renderer(
		const glm::vec3 &cam_pos, 
		const glm::vec3 &cam_focus, 
		float3 ulim, float3 llim, 
		std::function<void()> nextCb) 
		: m_ulim(ulim), m_llim(llim), m_nextFrameBtnCb(nextCb) { init(cam_pos, cam_focus); };
	~Renderer();

	void render(unsigned int pos, unsigned int iid, int m_nparticle);
	void setLim(const float3 &ulim, const float3 &llim);

	Input *m_input;

private:

    void Init();
	void init(const glm::vec3 &cam_pos, const glm::vec3 &cam_focus);
	void __binding();
	void __render();

	void __window_size_callback(GLFWwindow* window, int width, int height);
	void __mouse_move_callback(GLFWwindow* window, double xpos, double ypos);
	void __mouse_button_callback(GLFWwindow* w, int button, int action, int mods);
	void __mouse_scroll_callback(GLFWwindow* w, float dx, float dy);
	void __key_callback(GLFWwindow *w, int key, int scancode, int action, int mods);
	void __char_callback(GLFWwindow *w, unsigned int codepoint);

    int m_width;
    int m_height;
	int m_nparticle;
	
    unsigned int d_vao;
    unsigned int d_bbox_vao;
    unsigned int d_bbox_vbo;
    unsigned int d_pos;
    unsigned int d_iid;
    float3 m_llim;
    float3 m_ulim;

	Camera *m_camera = nullptr;

	Shader *m_box_shader = nullptr;
	Shader *m_particle_shader = nullptr;

    std::unique_ptr<GLFWwindow> m_glfwWindow;

	// NanoGUI
    // No need to manage these pointers, because nanogui does this.
	nanogui::Screen* m_nanoguiScreen = nullptr;
	nanogui::FormHelper* m_formHelper = nullptr;
    nanogui::ScrollFormHelper* m_scrollFormHelper = nullptr;
    nanogui::ref<nanogui::Window> m_scrollWindow;
	nanogui::Window* m_nanoguiWindow = nullptr;
    nanogui::Widget* m_widget = nullptr;
	std::function<void()> m_nextFrameBtnCb;

	int frameCount = 0;
    

	// Skybox
	unsigned int d_sky_texture;
	Shader* m_sky_shader = nullptr;
    unsigned int d_sky_vao;
    unsigned int d_sky_vbo;

    std::unique_ptr<rendering::SmoothRenderer> m_smoothRenderer = nullptr;
    SimulationParameters* m_simulationParams = SimulationParameters::GetInstancePtr();
};

