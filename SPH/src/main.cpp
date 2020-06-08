#include <GLFW/glfw3.h>

#include <rendering/renderer.h>
#include <rendering/rendering_parameters.h>
#include <simulation/particle_system.h>
#include <memory>

namespace {

constexpr int WindowWidth = 800;
constexpr int WindowHeight = 600;
constexpr char* WindowTitle = "Fluid movement visualizer";

unsigned int g_frameCount = 0;
double g_previousTime = 0;

std::unique_ptr<GLFWwindow> InitializeOpenGL();
void TerminateApplication();

std::unique_ptr<GLFWwindow> InitializeOpenGL()
{
    glfwInit();
    glfwSetTime(0);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwWindowHint(GLFW_SAMPLES, 0);
    glfwWindowHint(GLFW_RED_BITS, 8);
    glfwWindowHint(GLFW_GREEN_BITS, 8);
    glfwWindowHint(GLFW_BLUE_BITS, 8);
    glfwWindowHint(GLFW_ALPHA_BITS, 8);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

    auto windowOrdinaryPtr = glfwCreateWindow(WindowWidth, WindowHeight, WindowTitle, nullptr, nullptr);
    std::unique_ptr<GLFWwindow> windowPtr(windowOrdinaryPtr);
    if (!windowPtr)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        TerminateApplication();
    }
    glfwMakeContextCurrent(windowPtr.get());

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        auto errorCode = glGetError();
        std::cout << "Failed to create GLFW window, error code: " << errorCode << std::endl;
        TerminateApplication();
    }

    return windowPtr;
}

void TerminateApplication()
{
    std::cout << "Exit application" << std::endl;
    glfwTerminate();
    exit(-1);
}

} // namespace

int main()
{
    auto windowPtr = InitializeOpenGL();

	ParticleSystem particleystem;
    Renderer renderer(std::move(windowPtr));

	particleystem.InitializeParticles();
    RenderingParameters &renderingParameters = RenderingParameters::GetInstance();
    renderingParameters.fps = 0;
    g_previousTime = glfwGetTime();
	while (true)
    {
        ++g_frameCount;
        double currentTime = glfwGetTime();

		renderer.SetBoundaries(particleystem.GetUpperLimit(), particleystem.GetLowerLimit());
        renderer.Render(
            particleystem.GetPositionsForRenderingHandle(),
            particleystem.GetIndicesHandle(),
            particleystem.GetParticleNumber());

	    particleystem.PerformSimulationStep();

        if (currentTime - g_previousTime >= 1.0)
        {
            renderingParameters.fps = g_frameCount;
            g_frameCount = 0;
            g_previousTime = currentTime;
        }
	}
}