#include <input.h>
#include <GLFW/glfw3.h>

Input::Input() 
{
	reset();
}

Input& Input::GetInstance()
{
    static Input inputSingletone;
	return inputSingletone;
}

Input* Input::GetInstancePtr()
{
	return &GetInstance();
}

glm::vec2 Input::UpdateMousePosition(glm::vec2 new_mouse)
{
    last_mouse = mouse;
    mouse = new_mouse;
    return getMouseDiff();
}

glm::vec2 Input::getMouseDiff()
{
	return mouse - last_mouse;
}

void Input::reset()
{
	last_mouse_valid = false;
	right_mouse = left_mouse = UP;
}