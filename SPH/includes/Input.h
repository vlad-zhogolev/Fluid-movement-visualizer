#pragma once

#include <simulation/simulation_parameters.h>>
#include <glm/glm.hpp>

struct Input
{
    enum Pressed { UP, DOWN };

    Input();
    static Input& GetInstance();
    static Input* GetInstancePtr();

    Pressed left_mouse, right_mouse, mid_mouse;
    glm::vec2 last_mouse, mouse;
    bool last_mouse_valid;

    glm::vec2 UpdateMousePosition(glm::vec2 mouse);
    glm::vec2 getMouseDiff();
    void reset();

    int frameCount;
};