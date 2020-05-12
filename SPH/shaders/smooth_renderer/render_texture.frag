#version 330 core

out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D sourceTexture;

void render_normals()
{
    vec3 color = texture(sourceTexture, TexCoords).rgb;
    FragColor = vec4(color, 1.0);
}

void render_linear_depth()
{
    float depth = texture(sourceTexture, TexCoords).x;
    
    if (depth > 50) 
    {
        discard;
    }

	float color = exp(depth)/(exp(depth)+1);
	color = (color - 0.5) * 2;
    
    FragColor = vec4(color, color, color, 1.0f);
}

void render_thickness()
{
    float thickness = texture(sourceTexture, TexCoords).x;

	float color = exp(thickness)/(exp(thickness)+1);
	color = (color - 0.5) * 2;
    
    FragColor = vec4(color, color, color, 1.0f);
    //FragColor = vec4(thickness, thickness, thickness, 1.0f);
}

void main()
{
    render_thickness();
} 