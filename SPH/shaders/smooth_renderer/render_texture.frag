#version 330 core

out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D sourceTexture;

void main()
{
    vec3 color = texture(sourceTexture, TexCoords).rgb;

	//if (z > 50) discard;
    //
	//float color = exp(z)/(exp(z)+1);
	//color = (color - 0.5) * 2;
    FragColor = vec4(color, 1.0);
} 