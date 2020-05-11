# version 330 core

out vec4 FragColor;

in vec2 TexCoords;

uniform float f_x;
uniform float f_y;
uniform float windowWidth;
uniform float windowHeight;

uniform sampler2D depthTexture;

float GetDepth(float x, float y)
{
	return -texture(depthTexture, vec2(x, y)).x;
}

void main()
{
	float x = TexCoords.x;
    float y = TexCoords.y;

	float depth = GetDepth(x, y);

    float dx = 1.0f / windowWidth;
	float dzdx = GetDepth(x + dx, y) - depth;
    float dzdx2 = depth - GetDepth(x - dx, y);
    if (abs(dzdx) > abs(dzdx2))
    {
        //dzdx = 0;
        dzdx = dzdx2;
    }

    float dy = 1.0f / windowHeight;
    float dzdy = GetDepth(x, y + dy) - depth;
    float dzdy2 = depth - GetDepth(x, y - dy);
    if (abs(dzdy) > abs(dzdy2))
    {
        //dzdy = 0;
        dzdy = dzdy2;
    }

    float c_x = 2.0f / (windowWidth * f_x);
	float c_y = 2.0f / (windowHeight * f_y);
	vec3 extractedNormal = vec3(-c_y * dzdx, -c_x * dzdy, c_x * c_y * depth);
	extractedNormal.z = -extractedNormal.z;

	float extractedNormalLength = length(extractedNormal);
	FragColor = vec4(extractedNormal / extractedNormalLength, extractedNormalLength);
}