# version 330 core

out vec4 FragColor;

in vec2 TexCoords;

uniform float Fx;
uniform float Fy;
uniform float windowWidth;
uniform float windowHeight;

uniform sampler2D depthTexture;

float dx = 1.0f / windowWidth;
float dy = 1.0f / windowHeight;
float Cx = 2.0f / (windowWidth * Fx);
float Cy = 2.0f / (windowHeight * Fy);

float GetDepth(float x, float y)
{
	return -texture(depthTexture, vec2(x, y)).x;
}

void main()
{
	float x = TexCoords.x;
    float y = TexCoords.y;

	float depth = GetDepth(x, y);

	float dzdx = GetDepth(x + dx, y) - depth;
    float dzdx2 = depth - GetDepth(x - dx, y);
    if (abs(dzdx) > abs(dzdx2))
    {
        //dzdx = 0;
        dzdx = dzdx2;
    }

    float dzdy = GetDepth(x, y + dy) - depth;
    float dzdy2 = depth - GetDepth(x, y - dy);
    if (abs(dzdy) > abs(dzdy2))
    {
        //dzdy = 0;
        dzdy = dzdy2;
    }

	vec3 extractedNormal = vec3(-Cy * dzdx, -Cx * dzdy, Cx * Cy * depth);
	extractedNormal.z = -extractedNormal.z;

	float extractedNormalLength = length(extractedNormal);
	FragColor = vec4(extractedNormal / extractedNormalLength, 1.0f);
}