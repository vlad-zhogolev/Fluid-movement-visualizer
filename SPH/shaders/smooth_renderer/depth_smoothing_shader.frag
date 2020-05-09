#version 330 core

// uniform int filterRadius;
// uniform float blurScale;
// uniform float blurDepthFalloff;

int filterRadius = 10;
float blurScale = 6.f;
float blurDepthFalloff = 0.1f;

uniform sampler2D sourceDepthTexture;

in vec2 TexCoords;

out vec4 FragColor;

float GetDepth(int x, int y)
{
    return texelFetch(sourceDepthTexture, ivec2(x, y), 0).x;
    //return GetDepth(ivec2(x, y));
}

//TODO: think if can make this better: https://dsp.stackexchange.com/questions/36962/why-does-the-separable-filter-reduce-the-cost-of-computing-the-operator
float BilateralFilter(int x, int y)
{
	float sum = 0;
    float wsum = 0;
    float depth = GetDepth(x, y);

	for (int xOffset = -filterRadius; xOffset <= filterRadius; ++xOffset)
    {
		for (int yOffset = -filterRadius; yOffset <= filterRadius; ++yOffset)
        {
			float sample = GetDepth(x + xOffset, y + yOffset);
            
            // Separable filter is product of two 1D filters: http://www.cemyuksel.com/research/papers/narrowrangefilter.pdf
			float w = exp(-(xOffset * xOffset + yOffset * yOffset) * blurScale * blurScale);

			float r2 = (sample - depth) * blurDepthFalloff;
			float g = exp(-r2 * r2);

			sum += sample * w * g;
			wsum += w * g;
		}
    }

	if (wsum > 0) 
    {
        sum /= wsum;
    }
	return sum;
}

void main()
{
    float smoothedDepth = BilateralFilter(int(gl_FragCoord.x), int(gl_FragCoord.y));
    FragColor.x = smoothedDepth;
}