#version 330 core

uniform int filterRadius;
uniform float blurScale;
uniform float blurDepthFalloff;

// int filterRadius = 10;
// float blurScale = 0.2f;
// float blurDepthFalloff = 10.f;

uniform sampler2D sourceDepthTexture;

out vec4 FragColor;

const float Pi = 3.141592653;
const float TwoPi = 2.0f * Pi;
const float InversePi = 1.0f / Pi;


float InverseTwoSigmaS2 = 0.5 * blurScale * blurScale;
float InverseTwoSigmaR2 = 0.5 * blurDepthFalloff * blurDepthFalloff;
float MultiplierS = InversePi * InverseTwoSigmaS2;
float MultiplierR = InversePi * InverseTwoSigmaR2;

float BlurScale2 = 0.5 * blurScale * blurScale;
float BlurDepthFalloff2 = 0.5 * blurDepthFalloff * blurDepthFalloff;

float SGaussMultiplier = InversePi * BlurScale2;
float RGaussMultiplier = InversePi * BlurDepthFalloff2;

float GetDepth(int x, int y)
{
    return texelFetch(sourceDepthTexture, ivec2(x, y), 0).x;
    //return GetDepth(ivec2(x, y));
}

float BilateralFilterOther(int x, int y)
{
	float sum = 0;
    float weightSum = 0;
    float depth = GetDepth(x, y);

	for (int xOffset = -filterRadius; xOffset <= filterRadius; ++xOffset)
    {
		for (int yOffset = -filterRadius; yOffset <= filterRadius; ++yOffset)
        {            
            float offset2 = xOffset * xOffset + yOffset * yOffset;
            float weight = MultiplierS * exp(-(offset2) * InverseTwoSigmaS2);

            float sample = GetDepth(x + xOffset, y + yOffset);
			float difference = (sample - depth);
			float gauss = MultiplierR * exp(-difference * difference * InverseTwoSigmaR2);

			sum += sample * weight * gauss;
			weightSum += weight * gauss;
		}
    }
	if (weightSum > 0.0f) 
    {
        sum /= weightSum;
    }

	return sum;
}

// TODO: think if can make this better (more efficient).
// See: https://dsp.stackexchange.com/questions/36962/why-does-the-separable-filter-reduce-the-cost-of-computing-the-operator
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
            
            // Separable filter is product of two 1D filters.
            // See: http://www.cemyuksel.com/research/papers/narrowrangefilter.pdf
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
    float smoothedDepth = BilateralFilterOther(int(gl_FragCoord.x), int(gl_FragCoord.y));
    FragColor.x = smoothedDepth;
}