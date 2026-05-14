#pragma once	

#include <vector>
#include <algorithm>

#include "Particle.cuh"

struct RenderVertex
{
	float x, y, z;
	float r, g, b;

	RenderVertex() = default;

	RenderVertex(float x, float y, float z, float r, float g, float b)
		: x(x), y(y), z(z), r(r), g(g), b(b) {}

};

enum class RenderMode
{
	Thermal,
	Mass
};

inline void worldTo3D(float x, float y, float z, float width, float height, float depth, float& outX, float& outY, float& outZ)
{
	outX = x - width / 2.0f;
	outY = y - height / 2.0f;
	outZ = z - depth / 2.0f;
}

inline void temperatureColour(float temperature, float& r, float& g, float& b)
{
	float t = temperature;

	if (temperature < 0.0f)
		t = 0.0f;
	if (temperature > 1.0f)
		t = 1.0f;
	
	r = t;
	g = 0.0f;
	b = 1.0f - t;
}

inline void massColour(float mass, float& r, float& g, float& b)
{
	if (mass < 1.1f)
	{
		r = 0.0f; g = 0.5f; b = 1.0f; // blue = original / low mass
	}
	else if (mass < 2.1f)
	{
		r = 0.0f; g = 1.0f; b = 0.0f; // green = mass 2-ish
	}
	else if (mass < 3.1f)
	{
		r = 1.0f; g = 1.0f; b = 0.0f; // yellow = mass 3-ish
	}
	else
	{
		r = 1.0f; g = 0.0f; b = 0.0f; // red = mass 4+
	}
}

inline std::vector<RenderVertex> buildRenderVertices(
	const std::vector<Particle>& particles,
	RenderMode mode,
	int renderStride,
	float width,
	float height,
	float depth)
{
	const int stride = std::max(1, renderStride);

	std::vector<RenderVertex> vertices;

	auto addVertex = [&](const Particle& particle)
		{
			RenderVertex vertex;

			worldTo3D(
				particle.position.x,
				particle.position.y,
				particle.position.z,
				width,
				height,
				depth,
				vertex.x,
				vertex.y,
				vertex.z
			);

			if (mode == RenderMode::Thermal)
			{
				temperatureColour(particle.temperature, vertex.r, vertex.g, vertex.b);
			}
			else
			{
				massColour(particle.mass, vertex.r, vertex.g, vertex.b);
			}

			vertices.push_back(vertex);
		};

	// Normal thermal rendering: simple active-particle stride
	if (mode == RenderMode::Thermal)
	{
		int activeCounter = 0;

		for (const Particle& particle : particles)
		{
			if (particle.active != 1)
			{
				continue;
			}

			if (activeCounter % stride == 0)
			{
				addVertex(particle);
			}

			activeCounter++;
		}

		return vertices;
	}

	// Mass mode: prioritise merged/high-mass particles so tiers are visible
	std::vector<size_t> activeIndices;

	for (size_t i = 0; i < particles.size(); i++)
	{
		if (particles[i].active == 1)
		{
			activeIndices.push_back(i);
		}
	}

	const size_t targetCount = std::max<size_t>(1, activeIndices.size() / stride);
	vertices.reserve(targetCount);

	// Sort by mass descending so the largest merged particles are selected first
	std::sort(activeIndices.begin(), activeIndices.end(),
		[&](size_t a, size_t b)
		{
			return particles[a].mass > particles[b].mass;
		}
	);

	// Select high-mass particles first
	std::vector<size_t> selectedIndices;
	selectedIndices.reserve(targetCount);

	for (size_t index : activeIndices)
	{
		if (selectedIndices.size() >= targetCount / 2)
		{
			break;
		}

		if (particles[index].mass > 1.05f)
		{
			selectedIndices.push_back(index);
		}
	}

	// Fill the rest with normal strided active particles
	int activeCounter = 0;

	for (size_t i = 0; i < particles.size(); i++)
	{
		if (particles[i].active != 1)
		{
			continue;
		}

		if (activeCounter % stride == 0)
		{
			selectedIndices.push_back(i);

			if (selectedIndices.size() >= targetCount)
			{
				break;
			}
		}

		activeCounter++;
	}

	// Draw low mass first and high mass last, so coloured merged particles are not hidden
	std::sort(selectedIndices.begin(), selectedIndices.end(),
		[&](size_t a, size_t b)
		{
			return particles[a].mass < particles[b].mass;
		}
	);

	for (size_t index : selectedIndices)
	{
		addVertex(particles[index]);
	}

	return vertices;
}

inline std::vector<RenderVertex> buildCubicleVertices(float width, float height, float depth)
{
	std::vector<RenderVertex> vertices;

	float r = 1.0f;
	float g = 1.0f;
	float b = 1.0f;

	float p000x, p000y, p000z;
	float p100x, p100y, p100z;
	float p010x, p010y, p010z;
	float p110x, p110y, p110z;

	float p001x, p001y, p001z;
	float p101x, p101y, p101z;
	float p011x, p011y, p011z;
	float p111x, p111y, p111z;

	worldTo3D(0.0f, 0.0f, 0.0f, width, height, depth, p000x, p000y, p000z);
	worldTo3D(width, 0.0f, 0.0f, width, height, depth, p100x, p100y, p100z);
	worldTo3D(0.0f, height, 0.0f, width, height, depth, p010x, p010y, p010z);
	worldTo3D(width, height, 0.0f, width, height, depth, p110x, p110y, p110z);

	worldTo3D(0.0f, 0.0f, depth, width, height, depth, p001x, p001y, p001z);
	worldTo3D(width, 0.0f, depth, width, height, depth, p101x, p101y, p101z);
	worldTo3D(0.0f, height, depth, width, height, depth, p011x, p011y, p011z);
	worldTo3D(width, height, depth, width, height, depth, p111x, p111y, p111z);

	auto addLine = [&](float ax, float ay, float az, float bx, float by, float bz)
		{
			vertices.emplace_back(ax, ay, az, r, g, b);
			vertices.emplace_back(bx, by, bz, r, g, b);
		};

	// Back rectangle
	addLine(p000x, p000y, p000z, p100x, p100y, p100z);
	addLine(p100x, p100y, p100z, p110x, p110y, p110z);
	addLine(p110x, p110y, p110z, p010x, p010y, p010z);
	addLine(p010x, p010y, p010z, p000x, p000y, p000z);

	// Front rectangle
	addLine(p001x, p001y, p001z, p101x, p101y, p101z);
	addLine(p101x, p101y, p101z, p111x, p111y, p111z);
	addLine(p111x, p111y, p111z, p011x, p011y, p011z);
	addLine(p011x, p011y, p011z, p001x, p001y, p001z);

	// Connecting edges
	addLine(p000x, p000y, p000z, p001x, p001y, p001z);
	addLine(p100x, p100y, p100z, p101x, p101y, p101z);
	addLine(p110x, p110y, p110z, p111x, p111y, p111z);
	addLine(p010x, p010y, p010z, p011x, p011y, p011z);

	return vertices;
}

inline std::vector<RenderVertex> buildShowerHeadVertices(
	float width,
	float height,
	float depth,
	float emitterRadius
)
{
	std::vector<RenderVertex> vertices;

	constexpr int segments = 256;

	float r = 1.0f;
	float g = 1.0f;
	float b = 0.0f;

	float centreX, centreY, centreZ;

	worldTo3D(
		width / 2.0f,
		height,
		depth / 2.0f,
		width,
		height,
		depth,
		centreX,
		centreY,
		centreZ
	);

	vertices.emplace_back(centreX, centreY, centreZ, r, g, b);

	for (int i = 0; i <= segments; i++)
	{
		float theta = (static_cast<float>(i) / static_cast<float>(segments)) * 2.0f * 3.14159265f;

		float xWorld = width / 2.0f + emitterRadius * cosf(theta);
		float yWorld = height;
		float zWorld = depth / 2.0f + emitterRadius * sinf(theta);

		float x, y, z;

		worldTo3D(
			xWorld,
			yWorld,
			zWorld,
			width,
			height,
			depth,
			x,
			y,
			z
		);

		vertices.emplace_back(x, y, z, r, g, b);
	}

	return vertices;
}