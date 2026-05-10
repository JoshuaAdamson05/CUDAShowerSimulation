#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

class Particle
{
public:
	float3 position;
	float3 velocity;
	float mass;
	float temperature;

	unsigned int rngState;

	int active;

	__device__ float random01()
	{
		// Linear Congruential Generator (LCG) for per-particle random numbers
		rngState = (rngState * 1664525u + 1013904223u);

		// Return a float in the range [0, 1]
		return (rngState & 0xFFFFFF) / 16777216.0f;
	}

	__device__ void spawn(float width, float height, float depth)
	{
		const float emitterRadius = 0.05f;
		const float initialSpeed = 2.0f;

		float angle = random01() * 2.0f * 3.14159265f;
		float radius = sqrtf(random01()) * emitterRadius;

		position.x = width * 0.5f + radius * cosf(angle);
		position.y = height;
		position.z = depth * 0.5f + radius * sinf(angle);

		float coneAngle = (random01() * 60.0f - 30.0f) * 0.0174533f; // Convert to radians
		float azimuth = random01() * 2.0f * 3.14159265f;

		float horizontalSpeed = sinf(fabsf(coneAngle)) * initialSpeed;

		velocity.x = horizontalSpeed * cosf(azimuth);
		velocity.y = -cosf(coneAngle) * initialSpeed;
		velocity.z = horizontalSpeed * sinf(azimuth);

		mass = 1.0f;
		temperature = 1.0f;
		active = 1;
	}

	__device__ void mergeWith(Particle& other)
	{
		float combinedMass = mass + other.mass;

		position.x = ((position.x * mass) + (other.position.x * other.mass)) / combinedMass;
		position.y = ((position.y * mass) + (other.position.y * other.mass)) / combinedMass;
		position.z = ((position.z * mass) + (other.position.z * other.mass)) / combinedMass;

		velocity.x = ((velocity.x * mass) + (other.velocity.x * other.mass)) / combinedMass;
		velocity.y = ((velocity.y * mass) + (other.velocity.y * other.mass)) / combinedMass;
		velocity.z = ((velocity.z * mass) + (other.velocity.z * other.mass)) / combinedMass;

		temperature = ((temperature * mass) + (other.temperature * other.mass)) / combinedMass;

		mass = combinedMass;

		other.active = 0;
	}

	__device__ void applyPhysics(float deltaTime)
	{
		const float gravity = -9.81f;

		velocity.y += gravity * deltaTime;

		position.x += velocity.x * deltaTime;
		position.y += velocity.y * deltaTime;
		position.z += velocity.z * deltaTime;
	}

	__device__ void handleWalls(float width, float depth)
	{
		if (position.x < 0.0f) {
			position.x = 0.0f;
			velocity.x = -velocity.x;
		}
		if (position.x > width) {
			position.x = width;
			velocity.x = -velocity.x;
		}
		if (position.z < 0.0f) {
			position.z = 0.0f;
			velocity.z = -velocity.z;
		}
		if (position.z > depth) {
			position.z = depth;
			velocity.z = -velocity.z;
		}
	}

	__device__ void cool(float deltaTime)
	{
		const float coolingFactor = 1.0f;

		temperature -= (deltaTime * coolingFactor) / mass;

		if (temperature < 0.0f) 
		{
			temperature = 0.0f;
		}
	}

	__device__ bool hasHitFloor()
	{
		return position.y <= 0.0f;
	}
};