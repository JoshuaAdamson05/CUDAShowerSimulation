#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "Particle.cuh"

constexpr int NUM_PARTICLES = 100000;
constexpr int BLOCK_SIZE = 256;
constexpr float WIDTH = 1.0f;
constexpr float HEIGHT = 2.0f;
constexpr float DEPTH = 1.0f;
constexpr float DELTA_TIME = 1.0f / 60.0f;

#define checkCudaError(call) \
do \
{ \
	cudaError_t err = call; \
	if (err != cudaSuccess) \
	{ \
		std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
		exit(EXIT_FAILURE); \
	} \
} while (0)

__global__ void initialiseParticlesKernel(Particle* particles, int numParticles, float width, float height, float depth)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numParticles)
		return;

	particles[i].rngState = 1234u + i * 9781u;
	particles[i].spawn(width, height, depth);
}

__global__ void updateParticlesKernel(Particle* particles, int numParticles, unsigned int* floorHits, float deltaTime, float width, float height, float depth)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numParticles)
		return;

	Particle& particle = particles[i];

	particle.applyPhysics(deltaTime);
	particle.handleWalls(width, depth);
	particle.cool(deltaTime);

	if (particle.hasHitFloor())
	{
		atomicAdd(floorHits, 1u);
		particle.spawn(width, height, depth);
	}
}

int main()
{
	Particle* d_particles = nullptr;
	unsigned int* d_floorHits = nullptr;

    checkCudaError(cudaMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle)));
	checkCudaError(cudaMalloc(&d_floorHits, sizeof(unsigned int)));

	checkCudaError(cudaMemset(d_floorHits, 0, sizeof(unsigned int)));

	int blocks = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

	initialiseParticlesKernel << <blocks, BLOCK_SIZE >> > (d_particles, NUM_PARTICLES, WIDTH, HEIGHT, DEPTH);
	cudaDeviceSynchronize();

	for (int frame = 0; frame < 1000; frame++)
	{
		updateParticlesKernel << <blocks, BLOCK_SIZE >> > (d_particles, NUM_PARTICLES, d_floorHits, DELTA_TIME, WIDTH, HEIGHT, DEPTH);
		cudaDeviceSynchronize();

		if (frame % 60 == 0)
		{
			unsigned int floorHits = 0;
			checkCudaError(cudaMemcpy(&floorHits, d_floorHits, sizeof(unsigned int), cudaMemcpyDeviceToHost));
			std::cout << "Frame: " << frame << " | Particles: " << NUM_PARTICLES << " | Floor Hits: " << floorHits << std::endl;
		}
	}

	checkCudaError(cudaFree(d_particles));
	checkCudaError(cudaFree(d_floorHits));

	return 0;
}