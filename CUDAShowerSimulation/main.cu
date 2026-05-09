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

__global__ void updateParticlesKernel(Particle* particles, int numParticles, unsigned int* floorHits, float* totalTemperature, float deltaTime, float width, float height, float depth)
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

	atomicAdd(totalTemperature, particle.temperature);
}

int main()
{
	Particle* d_particles = nullptr;
	unsigned int* d_floorHits = nullptr;
	float* d_totalTemperature = nullptr;

    checkCudaError(cudaMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle)));
	checkCudaError(cudaMalloc(&d_floorHits, sizeof(unsigned int)));
	checkCudaError(cudaMalloc(&d_totalTemperature, sizeof(float)));

	checkCudaError(cudaMemset(d_floorHits, 0, sizeof(unsigned int)));

	int blocks = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

	cudaEvent_t startEvent;
	cudaEvent_t endEvent;

	checkCudaError(cudaEventCreate(&startEvent));
	checkCudaError(cudaEventCreate(&endEvent));

	initialiseParticlesKernel << <blocks, BLOCK_SIZE >> > (d_particles, NUM_PARTICLES, WIDTH, HEIGHT, DEPTH);
	checkCudaError(cudaDeviceSynchronize());

	for (int frame = 0; frame < 1000; frame++)
	{
		checkCudaError(cudaMemset(d_totalTemperature, 0, sizeof(float)));

		checkCudaError(cudaEventRecord(startEvent));

		updateParticlesKernel << <blocks, BLOCK_SIZE >> > (d_particles, NUM_PARTICLES, d_floorHits, d_totalTemperature, DELTA_TIME, WIDTH, HEIGHT, DEPTH);
		checkCudaError(cudaGetLastError());
		checkCudaError(cudaDeviceSynchronize());

		checkCudaError(cudaEventRecord(endEvent));
		checkCudaError(cudaEventSynchronize(endEvent));
		float milliseconds = 0.0f;
		checkCudaError(cudaEventElapsedTime(&milliseconds, startEvent, endEvent));

		if (frame % 60 == 0)
		{
			unsigned int floorHits = 0;
			float totalTemperature = 0.0f;
			checkCudaError(cudaMemcpy(&floorHits, d_floorHits, sizeof(unsigned int), cudaMemcpyDeviceToHost));
			checkCudaError(cudaMemcpy(&totalTemperature, d_totalTemperature, sizeof(float), cudaMemcpyDeviceToHost));

			float averageTemperature = totalTemperature / NUM_PARTICLES;
			std::cout << "Frame: " << frame << " | Particles: " << NUM_PARTICLES << " | Floor Hits: " << floorHits << " | Avg Temperature: " << averageTemperature << " | GPU Update: " << milliseconds << " ms" << std::endl;
		}
	}

	checkCudaError(cudaFree(d_particles));
	checkCudaError(cudaFree(d_floorHits));
	checkCudaError(cudaFree(d_totalTemperature));
	checkCudaError(cudaEventDestroy(startEvent));
	checkCudaError(cudaEventDestroy(endEvent));

	return 0;
}