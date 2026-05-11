#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>

#include "Particle.cuh"
#include "Rendering.h"
#include "OpenGLRenderer.h"

constexpr int NUM_PARTICLES = 100000;
constexpr int BLOCK_SIZE = 256;
constexpr float WIDTH = 1.0f;
constexpr float HEIGHT = 2.0f;
constexpr float DEPTH = 1.0f;
constexpr float DELTA_TIME = 1.0f / 60.0f;
constexpr float COLLISION_RADIUS = 0.0001f;
constexpr float CELL_SIZE = 0.02f;
constexpr float EMITTER_RADIUS = 0.05f;

constexpr int GRID_X = static_cast<int>(WIDTH / CELL_SIZE) + 1;
constexpr int GRID_Y = static_cast<int>(HEIGHT / CELL_SIZE) + 1;
constexpr int GRID_Z = static_cast<int>(DEPTH / CELL_SIZE) + 1;

constexpr int NUM_CELLS = GRID_X * GRID_Y * GRID_Z;

static_assert(NUM_CELLS > 0, "NUM_CELLS must be positive");

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

	// Initialize RNG state with a unique seed for each particle
	particles[i].rngState = 1234u + i * 9781u;

	particles[i].spawn(width, height, depth);
}

__global__ void updateParticlesKernel(Particle* particles, int numParticles, unsigned int* floorHits, float deltaTime, float width, float height, float depth)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numParticles || particles[i].active != 1)
		return;

	Particle& particle = particles[i];

	particle.applyPhysics(deltaTime);
	particle.handleWalls(width, depth);

	if (particle.hasHitFloor())
	{
		atomicAdd(floorHits, 1u);
		particle.spawn(width, height, depth);
	}
}

__global__ void updateTemperatureKernel(Particle* particles, int numParticles, float deltaTime)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numParticles || particles[i].active != 1)
		return;
	particles[i].cool(deltaTime);
}

__global__ void calculateParticleStatsKernel(Particle* particles, int numParticles, unsigned int* activeParticles, float* totalTemperature)
{
	__shared__ unsigned int sharedActive[BLOCK_SIZE];
	__shared__ float sharedTemperature[BLOCK_SIZE];

	int threadId = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int localActive = 0;
	float localTemperature = 0.0f;

	if (i < numParticles && particles[i].active == 1)
	{
		localActive = 1;
		localTemperature = particles[i].temperature;
	}

	sharedActive[threadId] = localActive;
	sharedTemperature[threadId] = localTemperature;

	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (threadId < stride)
		{
			sharedActive[threadId] += sharedActive[threadId + stride];
			sharedTemperature[threadId] += sharedTemperature[threadId + stride];
		}
		__syncthreads();
	}

	if (threadId == 0)
	{
		atomicAdd(activeParticles, sharedActive[0]);
		atomicAdd(totalTemperature, sharedTemperature[0]);
	}
}

__device__ int getCellIndex(float3 position)
{
	int cellX = static_cast<int>(position.x / CELL_SIZE);
	int cellY = static_cast<int>(position.y / CELL_SIZE);
	int cellZ = static_cast<int>(position.z / CELL_SIZE);

	cellX = max(0, min(cellX, GRID_X - 1));
	cellY = max(0, min(cellY, GRID_Y - 1));
	cellZ = max(0, min(cellZ, GRID_Z - 1));

	return (cellY * GRID_Z + cellZ) * GRID_X + cellX;
}

__global__ void buildGridKernel(Particle* particles, int numParticles, int* cellHeads, int* nextParticle)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= numParticles || particles[i].active != 1)
		return;

	int cellIndex = getCellIndex(particles[i].position);
	nextParticle[i] = atomicExch(&cellHeads[cellIndex], i);
}

__global__ void mergeParticlesKernel(Particle* particles, int numParticles, int* cellHeads, int* nextParticle, unsigned int* mergeCount)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numParticles)
		return;

	// Lock the particle - 0 = inactive, 1 = active, 2 = locked for merging
	if (atomicCAS(&particles[i].active, 1, 2) != 1)
		return;
	
	float3 position = particles[i].position;

	int cellX = static_cast<int>(position.x / CELL_SIZE);
	int cellY = static_cast<int>(position.y / CELL_SIZE);
	int cellZ = static_cast<int>(position.z / CELL_SIZE);

	cellX = max(0, min(cellX, GRID_X - 1));
	cellY = max(0, min(cellY, GRID_Y - 1));
	cellZ = max(0, min(cellZ, GRID_Z - 1));

	float collisionDistanceSq = COLLISION_RADIUS * COLLISION_RADIUS;

	for (int y = -1; y <= 1; y++)
	{
		for (int z = -1; z <= 1; z++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int neighborX = cellX + x;
				int neighborY = cellY + y;
				int neighborZ = cellZ + z;

				if (neighborX < 0 || neighborX >= GRID_X ||
					neighborY < 0 || neighborY >= GRID_Y ||
					neighborZ < 0 || neighborZ >= GRID_Z)
				{
					continue;
				}

				int neighbourCell = (neighborY * GRID_Z + neighborZ) * GRID_X + neighborX;

				int j = cellHeads[neighbourCell];

				while (j != -1)
				{
					if (j != i)
					{
						float3 otherPosition = particles[j].position;

						float dx = position.x - otherPosition.x;
						float dy = position.y - otherPosition.y;
						float dz = position.z - otherPosition.z;
						float distSq = dx * dx + dy * dy + dz * dz;
						if (distSq < collisionDistanceSq)
						{
							if (atomicCAS(&particles[j].active, 1, 0) == 1)
							{
								particles[i].mergeWith(particles[j]);
								atomicAdd(mergeCount, 1u);
							}
						}
					}

					// Move to the next particle in the cell
					j = nextParticle[j];
				}
			}
		}
	}

	// Unlock the particle
	particles[i].active = 1;
}	

void runCudaSimulationStep(Particle* d_particles, unsigned int* d_floorHits, int* d_cellHeads, int* d_nextParticle, unsigned int* d_mergedCount, int blocks)
{
	updateParticlesKernel << <blocks, BLOCK_SIZE >> > (d_particles, NUM_PARTICLES, d_floorHits, DELTA_TIME, WIDTH, HEIGHT, DEPTH);
	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceSynchronize());

	checkCudaError(cudaMemset(d_cellHeads, -1, NUM_CELLS * sizeof(int)));

	buildGridKernel << <blocks, BLOCK_SIZE >> > (d_particles, NUM_PARTICLES, d_cellHeads, d_nextParticle);
	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceSynchronize());

	mergeParticlesKernel << <blocks, BLOCK_SIZE >> > (d_particles, NUM_PARTICLES, d_cellHeads, d_nextParticle, d_mergedCount);
	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceSynchronize());

	updateTemperatureKernel << <blocks, BLOCK_SIZE >> > (d_particles, NUM_PARTICLES, DELTA_TIME);
	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceSynchronize());
}

int main()
{	
	Particle* d_particles = nullptr;
	unsigned int* d_floorHits = nullptr;
	unsigned int* d_activeParticles = nullptr;
	float* d_totalTemperature = nullptr;

	int* d_cellHeads = nullptr;
	int* d_nextParticle = nullptr;
	unsigned int* d_mergeCount = nullptr;

    checkCudaError(cudaMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle)));
	checkCudaError(cudaMalloc(&d_floorHits, sizeof(unsigned int)));
	checkCudaError(cudaMalloc(&d_activeParticles, sizeof(unsigned int)));
	checkCudaError(cudaMalloc(&d_totalTemperature, sizeof(float)));
	checkCudaError(cudaMalloc(&d_cellHeads, NUM_CELLS * sizeof(int)));
	checkCudaError(cudaMalloc(&d_nextParticle, NUM_PARTICLES * sizeof(int)));
	checkCudaError(cudaMalloc(&d_mergeCount, sizeof(unsigned int)));

	int blocks = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

	cudaEvent_t startEvent;
	cudaEvent_t endEvent;

	checkCudaError(cudaEventCreate(&startEvent));
	checkCudaError(cudaEventCreate(&endEvent));

	initialiseParticlesKernel << <blocks, BLOCK_SIZE >> > (d_particles, NUM_PARTICLES, WIDTH, HEIGHT, DEPTH);
	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceSynchronize());

	OpenGLRenderer renderer(900, 700, "CUDA Shower Simulation");

	std::vector<Particle> h_particles(NUM_PARTICLES);	

	std::vector<RenderVertex> particleVertices;
	std::vector<RenderVertex> cubicleVertices = buildCubicleVertices(WIDTH, HEIGHT, DEPTH);
	std::vector<RenderVertex> showerHeadVertices = buildShowerHeadVertices(WIDTH, HEIGHT, DEPTH, EMITTER_RADIUS);

	RenderMode renderMode = RenderMode::Mass;

	int renderStride = 100;
	int frame = 0;

	while (!renderer.shouldClose())
	{
		renderer.pollEvents();

		if (renderer.isKeyPressed(GLFW_KEY_ESCAPE))
		{
			break;
		}

		if (renderer.isKeyPressed(GLFW_KEY_T))
		{
			renderMode = RenderMode::Thermal;
		}
		
		if (renderer.isKeyPressed(GLFW_KEY_M))
		{
			renderMode = RenderMode::Mass;
		}

		checkCudaError(cudaEventRecord(startEvent));

		runCudaSimulationStep(d_particles, d_floorHits, d_cellHeads, d_nextParticle, d_mergeCount, blocks);

		checkCudaError(cudaEventRecord(endEvent));
		checkCudaError(cudaEventSynchronize(endEvent));

		checkCudaError(cudaMemcpy(h_particles.data(), d_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost));

		particleVertices = buildRenderVertices(h_particles, renderMode, renderStride, WIDTH, HEIGHT, DEPTH);

		renderer.drawFrame(particleVertices, cubicleVertices, showerHeadVertices);

		if (frame % 300 == 0)
		{
			unsigned int activeParticles = 0;
			unsigned int floorHits = 0;
			unsigned int mergeCount = 0;
			float totalTemperature = 0.0f;

			checkCudaError(cudaMemset(d_activeParticles, 0, sizeof(unsigned int)));
			checkCudaError(cudaMemset(d_totalTemperature, 0, sizeof(float)));

			calculateParticleStatsKernel << <blocks, BLOCK_SIZE >> > (d_particles, NUM_PARTICLES, d_activeParticles, d_totalTemperature);
			checkCudaError(cudaGetLastError());
			checkCudaError(cudaDeviceSynchronize());

			checkCudaError(cudaMemcpy(&activeParticles, d_activeParticles, sizeof(unsigned int), cudaMemcpyDeviceToHost));
			checkCudaError(cudaMemcpy(&floorHits, d_floorHits, sizeof(unsigned int), cudaMemcpyDeviceToHost));
			checkCudaError(cudaMemcpy(&mergeCount, d_mergeCount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
			checkCudaError(cudaMemcpy(&totalTemperature, d_totalTemperature, sizeof(float), cudaMemcpyDeviceToHost));

			float averageTemperature = activeParticles > 0 ? totalTemperature / activeParticles : 0.0f;

			std::cout << "Frame: " << frame
				<< " | Active Particles: " << activeParticles
				<< " | Floor Hits: " << floorHits
				<< " | Merges: " << mergeCount
				<< " | Average Temperature: " << averageTemperature
				<< std::endl;
		}
		frame++;
	}

	checkCudaError(cudaFree(d_particles));
	checkCudaError(cudaFree(d_activeParticles));
	checkCudaError(cudaFree(d_floorHits));
	checkCudaError(cudaFree(d_totalTemperature));
	checkCudaError(cudaFree(d_cellHeads));
	checkCudaError(cudaFree(d_nextParticle));
	checkCudaError(cudaEventDestroy(startEvent));
	checkCudaError(cudaEventDestroy(endEvent));
	checkCudaError(cudaFree(d_mergeCount));

	return 0;
}