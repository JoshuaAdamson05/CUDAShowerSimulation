#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <array>
#include <iomanip>

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
constexpr float CELL_SIZE = 0.005f;
constexpr float EMITTER_RADIUS = 0.05f;

constexpr int GRID_X = static_cast<int>(WIDTH / CELL_SIZE) + 1;
constexpr int GRID_Y = static_cast<int>(HEIGHT / CELL_SIZE) + 1;
constexpr int GRID_Z = static_cast<int>(DEPTH / CELL_SIZE) + 1;

constexpr int NUM_CELLS = GRID_X * GRID_Y * GRID_Z;

static_assert(NUM_CELLS > 0, "NUM_CELLS must be positive");
static_assert(CELL_SIZE >= COLLISION_RADIUS, "CELL_SIZE must be at least as large as COLLISION_RADIUS for the 27-cell neighbour search");

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

__global__ void clearOccupiedCellsKernel(int* cellHeads, const int* occupiedCells, const unsigned int* occupiedCellCount)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < *occupiedCellCount)
		cellHeads[occupiedCells[i]] = -1;
}

__global__ void buildGridKernel(Particle* particles, int numParticles, int* cellHeads, int* nextParticle, int* occupiedCells, unsigned int* occupiedCellCount)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= numParticles || particles[i].active != 1)
		return;

	int cellIndex = getCellIndex(particles[i].position);
	int previousHead = atomicExch(&cellHeads[cellIndex], i);
	nextParticle[i] = previousHead;

	if (previousHead == -1)
	{
		unsigned int slot = atomicAdd(occupiedCellCount, 1u);
		occupiedCells[slot] = cellIndex;
	}
}

__global__ void mergeParticlesKernel(Particle* particles, int numParticles, int* cellHeads, int* nextParticle, unsigned int* mergeCount)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numParticles)
		return;

	if (particles[i].active != 1)
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
					if (j != i && particles[j].active == 1)
					{
						float3 otherPosition = particles[j].position;

						float dx = position.x - otherPosition.x;
						float dy = position.y - otherPosition.y;
						float dz = position.z - otherPosition.z;
						float distSq = dx * dx + dy * dy + dz * dz;
						if (distSq <= collisionDistanceSq)
						{
							if (atomicCAS(&particles[i].active, 1, 2) != 1)
								return;

							if (atomicCAS(&particles[j].active, 1, 0) == 1)
							{
								particles[i].mergeWith(particles[j]);
								atomicAdd(mergeCount, 1u);

								particles[i].active = 3; // Mark as merged but not yet unlocked
								return;
							}

							particles[i].active = 1; // Unlock if merge failed
						}
					}

					// Move to the next particle in the cell
					j = nextParticle[j];
				}
			}
		}
	}
}	

__global__ void resetSurvivorsKernel(Particle* particles, int numParticles)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= numParticles)
		return;

	if (particles[i].active == 3)
	{
		particles[i].active = 1; // Unlock merged particles for the next iteration
	}
}


using ProfileClock = std::chrono::steady_clock;

double millisecondsSince(ProfileClock::time_point start)
{
	return std::chrono::duration<double, std::milli>(ProfileClock::now() - start).count();
}

struct ProfileWindow
{
	std::array<double, 11> sums{};
	std::array<double, 11> peaks{};
	int samples = 0;

	void add(const std::array<double, 11>& values)
	{
		for (size_t i = 0; i < values.size(); ++i)
		{
			sums[i] += values[i];
			if (values[i] > peaks[i]) peaks[i] = values[i];
		}
		++samples;
	}

	void print(int lastFrame) const
	{
		static const char* names[] = {
			"Physics GPU", "Grid clear GPU", "Grid build GPU", "Collision/merge GPU",
			"Survivor reset GPU", "Temperature GPU", "Particle D2H wall",
			"CPU vertices wall", "Render+swap wall", "Simulation wall",
			"Frame through swap wall (excludes stats/logging)"
		};
		const auto oldFlags = std::cout.flags();
		const auto oldPrecision = std::cout.precision();
		std::cout << std::fixed << std::setprecision(4)
			<< "PROFILE frames " << (lastFrame - samples + 1) << "-" << lastFrame
			<< " | samples=" << samples << " | mean / max ms\n";
		for (size_t i = 0; i < sums.size(); ++i)
			std::cout << "  " << names[i] << ": " << sums[i] / samples
				<< " / " << peaks[i] << "\n";
		std::cout.flags(oldFlags);
		std::cout.precision(oldPrecision);
	}
};

void runCudaSimulationStep(Particle* d_particles, unsigned int* d_floorHits, int* d_cellHeads, int* d_nextParticle, unsigned int* d_frameMerges, int blocks, cudaEvent_t* stageEvents, int* d_occupiedCells, unsigned int* d_occupiedCellCount)
{
	// All work uses the default CUDA stream, so these operations execute in order
	// without forcing the CPU to wait after every kernel launch.
	updateParticlesKernel << <blocks, BLOCK_SIZE >> > (d_particles, NUM_PARTICLES, d_floorHits, DELTA_TIME, WIDTH, HEIGHT, DEPTH);
	checkCudaError(cudaGetLastError());

	checkCudaError(cudaEventRecord(stageEvents[1]));

	clearOccupiedCellsKernel << <blocks, BLOCK_SIZE >> > (d_cellHeads, d_occupiedCells, d_occupiedCellCount);
	checkCudaError(cudaGetLastError());
	checkCudaError(cudaMemset(d_occupiedCellCount, 0, sizeof(unsigned int)));
	checkCudaError(cudaEventRecord(stageEvents[2]));

	buildGridKernel << <blocks, BLOCK_SIZE >> > (d_particles, NUM_PARTICLES, d_cellHeads, d_nextParticle, d_occupiedCells, d_occupiedCellCount);
	checkCudaError(cudaGetLastError());

	checkCudaError(cudaEventRecord(stageEvents[3]));

	checkCudaError(cudaMemset(d_frameMerges, 0, sizeof(unsigned int)));
	mergeParticlesKernel << <blocks, BLOCK_SIZE >> > (d_particles, NUM_PARTICLES, d_cellHeads, d_nextParticle, d_frameMerges);
	checkCudaError(cudaGetLastError());

	checkCudaError(cudaEventRecord(stageEvents[4]));

	resetSurvivorsKernel << <blocks, BLOCK_SIZE >> > (d_particles, NUM_PARTICLES);
	checkCudaError(cudaGetLastError());

	checkCudaError(cudaEventRecord(stageEvents[5]));

	updateTemperatureKernel << <blocks, BLOCK_SIZE >> > (d_particles, NUM_PARTICLES, DELTA_TIME);
	checkCudaError(cudaGetLastError());
}

int main()
{	
	Particle* d_particles = nullptr;
	unsigned int* d_floorHits = nullptr;
	unsigned int* d_activeParticles = nullptr;
	float* d_totalTemperature = nullptr;

	int* d_cellHeads = nullptr;
	int* d_nextParticle = nullptr;
	int* d_occupiedCells = nullptr;
	unsigned int* d_occupiedCellCount = nullptr;
	unsigned int* d_frameMerges = nullptr;
	unsigned int totalMerges = 0;
	unsigned int mergesSinceLastLog = 0;

    checkCudaError(cudaMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle)));
	checkCudaError(cudaMalloc(&d_floorHits, sizeof(unsigned int)));
	checkCudaError(cudaMalloc(&d_activeParticles, sizeof(unsigned int)));
	checkCudaError(cudaMalloc(&d_totalTemperature, sizeof(float)));
	checkCudaError(cudaMalloc(&d_cellHeads, NUM_CELLS * sizeof(int)));
	checkCudaError(cudaMalloc(&d_nextParticle, NUM_PARTICLES * sizeof(int)));
	checkCudaError(cudaMalloc(&d_occupiedCells, NUM_PARTICLES * sizeof(int)));
	checkCudaError(cudaMalloc(&d_occupiedCellCount, sizeof(unsigned int)));
	checkCudaError(cudaMemset(d_cellHeads, -1, NUM_CELLS * sizeof(int)));
	checkCudaError(cudaMemset(d_occupiedCellCount, 0, sizeof(unsigned int)));
	checkCudaError(cudaMalloc(&d_frameMerges, sizeof(unsigned int)));

	int blocks = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

	cudaEvent_t stageEvents[7];
	for (auto& event : stageEvents) checkCudaError(cudaEventCreate(&event));
	const cudaEvent_t startEvent = stageEvents[0];
	const cudaEvent_t endEvent = stageEvents[6];
	ProfileWindow profileWindow;
	std::cout << "Profiling enabled: CUDA event intervals and CPU wall timings. "
		<< "Render+swap includes submission/presentation waits, not isolated GPU rendering. "
		<< "Timings overlap (simulation wall includes GPU work); do not add all rows.\n";

	float totalGpuFrameTime = 0.0f;

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
		const auto frameStart = ProfileClock::now();
		std::array<double, 11> timings{};
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

		const auto simulationStart = ProfileClock::now();
		checkCudaError(cudaEventRecord(startEvent));

		runCudaSimulationStep(d_particles, d_floorHits, d_cellHeads, d_nextParticle, d_frameMerges, blocks, stageEvents, d_occupiedCells, d_occupiedCellCount);

		checkCudaError(cudaEventRecord(endEvent));
		checkCudaError(cudaEventSynchronize(endEvent));
		timings[9] = millisecondsSince(simulationStart);
		for (int stage = 0; stage < 6; ++stage)
		{
			float elapsed = 0.0f;
			checkCudaError(cudaEventElapsedTime(&elapsed, stageEvents[stage], stageEvents[stage + 1]));
			timings[stage] = elapsed;
		}

		unsigned int frameMerges = 0;
		checkCudaError(cudaMemcpy(&frameMerges, d_frameMerges, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		totalMerges += frameMerges;

		mergesSinceLastLog += frameMerges;

		float frameMilliseconds = 0.0f;
		checkCudaError(cudaEventElapsedTime(&frameMilliseconds, startEvent, endEvent));
		totalGpuFrameTime += frameMilliseconds;

		const auto transferStart = ProfileClock::now();
		checkCudaError(cudaMemcpy(h_particles.data(), d_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost));

		timings[6] = millisecondsSince(transferStart);

		const auto verticesStart = ProfileClock::now();
		particleVertices = buildRenderVertices(h_particles, renderMode, renderStride, WIDTH, HEIGHT, DEPTH);

		timings[7] = millisecondsSince(verticesStart);

		const auto renderStart = ProfileClock::now();
		renderer.drawFrame(particleVertices, cubicleVertices, showerHeadVertices);

		timings[8] = millisecondsSince(renderStart);
		timings[10] = millisecondsSince(frameStart);
		profileWindow.add(timings);

		if (frame % 300 == 0)
		{
			unsigned int activeParticles = 0;
			unsigned int floorHits = 0;
			float totalTemperature = 0.0f;

			checkCudaError(cudaMemset(d_activeParticles, 0, sizeof(unsigned int)));
			checkCudaError(cudaMemset(d_totalTemperature, 0, sizeof(float)));

			calculateParticleStatsKernel << <blocks, BLOCK_SIZE >> > (d_particles, NUM_PARTICLES, d_activeParticles, d_totalTemperature);
			checkCudaError(cudaGetLastError());
			checkCudaError(cudaDeviceSynchronize());

			checkCudaError(cudaMemcpy(&activeParticles, d_activeParticles, sizeof(unsigned int), cudaMemcpyDeviceToHost));
			checkCudaError(cudaMemcpy(&floorHits, d_floorHits, sizeof(unsigned int), cudaMemcpyDeviceToHost));
			checkCudaError(cudaMemcpy(&totalTemperature, d_totalTemperature, sizeof(float), cudaMemcpyDeviceToHost));

			float averageTemperature = activeParticles > 0 ? totalTemperature / activeParticles : 0.0f;

			std::cout << "Frame: " << frame
				<< " | Active Particles: " << activeParticles
				<< " | Floor Hits: " << floorHits
				<< " | Merges Since Last Log: " << mergesSinceLastLog
				<< " | Total Merges: " << totalMerges
				<< " | Average Temperature: " << averageTemperature
				<< " | GPU Frame Time: " << frameMilliseconds << " ms"
				<< " | Average GPU Frame Time: " << (totalGpuFrameTime / (frame + 1)) << " ms"
				<< std::endl;

			profileWindow.print(frame);
			profileWindow = ProfileWindow{};
			mergesSinceLastLog = 0;
		}
		frame++;
	}

	if (profileWindow.samples > 0) profileWindow.print(frame - 1);

	checkCudaError(cudaFree(d_particles));
	checkCudaError(cudaFree(d_activeParticles));
	checkCudaError(cudaFree(d_floorHits));
	checkCudaError(cudaFree(d_totalTemperature));
	checkCudaError(cudaFree(d_cellHeads));
	checkCudaError(cudaFree(d_nextParticle));
	checkCudaError(cudaFree(d_occupiedCells));
	checkCudaError(cudaFree(d_occupiedCellCount));
	for (auto event : stageEvents) checkCudaError(cudaEventDestroy(event));
	checkCudaError(cudaFree(d_frameMerges));

	return 0;
}