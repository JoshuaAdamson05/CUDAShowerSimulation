# CUDA Shower Particle Simulation

**C++ | CUDA | OpenGL | GPU Programming | Performance Optimisation**

A GPU-accelerated 3D particle simulation developed in C++ and CUDA, focused on parallel simulation, collision detection, spatial partitioning and performance optimisation.

The simulation models water particles emitted from a shower head into a 3D cubicle. Particles are affected by gravity, interact with the surrounding environment, merge when they collide and gradually cool over time.

CUDA is used to process the simulation in parallel on the GPU, while OpenGL is used to visualise a sampled subset of particles in real time.

![CUDA Shower Simulation - Thermal Mode](docs/images/Thermal%20CUDA.gif)

---

## What I Implemented

- Developed the particle simulation in C++ using CUDA
- Implemented GPU-parallel particle physics
- Built circular shower-head particle spawning with randomised emission direction
- Implemented gravity, wall reflection and floor-hit detection
- Added automatic particle respawning after particles reach the floor
- Built a uniform 3D spatial grid for collision detection
- Implemented particle collision and merging
- Used CUDA atomic operations to coordinate concurrent particle merges
- Implemented mass-weighted position, velocity and temperature after merging
- Added mass-dependent particle cooling
- Built thermal and mass-based visualisation modes
- Implemented GPU-side render selection and compaction
- Added detailed CUDA stage profiling to identify performance bottlenecks
- Iteratively optimised the collision and rendering pipeline based on profiling results

---

## Simulation Architecture

The simulation is split into several CUDA stages, allowing different parts of the particle system to run in parallel across the GPU.

The main simulation stages are:

1. Particle physics
2. Spatial-grid clearing
3. Spatial-grid construction
4. Collision detection and merging
5. Survivor-state reset
6. Temperature update
7. Render-particle selection and transfer

Particles remain in GPU memory while the simulation is running. Only the smaller set of vertices required for rendering is transferred back to the CPU each frame.

---

## Particle Simulation

Each active particle stores:

- position
- velocity
- mass
- temperature
- active state
- per-particle random-number state

Particles are emitted from a circular shower head at the top of the cubicle and given an initial velocity within a downward-facing cone.

Each simulation step applies gravity and updates the particle position. Particles reflect from the sides of the cubicle, while particles reaching the floor are counted and immediately respawned from the shower head.

---

## Spatial Grid

A uniform 3D spatial grid is used to accelerate collision detection.

Without spatial partitioning, checking every particle against every other particle would become prohibitively expensive as the particle count increased.

Each active particle is assigned to a grid cell based on its position. Particles within relevant nearby cells can then be tested for collision without searching the complete particle set.

Each grid cell stores the index of its first particle, while a separate array links each particle to the next particle occupying the same cell. This allows any number of particles to share a grid cell without allocating a fixed number of particle slots per cell.

To reduce unnecessary work, the implementation also records which cells were occupied and clears only those cells before rebuilding the grid.

---

## Collision Detection and Merging

Particles are considered to collide when the squared distance between them is within the configured collision radius.

When two particles merge, the resulting particle:

- combines the mass of both particles
- calculates position using mass-weighted averaging
- calculates velocity using mass-weighted momentum
- calculates temperature using mass-weighted averaging

The consumed particle is then marked as inactive.

CUDA atomic operations are used to coordinate particle state changes so that multiple GPU threads cannot consume the same particle at the same time.

---

## Temperature

Particles begin with a temperature value of `1.0` and gradually cool towards `0.0`.

Cooling is inversely proportional to particle mass, meaning larger particles created through merging retain heat for longer.

---

## Rendering

The simulation is visualised using OpenGL.

Rendering every simulated particle is unnecessary at high particle counts, so only a sampled subset is displayed.

Rather than transferring the complete particle array from GPU memory every frame, renderable particles are selected and converted into lightweight render vertices on the GPU. Only this compact render data is copied back to the CPU.

Two visualisation modes are available.

### Thermal Mode

Particle colour represents temperature:

- **Red** — hotter particles
- **Blue** — colder particles

![CUDA Shower Simulation - Thermal Mode](docs/images/Thermal%20CUDA.gif)

### Mass Mode

Particle colour represents mass:

- **Blue** — standard / low-mass particles
- **Green** — particles that have increased in mass
- **Yellow** — larger merged particles
- **Red** — the highest-mass particles

Higher-mass merged particles are prioritised during render selection so that the results of particle merging remain visible even when only a sampled subset of the simulation is displayed.

![CUDA Shower Simulation - Mass Mode](docs/images/Mass%20CUDA.gif)

---

## Performance Optimisation

The implementation was progressively profiled and optimised rather than relying only on the initial version.

### Smaller Spatial-Grid Cells

The grid cell size was reduced to better match the scale of the collision system.

A smaller cell size reduces the number of unrelated particles contained within each cell and therefore reduces the number of collision candidates.

Current values:

```cpp
constexpr float COLLISION_RADIUS = 0.0001f;
constexpr float CELL_SIZE = 0.005f;
```

### Collision-Bounded Cell Search

The original collision implementation searched a fixed **3 × 3 × 3 neighbourhood**, meaning up to 27 grid cells could be visited for every particle.

Profiling showed this was unnecessary because the collision radius is much smaller than the grid-cell size.

The collision system was changed to calculate the exact range of cells intersected by each particle's collision-radius bounds.

A particle will therefore normally search only its current cell, while adjacent cells are included only when the particle is close enough to a cell boundary.

This produced a major reduction in collision-processing time in dense simulations.

### Sparse Grid Clearing

Instead of resetting every grid cell each frame, the simulation records which cells were occupied.

Only those cells are cleared before the next grid build, avoiding work on large areas of the grid that contain no particles.

### Reduced CUDA Synchronisation

Unnecessary `cudaDeviceSynchronize()` calls between simulation stages were removed.

The simulation relies on CUDA's ordered execution within the default stream where explicit device-wide synchronisation is not required.

### GPU-Side Render Compaction

The original implementation transferred the complete particle array from GPU memory back to the CPU for rendering.

This was replaced with GPU-side render selection and compaction.

Only the vertices that will actually be rendered are transferred to the CPU, significantly reducing device-to-host transfer bandwidth.

---

## Profiling

The program contains built-in profiling for both individual CUDA stages and overall frame cost.

Measured stages include:

- particle physics
- grid clearing
- grid construction
- collision and merging
- survivor-state reset
- temperature update
- compact render selection and device-to-host transfer
- CPU render-buffer preparation
- rendering and presentation
- complete simulation time
- complete frame time

This profiling was used to identify collision processing and unnecessary memory transfer as the main bottlenecks in the earlier implementation.

---

## Performance Results

Performance profiling was carried out on an **NVIDIA GeForce RTX 4070 Ti** using a **500,000-particle stress test**.

To reduce the effect of one-off timing variation, the final optimised implementation was tested across three runs using the same simulation configuration.

### Final Optimised Results

| Metric | Run 1 | Run 2 | Run 3 | 3-Run Average |
|---|---:|---:|---:|---:|
| Initial dense-frame collision/merge | 25.086 ms | 24.852 ms | 24.058 ms | **24.665 ms** |
| Collision/merge, frames 1-300 mean | 0.583 ms | 0.601 ms | 0.599 ms | **0.595 ms** |
| Simulation wall, frames 1-300 mean | 0.947 ms | 0.938 ms | 0.904 ms | **0.929 ms** |

The initial frame represents the most collision-dense part of the simulation, where almost all 500,000 particles are concentrated near the emitter.

Across frames 1-300, the optimised collision stage averaged approximately **0.595 ms**, while the complete CUDA simulation step averaged approximately **0.929 ms**.

### Effect of the Collision Search Optimisation

Before the collision-bounded search was introduced, a 500,000-particle stress test produced an initial dense-frame collision time of approximately **380.7 ms** in a single baseline run.

The final optimised implementation averaged approximately **24.7 ms** for the same initial dense-frame collision stage across three runs.

The earlier fixed 27-cell search also averaged approximately **6.27 ms** across frames 1-300 in the baseline run, compared with approximately **0.595 ms** for the final optimised version.

These measurements are included to show the effect of the optimisation process rather than as hardware-independent performance claims. Results will vary with GPU hardware, drivers and system conditions.

---

## Limitations and Future Work

The current implementation transfers the compacted render data from CUDA device memory back to the CPU before uploading it to OpenGL. A potential future optimisation would be to use **CUDA-OpenGL interoperability**, allowing CUDA to write directly to an OpenGL buffer and avoid this intermediate CPU transfer.

Further work could also investigate alternative spatial-partitioning approaches and compare performance across different GPU architectures.

## Controls

| Key | Action |
|---|---|
| `T` | Thermal visualisation |
| `M` | Mass visualisation |
| `Esc` | Exit |

---

## Technologies

- **C++**
- **CUDA**
- **OpenGL**
- **GLFW**
- **GLEW**
- **Visual Studio**
- **vcpkg**

---

## Project Structure

```text
CUDAShowerSimulation/
│
├── main.cu
│   CUDA kernels, simulation loop, spatial grid,
│   collision handling, profiling and render selection
│
├── Particle.cuh
│   Particle state, spawning, physics, merging
│   and temperature behaviour
│
├── OpenGLRenderer.cpp / .h
│   OpenGL rendering, shaders and window management
│
└── Rendering.h
    Render structures and static geometry generation
```

---

## Building

### Requirements

- Windows
- NVIDIA CUDA-capable GPU
- Visual Studio with C++ development tools
- CUDA Toolkit
- vcpkg

The current Visual Studio project is configured for **CUDA 12.9**.

OpenGL dependencies are managed through `vcpkg`:

- GLFW
- GLEW

The repository contains a `vcpkg.json` manifest for these dependencies.

For performance testing, the project should be built using the **Release x64** configuration.

---

## Configuration

The main simulation parameters can be changed near the top of `main.cu`.

```cpp
constexpr int NUM_PARTICLES = 100000;
constexpr int BLOCK_SIZE = 256;

constexpr float WIDTH = 1.0f;
constexpr float HEIGHT = 2.0f;
constexpr float DEPTH = 1.0f;

constexpr float COLLISION_RADIUS = 0.0001f;
constexpr float CELL_SIZE = 0.005f;

constexpr int RENDER_STRIDE = 100;
```

The default configuration uses **100,000 particles**.

Higher particle counts can be used to stress-test the simulation and compare GPU performance.

---

## Technical Areas

`C++` `CUDA` `GPU Programming` `Parallel Computing` `Spatial Partitioning` `Collision Detection` `Atomic Operations` `OpenGL` `Performance Profiling`
