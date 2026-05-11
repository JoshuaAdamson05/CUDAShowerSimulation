#include "OpenGLRenderer.h"

#include <iostream>
#include <cstdlib>

#pragma comment(lib, "opengl32.lib")

struct Vec3
{
	float x;
	float y;
	float z;
};

static Vec3 normalize(Vec3 v)
{
	float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

	return {
		v.x / length,
		v.y / length,
		v.z / length
	};
}

static Vec3 cross(Vec3 a, Vec3 b)
{
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

static float dot(Vec3 a, Vec3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

static void multiplyMatrix(const float a[16], const float b[16], float result[16])
{
	for (int row = 0; row < 4; row++)
	{
		for (int col = 0; col < 4; col++)
		{
			result[row * 4 + col] =
				a[row * 4 + 0] * b[0 * 4 + col] +
				a[row * 4 + 1] * b[1 * 4 + col] +
				a[row * 4 + 2] * b[2 * 4 + col] +
				a[row * 4 + 3] * b[3 * 4 + col];
		}
	}
}

OpenGLRenderer::OpenGLRenderer(int width, int height, const char* title) : window(nullptr), pointShaderProgram(0), flatShaderProgram(0), vao(0), vbo(0), width(width), height(height)
{
	if (!glfwInit())
	{
		std::cerr << "Failed to initialize GLFW" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(width, height, title, nullptr, nullptr);

	if (!window)
	{
		std::cerr << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		std::exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;

	GLenum glewResult = glewInit();

	if (glewResult != GLEW_OK)
	{
		std::cerr << "Failed to initialize GLEW" << std::endl;
		glfwDestroyWindow(window);
		glfwTerminate();
		std::exit(EXIT_FAILURE);
	}
	glViewport(0, 0, width, height);

	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_DEPTH_TEST);

	pointShaderProgram = createPointShaderProgram();
	flatShaderProgram = createFlatShaderProgram();

	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);

	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), reinterpret_cast<void*>(0));

	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), reinterpret_cast<void*>(3 * sizeof(float)));

	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

OpenGLRenderer::~OpenGLRenderer()
{
	glDeleteBuffers(1, &vbo);
	glDeleteVertexArrays(1, &vao);
	glDeleteProgram(pointShaderProgram);

	if (window)
	{
		glfwDestroyWindow(window);
	}

	glfwTerminate();
}

bool OpenGLRenderer::shouldClose() const
{
	return glfwWindowShouldClose(window);
}

void OpenGLRenderer::pollEvents()
{
	glfwPollEvents();
}

bool OpenGLRenderer::isKeyPressed(int key) const
{
	return glfwGetKey(window, key) == GLFW_PRESS;
}

void OpenGLRenderer::drawParticles(const std::vector<RenderVertex>& vertices)
{
	glClearColor(0.02f, 0.02f, 0.02f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(pointShaderProgram);
	setCameraMatrix(pointShaderProgram);

	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	if (!vertices.empty())
	{
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(RenderVertex), vertices.data(), GL_DYNAMIC_DRAW);
		glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(vertices.size()));
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	glfwSwapBuffers(window);
}

GLuint OpenGLRenderer::compileShader(GLenum type, const char* source)
{
	GLuint shader = glCreateShader(type);

	glShaderSource(shader, 1, &source, nullptr);
	glCompileShader(shader);

	GLint success = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

	if (!success)
	{
		GLchar infoLog[512];
		glGetShaderInfoLog(shader, 512, nullptr, infoLog);

		std::cerr << "Shader compilation failed: " << infoLog << std::endl;

		std::exit(EXIT_FAILURE);
	}
	return shader;
}

GLuint OpenGLRenderer::createPointShaderProgram()
{
	const char* vertexShaderSource = R"(
		#version 330 core

		layout(location = 0) in vec3 aPos;
		layout(location = 1) in vec3 aColor;

		out vec3 vertexColor;

		uniform mat4 matrix;

		void main()
		{	
			vertexColor = aColor;
			gl_Position = matrix * vec4(aPos, 1.0);
			gl_PointSize = 8.0;
		}
	)";

	const char* fragmentShaderSource = R"(
		#version 330 core

		in vec3 vertexColor;
		out vec4 FragColor;

		void main()
		{
			vec2 coord = gl_PointCoord - vec2(0.5);

		if (length(coord) > 0.5)
				discard;

			FragColor = vec4(vertexColor, 1.0);
		}
	)";

	GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
	GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

	GLuint program = glCreateProgram();

	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);

	glLinkProgram(program);

	GLint success = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &success);

	if (!success)
	{
		GLchar infoLog[512];
		glGetProgramInfoLog(program, 512, nullptr, infoLog);
		std::cerr << "Shader program linking failed: " << infoLog << std::endl;
		std::exit(EXIT_FAILURE);
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return program;
}

GLuint OpenGLRenderer::createFlatShaderProgram()
{
	const char* vertexShaderSource = R"(
		#version 330 core

		layout(location = 0) in vec3 aPos;
		layout(location = 1) in vec3 aColor;

		out vec3 vertexColor;

		uniform mat4 matrix;

		void main()
		{	
			vertexColor = aColor;
			gl_Position = matrix * vec4(aPos, 1.0);
		}
	)";

	const char* fragmentShaderSource = R"(
		#version 330 core

		in vec3 vertexColor;
		out vec4 FragColor;

		void main()
		{
			FragColor = vec4(vertexColor, 1.0);
		}
	)";

	GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
	GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

	GLuint program = glCreateProgram();

	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);
	glLinkProgram(program);

	GLint success = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if (!success)
	{
		GLchar infoLog[512];
		glGetProgramInfoLog(program, 512, nullptr, infoLog);
		std::cerr << "Shader program linking failed: " << infoLog << std::endl;
		std::exit(EXIT_FAILURE);
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return program;
}

void OpenGLRenderer::setCameraMatrix(GLuint shader)
{
	const float fovRadians = 3.14159265f / 4.0f;
	const float aspect = static_cast<float>(width) / static_cast<float>(height);
	const float nearPlane = 0.1f;
	const float farPlane = 100.0f;

	const float f = 1.0f / tanf(fovRadians / 2.0f);

	float perspective[16] =
	{
		f / aspect, 0.0f, 0.0f,                                      0.0f,
		0.0f,       f,    0.0f,                                      0.0f,
		0.0f,       0.0f, (farPlane + nearPlane) / (nearPlane - farPlane), (2.0f * farPlane * nearPlane) / (nearPlane - farPlane),
		0.0f,       0.0f, -1.0f,                                     0.0f
	};

	Vec3 eye = { 1.5f, 1.2f, 3.0f };
	Vec3 direction = { -1.5f, -1.2f, -3.0f };
	Vec3 up = { 0.0f, 1.0f, 0.0f };

	Vec3 fwd = normalize(direction);
	Vec3 side = normalize(cross(fwd, up));
	Vec3 cameraUp = cross(side, fwd);

	float view[16] =
	{
		side.x,      side.y,      side.z,      -dot(side, eye),
		cameraUp.x,  cameraUp.y,  cameraUp.z,  -dot(cameraUp, eye),
	   -fwd.x,      -fwd.y,      -fwd.z,       dot(fwd, eye),
		0.0f,        0.0f,        0.0f,        1.0f
	};

	float matrix[16];
	multiplyMatrix(perspective, view, matrix);

	GLint matrixLocation = glGetUniformLocation(shader, "matrix");

	glUniformMatrix4fv(
		matrixLocation,
		1,
		GL_TRUE,
		matrix
	);
}

void OpenGLRenderer::drawFrame(
	const std::vector<RenderVertex>& particles,
	const std::vector<RenderVertex>& cubicle,
	const std::vector<RenderVertex>& showerHead
)
{
	int framebufferWidth = 0;
	int framebufferHeight = 0;

	glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);

	if (framebufferWidth <= 0 || framebufferHeight <= 0)
	{
		return;
	}

	width = framebufferWidth;
	height = framebufferHeight;

	glViewport(0, 0, framebufferWidth, framebufferHeight);

	glClearColor(0.02f, 0.02f, 0.02f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	drawVertices(cubicle, GL_LINES, flatShaderProgram);
	drawVertices(showerHead, GL_TRIANGLE_FAN, flatShaderProgram);
	drawVertices(particles, GL_POINTS, pointShaderProgram);

	glfwSwapBuffers(window);
}

void OpenGLRenderer::drawVertices(const std::vector<RenderVertex>& vertices, GLenum primitiveType, GLuint shader)
{
	if (vertices.empty())
		return;

	glUseProgram(shader);
	setCameraMatrix(shader);

	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(RenderVertex), vertices.data(), GL_DYNAMIC_DRAW);

	glDrawArrays(primitiveType, 0, static_cast<GLsizei>(vertices.size()));

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}