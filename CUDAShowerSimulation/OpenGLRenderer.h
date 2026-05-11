#pragma once

#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cmath>

#include "Rendering.h"

class OpenGLRenderer
{
public:
	OpenGLRenderer(int width, int height, const char* title);
	~OpenGLRenderer();

	bool shouldClose() const;
	void pollEvents();

	bool isKeyPressed(int key) const;

	void drawParticles(const std::vector<RenderVertex>& vertices);

	void drawFrame(const std::vector<RenderVertex>& particles, const std::vector<RenderVertex>& cubicle, const std::vector<RenderVertex>& showerHead);

private:
	GLFWwindow* window;

	GLuint pointShaderProgram;
	GLuint flatShaderProgram;

	void drawVertices(const std::vector<RenderVertex>& vertices, GLenum primitiveType, GLuint shader);


	GLuint vao;
	GLuint vbo;

	int width;
	int height;

	GLuint compileShader(GLenum type, const char* source);
	GLuint createPointShaderProgram();
	GLuint createFlatShaderProgram();

	void setCameraMatrix(GLuint shader);
};
