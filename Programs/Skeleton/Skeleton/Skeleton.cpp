#include "framework.h"

template<class T> struct Dnum {
	float f;
	T d;
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

struct Camera {
	vec3 wEye, wLookat, wVup;
	float fov = 30.0f * (float)M_PI / 180.0f;
	float asp = (float)windowWidth / windowHeight;
	float fp = 0.3;
	float bp = 50;
public:
	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
												   u.y, v.y, w.y, 0,
												   u.z, v.z, w.z, 0,
												   0, 0, 0, 1);
	}

	mat4 P() {
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
					0, 1 / tan(fov / 2), 0, 0,
					0, 0, -(fp + bp) / (bp - fp), -1,
					0, 0, -2 * fp * bp / (bp - fp), 0);
	}
};

struct Material {
	vec3 kd, ks, ka;
	float shininess;
};

class Light {
public:
	vec3 La, Le;
	vec4 wLightPos;

	Light(vec3 La, vec3 Le, vec4 wLightPos, vec3 pivot) {
		this->La = La;
		this->Le = Le;
		this->wLightPos = wLightPos;
		this->pivot = pivot;
	}

	void Animate(float tStart, float tEnd) {
		q = vec4(sin(tEnd / 4) * cos(tEnd) / 2, sin(tEnd / 4) * sin(tEnd) / 2, sin(tEnd / 4) * sqrtf(0.75), cos(tEnd / 4));
		q = q / sqrtf(dot(q, q));
		qinv = vec4(-q.x, -q.y, -q.z, q.w);

		mat4 M = TranslateMatrix(-pivot) * quatToMatrix(q, qinv) * TranslateMatrix(pivot);
		wLightPos = vec4(0, 0, 0, 1) * M;
	}

private:
	vec4 q, qinv;
	vec3 pivot;

	vec4 qmul(vec4 q1, vec4 q2) {
		vec3 d1(q1.x, q1.y, q1.z), d2(q2.x, q2.y, q2.z);
		vec3 imag = d2 * q1.w + d1 * q2.w + cross(d1, d2);
		return vec4(imag.x, imag.y, imag.z, q1.w * q2.w - dot(d1, d2));
	}

	mat4 quatToMatrix(vec4 q, vec4 qinv) {
		return mat4(qmul(qmul(q, vec4(1, 0, 0, 0)), qinv),
			qmul(qmul(q, vec4(0, 1, 0, 0)), qinv),
			qmul(qmul(q, vec4(0, 0, 1, 0)), qinv),
			vec4(0, 0, 0, 1));
	}
};

struct RenderState {
	mat4 MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light*> lights;
	vec3 wEye;
};

class PhongShader : public GPUProgram {
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv;
		uniform Light[2] lights;
		uniform vec3 wEye;
		uniform bool sheet;

		layout(location = 0) in vec3 vtxPos;
		layout(location = 1) in vec3 vtxNorm;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[2];
		out vec2 texcoord;
		out float zCoord;
		out float rubberSheet;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP;
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < 2; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;

			if (sheet) {
				zCoord = wPos.z / wPos.w;
				rubberSheet = 0;
			}
			else {
				zCoord = 0;
				rubberSheet = 1;
			}
		}
	)";

	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[2] lights;

		in  vec3 wNormal;
		in  vec3 wView;
		in  vec3 wLight[2];
		in  float zCoord;
		in	float rubberSheet;
		
        out vec4 fragmentColor;

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;
			vec3 ka = material.ka;
			vec3 kd = material.kd;
			vec3 ks = material.ks;

			float darken = 1;
			if (zCoord < -2)	darken = 0.9f;
			if (zCoord < -2.5f)	darken = 0.7f;
			if (zCoord < -4)	darken = 0.5f;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < 2; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				vec3 LeIn = lights[i].Le / dot(wLight[i], wLight[i]);
				radiance += darken * ka * lights[i].La + darken * kd * cost * LeIn + rubberSheet * ks * pow(cosd, material.shininess) * LeIn;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light* light, const std::string& name) {
		setUniform(light->La, name + ".La");
		setUniform(light->Le, name + ".Le");
		setUniform(light->wLightPos, name + ".wLightPos");
	}
};

struct VertexData {
	vec3 position, normal;
};

class Geometry {
protected:
	unsigned int vao, vbo;
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}

	virtual void Draw() = 0;

	void Load(const std::vector<VertexData>& vtxData) {
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vtxData.size() * sizeof(VertexData), &vtxData[0], GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
	}

	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

class ParamSurface : public Geometry {
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create(int N = 100, int M = 100) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		this->Load(vtxData);
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

class Sphere : public ParamSurface {
public:
	Sphere() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
	}
};

struct Weight {
	float weight;
	vec2 position;
};

std::vector<Weight> weights;

float sheetWidth = 6;

Dnum2 WeightEval(Dnum2 x, Dnum2 y) {
	Dnum2 result = 0;
	for (Weight w : weights) {
		result = result + (Pow(Pow(Pow(x - w.position.x, 2) + Pow(y - w.position.y, 2), 0.5) + 0.05 * sheetWidth, -1) * w.weight);
	}
	return result * -1;
}

class Sheet : public ParamSurface {
public:
	Sheet() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		X = ((U * 2) - 1) * sheetWidth;
		Y = ((V * 2) - 1) * sheetWidth;
		Z = WeightEval(X, Y);
	}
};

struct Object {
	PhongShader* shader;
	Material* material;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(PhongShader* shader, Material* material, Geometry* geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		this->shader = shader;
		this->material = material;
		this->geometry = geometry;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { }
};

float rnd() {
	return (float)rand() / RAND_MAX;
}

bool defaultView = true;
Camera camera;

class Ball : public Object {

	vec3 velocity = vec3(0, 0, 0);
	vec3 acceleration = vec3(0, 0, 0);
	vec3 touchPoint;
	vec3 normal = vec3(0, 0, 1);
	float m = 5;
	vec3 g = vec3(0, 0, -9.81f);
	float energy = 0;
	bool follow = false;

public:
	Ball() : Object(new PhongShader(), RandomMaterial(), new Sphere()) {
		this->scale = vec3(0.25, 0.25, 0.25);
		this->translation = vec3(-3.694, -3.694, 0.25);
		touchPoint = vec3(translation.x, translation.y, 0);
	}

	void Draw(RenderState state) {
		if (!follow) {
			shader->setUniform(false, "sheet");
			Object::Draw(state);
		}
	}

	void Animate(float tstart, float tend) {
		velocity = velocity + acceleration * (tend - tstart);
		while (Energy() <= energy - 0.1 || Energy() >= energy + 0.1) {
			if (energy - Energy() > 0)
				velocity = velocity * 1.01;
			else
				velocity = velocity * 0.99;
		}

		touchPoint = touchPoint + velocity * (tend - tstart);
		translation = translation + velocity * (tend - tstart);

		Dnum2 Z;
		Dnum2 X(touchPoint.x, vec2(2 * sheetWidth, 0)), Y(touchPoint.y, vec2(0, 2 * sheetWidth));
		Z = WeightEval(X, Y);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		normal = normalize(cross(drdU, drdV));
		touchPoint = vec3(X.f, Y.f, Z.f);
		translation = vec3(X.f, Y.f, Z.f) + normal * 0.25;

		acceleration = m*g - dot(m*g, normal) * normal;
		
		if (touchPoint.x > 4.2) {
			touchPoint = vec3(-4.18, touchPoint.y, 0);
			translation = vec3(-4.18, touchPoint.y, 0.25);
		}
		else if (touchPoint.x < -4.2) {
			touchPoint = vec3(4.18, touchPoint.y, 0);
			translation = vec3(4.18, touchPoint.y, 0.25);
		}
		else if (touchPoint.y > 4.2) {
			touchPoint = vec3(touchPoint.x, -4.18, 0);
			translation = vec3(touchPoint.x, -4.18, 0.25);
		}
		else if (touchPoint.y < -4.2) {
			touchPoint = vec3(touchPoint.x, 4.18, 0);
			translation = vec3(touchPoint.x, 4.18, 0.25);
		}

		if (follow && !defaultView) {
			camera.wEye = translation + vec3(0, 0, 0.3);
			camera.wLookat = camera.wEye + normalize(velocity);
			camera.wVup = vec3(0, 0, 1);
		}
	}

	void SetVelocity(vec3 v) {
		velocity = v - touchPoint;
		energy = Energy();
	}

	float Energy() {
		return 0.5 * m * dot(velocity, velocity) + m * sqrtf(dot(g, g)) * (touchPoint.z + 10);
	}

	void SetFollow(bool b) {
		follow = b;
	}

private:
	Material* RandomMaterial() {
		Material* mat = new Material();
		mat->kd = vec3(rnd(), rnd(), rnd());
		mat->ka = 3.14 * mat->kd;
		mat->ks = vec3(1, 1, 1);
		mat->shininess = 20.0f;
		return mat;
	}
};

class RubberSheet : public Object {
public:
	RubberSheet() : Object(new PhongShader(), SheetMaterial(), new Sheet()) { }

	void Draw(RenderState state) {
		shader->setUniform(true, "sheet");
		Object::Draw(state);
	}

private:
	Material* SheetMaterial() {
		Material* mat = new Material();
		mat->kd = vec3(0.23f, 0.9f, 0.88f);
		//mat->kd = vec3(0.66f, 0.83f, 0.27f);
		mat->ka = 3.14 * mat->kd;
		mat->ks = vec3(1, 1, 1);
		mat->shininess = 1.0f;
		return mat;
	}
};

class Scene {
	Ball* waitingBall;
	Ball* followedBall;
	std::vector<Ball*> balls;
	float nextWeight = 1;
	RubberSheet* sheet;
	std::vector<Light*> lights;
public:
	void Build() {
		sheet = new RubberSheet();
		waitingBall = new Ball();

		camera.wEye = vec3(0, 0, 15);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		vec3 La = vec3(0.1f, 0.1f, 0.1f);
		vec3 Le = vec3(4.0f, 4.0f, 4.0f);
		Light* light1 = new Light(La, Le, vec4(2, 2, 2, 1), vec3(-1, -3, 2));
		Light* light2 = new Light(La, Le, vec4(-1, -3, 2, 1), vec3(2, 2, 2));
		lights.push_back(light1);
		lights.push_back(light2);
	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		sheet->Draw(state);
		waitingBall->Draw(state);
		
		for (Ball* ball : balls) {
			ball->Draw(state);
		}
		DeleteFallenBalls();
	}

	void Animate(float tstart, float tend) {
		for (Ball* ball : balls) ball->Animate(tstart, tend);
		for (Light* light : lights) light->Animate(tstart, tend);
	}

	void StartBall(vec3 v) {
		waitingBall->SetVelocity(v);
		balls.push_back(waitingBall);
		waitingBall = new Ball();
	}

	void PlaceWeight(vec2 pos) {
		Weight w;
		w.weight = nextWeight;
		w.position = pos;
		weights.push_back(w);
		nextWeight += 0.3;
		sheet = new RubberSheet();
	}

	void DeleteFallenBalls() {
		std::vector<std::vector<Ball*>::iterator> deleteList;
		for (int i = 0; i < balls.size(); i++) {
			for (Weight w : weights) {
				if (balls[i]->translation.x <= w.position.x + 0.02 && balls[i]->translation.x >= w.position.x - 0.02 &&
					balls[i]->translation.y <= w.position.y + 0.02 && balls[i]->translation.y >= w.position.y - 0.02)
					deleteList.push_back(balls.begin() + i);
			}
		}
		int j = 0;
		for (int i = 0; i < deleteList.size(); i++) {
			balls.erase(deleteList[i] - j);
			j++;
		}
	}

	void SetView() {
		if (defaultView) {
			camera.fov = 90.0f * (float)M_PI / 180.0f;
			//camera.fov = 60.0f * (float)M_PI / 180.0f; �rdemes ezt is �ll�tgatni
			if (balls.empty()) {
				followedBall = waitingBall;
				followedBall->SetFollow(true);
				camera.wEye = followedBall->translation + vec3(0, 0, 0.3);
				camera.wLookat = normalize(-camera.wEye);
				camera.wVup = vec3(0, 0, 1);
			}
			else {
				followedBall = balls[0];
				followedBall->SetFollow(true);
			}
		}
		else {
			camera.fov = 30.0f * (float)M_PI / 180.0f;
			followedBall->SetFollow(false);
			camera.wEye = vec3(0, 0, 15);
			camera.wLookat = vec3(0, 0, 0);
			camera.wVup = vec3(0, 1, 0);
		}
		defaultView = !defaultView;
	}
};

Scene scene;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

void onDisplay() {
	glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene.Render();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key = ' ') {
		scene.SetView();
	}
	/*else if (key == 'w') {
		camera.wEye = camera.wEye + 0.05 * normalize(camera.wLookat - camera.wEye);
	}
	else if (key == 's') {
		camera.wEye = camera.wEye - 0.05 * normalize(camera.wLookat - camera.wEye);
	}
	else if (key == 'a') {
		camera.wEye = camera.wEye - 0.05 * normalize(camera.wLookat - camera.wEye);
	}
	else if (key == 'd') {
		camera.wEye = camera.wEye + 0.05 * normalize(camera.wLookat - camera.wEye);
	}
	else if (key == '4') {
		camera.wLookat = camera.wLookat + 0.05 * normalize(cross(camera.wVup, camera.wLookat - camera.wEye));
	}
	else if (key == '6') {
		camera.wLookat = camera.wLookat - 0.05 * normalize(cross(camera.wVup, camera.wLookat - camera.wEye));
	}
	else if (key == '8') {
		camera.wLookat = camera.wLookat + 0.05 * camera.wVup;
	}
	else if (key == '2') {
		camera.wLookat = camera.wLookat - 0.05 * camera.wVup;
	}
	glutPostRedisplay();*/
}

void onKeyboardUp(unsigned char key, int pX, int pY) { }

void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		float X = 4.0f * (2.0f * pX / windowWidth - 1);
		float Y = 4.0f * (1.0f - 2.0f * pY / windowHeight);
		scene.StartBall(vec3(X, Y, 0));
	}
	else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
		float X = 4.0f * (2.0f * pX / windowWidth - 1);
		float Y = 4.0f * (1.0f - 2.0f * pY / windowHeight);
		scene.PlaceWeight(vec2(X, Y));
	}
}

void onMouseMotion(int pX, int pY) { }

void onIdle() {
	static float tend = 0;
	const float dt = 0.1f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}
