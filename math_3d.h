#pragma once

#include <GL/glew.h>
#include <initializer_list>
#include <cmath>

template <typename T>
inline T sqr(T x) {return x*x;}

struct Vector2f {
        Vector2f() {}
        Vector2f(GLfloat x, GLfloat y) : x_(x), y_(y) {}

        GLfloat x_, y_;
};

struct Vector3i {
        GLint x_, y_, z_;
};

struct Vector3ub {
        GLubyte x_, y_, z_;
};



struct Vector3f {
        Vector3f() {}
        Vector3f(GLfloat x, GLfloat y, GLfloat z) : x_(x), y_(y), z_(z) {}

        Vector3f& operator+=(const Vector3f& a)
        {
                x_ += a.x_;
                y_ += a.y_;
                z_ += a.z_;
                return *this;
        }

        Vector3f& operator-=(const Vector3f& a)
        {
                x_ -= a.x_;
                y_ -= a.y_;
                z_ -= a.z_;
                return *this;
        }

        Vector3f& operator*=(float a)
        {
                x_ *= a;
                y_ *= a;
                z_ *= a;
                return *this;
        }

        Vector3f& operator/=(float a)
        {
                float inv_a = 1/a;
                x_ *= inv_a;
                y_ *= inv_a;
                z_ *= inv_a;
                return *this;
        }

        Vector3f operator-() const
        {
                return Vector3f(-x_, -y_, -z_);
        }

        // Vector3f operator*(float a) const
        // {
        //         return Vector3f(a*x_, a*y_, a*z_);
        // }

        GLfloat distance(const Vector3f& v)
        {
                return sqrt(sqr(x_-v.x_) + sqr(y_-v.y_) + sqr(z_-v.z_));
        }

        void normalize() {
                GLfloat d = sqrt(sqr(x_) + sqr(y_) + sqr(z_));
                *this /= d;
        }


        GLfloat x_, y_, z_;
};

template<typename T>
inline Vector3f operator*(T a, const Vector3f &v)
{
        return Vector3f(a*v.x_, a*v.y_, a*v.z_);
}



inline float dot (const Vector3f& a, const Vector3f& b)
{
        return a.x_*b.x_ + a.y_*b.y_ + a.z_*b.z_;
}

inline Vector3f cross(const Vector3f& a, const Vector3f& b)
{
        Vector3f c;
        c.x_ = a.y_*b.z_ - a.z_*b.y_;
        c.y_ = a.z_*b.x_ - a.x_*b.z_;
        c.z_ = a.x_*b.y_ - a.y_*b.x_;
        return c;
}

inline Vector3f normalize(Vector3f a)
{
        float d = sqrt(sqr(a.x_) + sqr(a.y_) + sqr(a.z_));
        a *= 1/d;
        return a;
}

inline Vector3f operator-(const Vector3f& a, const Vector3f& b)
{
        return Vector3f(a.x_-b.x_, a.y_-b.y_, a.z_-b.z_);
}

inline Vector3f operator+(const Vector3f& a, const Vector3f& b)
{
        return Vector3f(a.x_+b.x_, a.y_+b.y_, a.z_+b.z_);
}


// column major math - the number is the 1d array index where to store that matrix value
//  0  4  8 12
//  1  5  9 13
//  2  6 10 14
//  3  7 11 15
// 12, 13, 14 are the positions of the translation vector
// [c][r]

struct Matrix4f {
        Matrix4f(std::initializer_list<float> l)
        {
                std::copy(l.begin(), l.end(), &m[0][0]);
        }

        Matrix4f() : m{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1} {}


        inline Matrix4f operator*(const Matrix4f& Right) const
        {
                Matrix4f Ret;
                for (unsigned int j = 0 ; j < 4 ; j++) {
                        for (unsigned int i = 0 ; i < 4 ; i++) {
                                Ret.m[j][i] = m[0][i] * Right.m[j][0] +
                                              m[1][i] * Right.m[j][1] +
                                              m[2][i] * Right.m[j][2] +
                                              m[3][i] * Right.m[j][3];
                        }
                }
                return Ret;
        }

        inline Matrix4f& operator*=(const Matrix4f& Right)
        {
                Matrix4f ret;
                for (unsigned int j = 0 ; j < 4 ; j++) {
                        for (unsigned int i = 0 ; i < 4 ; i++) {
                                ret.m[j][i] = m[0][i] * Right.m[j][0] +
                                              m[1][i] * Right.m[j][1] +
                                              m[2][i] * Right.m[j][2] +
                                              m[3][i] * Right.m[j][3];
                        }
                }
                *this = ret;
                return *this;
        }

        inline GLfloat* glformat() {return (GLfloat*)m;}

    float m[4][4];
};


inline Matrix4f projection(const float a)
{
        float n = 0.1, f = 100.0, l = -1, r = 1, t = 1, b = -1;
        t = tan(M_PI_2/2) * n;
        r = t * a;
        l = -r;
        b = -t;
        Matrix4f translate_matrix;
	translate_matrix.m[0][0] = 2*n/(r-l); translate_matrix.m[0][1] = 0; translate_matrix.m[0][2] = 0; translate_matrix.m[0][3] = 0;
	translate_matrix.m[1][0] = 0; translate_matrix.m[1][1] = 2*n/(t-b); translate_matrix.m[1][2] = 0; translate_matrix.m[1][3] = 0;
	translate_matrix.m[2][0] = (r+l)/(r-l); translate_matrix.m[2][1] = (t+b)/(t-b); translate_matrix.m[2][2] = -(f+n)/(f-n); translate_matrix.m[2][3] = -1;
	translate_matrix.m[3][0] = 0; translate_matrix.m[3][1] = 0; translate_matrix.m[3][2] = -2*f*n/(f-n); translate_matrix.m[3][3] = 0;
        return translate_matrix;
}

inline Matrix4f translation_matrix(GLfloat x, GLfloat y, GLfloat z)
{
        return {1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1};
}

inline Matrix4f translation_matrix(const Vector3f& v)
{
        return translation_matrix(v.x_, v.y_, v.z_);
}

inline Matrix4f shear_matrix_x(GLfloat shear)
{
        return {1,0,0,0, shear,1,0,0, 0,0,1,0, 0,0,0,1};
}

inline Matrix4f rotation_matrix_xy(GLfloat angle)
{
        GLfloat cos_angle = cos(angle);
        GLfloat sin_angle = sin(angle);
        return {cos_angle,sin_angle,0,0, -sin_angle,cos_angle,0,0, 0,0,1,0, 0,0,0,1};
}

inline Matrix4f rotation_matrix_yz(GLfloat angle)
{
        Matrix4f rotate_matrix;
	rotate_matrix.m[0][0] = 1; rotate_matrix.m[0][1] = 0; rotate_matrix.m[0][2] = 0; rotate_matrix.m[0][3] = 0;
	rotate_matrix.m[1][0] = 0; rotate_matrix.m[1][1] = cos(angle); rotate_matrix.m[1][2] = -sin(angle); rotate_matrix.m[1][3] = 0;
	rotate_matrix.m[2][0] = 0; rotate_matrix.m[2][1] = sin(angle); rotate_matrix.m[2][2] = cos(angle); rotate_matrix.m[2][3] = 0;
	rotate_matrix.m[3][0] = 0; rotate_matrix.m[3][1] = 0; rotate_matrix.m[3][2] = 0; rotate_matrix.m[3][3] = 1;
        return rotate_matrix;
}

inline Matrix4f rotation_matrix_zx(GLfloat angle)
{
        Matrix4f rotate_matrix;
	rotate_matrix.m[0][0] = cos(angle); rotate_matrix.m[0][1] = 0; rotate_matrix.m[0][2] = -sin(angle); rotate_matrix.m[0][3] = 0;
	rotate_matrix.m[1][0] = 0; rotate_matrix.m[1][1] = 1; rotate_matrix.m[1][2] = 0; rotate_matrix.m[1][3] = 0;
	rotate_matrix.m[2][0] = sin(angle); rotate_matrix.m[2][1] = 0; rotate_matrix.m[2][2] = cos(angle); rotate_matrix.m[2][3] = 0;
	rotate_matrix.m[3][0] = 0; rotate_matrix.m[3][1] = 0; rotate_matrix.m[3][2] = 0; rotate_matrix.m[3][3] = 1;
        return rotate_matrix;
}

inline Matrix4f scale_matrix(GLfloat s)
{
        return {s,0,0,0, 0,s,0,0, 0,0,s,0, 0,0,0,1};
}

inline Matrix4f scale_matrix(GLfloat sx, GLfloat sy, GLfloat sz)
{
        return {sx,0,0,0, 0,sy,0,0, 0,0,sz,0, 0,0,0,1};
}

inline Matrix4f identity_matrix()
{
        return {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
}

