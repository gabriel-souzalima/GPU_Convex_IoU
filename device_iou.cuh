#ifndef DEVICE_IOU_CUH
#define DEVICE_IOU_CUH

#include <cuda_runtime.h>
#include <math.h>

// basic definitions and structs

#define MAX_VERTS 64
#define EPS 1e-6

namespace DeviceIOU
{

    struct Point
    {
        double x, y;
        __host__ __device__ Point() : x(0), y(0) {}
        __host__ __device__ Point(double _x, double _y) : x(_x), y(_y) {}

        __host__ __device__ Point operator+(const Point &b) const { return Point(x + b.x, y + b.y); }
        __host__ __device__ Point operator-(const Point &b) const { return Point(x - b.x, y - b.y); }
        __host__ __device__ Point operator*(double s) const { return Point(x * s, y * s); }
        __host__ __device__ double cross(const Point &b) const { return x * b.y - y * b.x; }
        __host__ __device__ double dot(const Point &b) const { return x * b.x + y * b.y; }
    };

    struct Polygon
    {
        Point pts[MAX_VERTS];
        int count;

        __host__ __device__ Polygon() : count(0) {}

        __host__ __device__ void push_back(const Point &p)
        {
            if (count < MAX_VERTS)
            {
                pts[count++] = p;
            }
        }

        __host__ __device__ void clear() { count = 0; }
    };

    // geometric operations
    __device__ double area(const Polygon &poly)
    {
        double sArea = 0.0;
        if (poly.count > 2)
        {
            for (int i = 0; i < poly.count; ++i)
            {
                sArea += poly.pts[i].cross(poly.pts[(i + 1) % poly.count]);
            }
        }
        return fabs(sArea) * 0.5;
    }

    __device__ bool is_inside(const Polygon &C, const Point &p)
    {
        for (int i = 0; i < C.count; ++i)
        {
            Point p1 = C.pts[i];
            Point p2 = C.pts[(i + 1) % C.count];
            Point edge = p2 - p1;
            Point vec = p - p1;
            if (edge.cross(vec) < -EPS)
                return false;
        }
        return true;
    }

    __device__ bool get_intersection(Point p1, Point p2, Point p3, Point p4, Point &out)
    {
        Point r = p2 - p1;
        Point s = p4 - p3;
        double rxs = r.cross(s);
        Point qp = p3 - p1;
        double qpxr = qp.cross(r);

        if (fabs(rxs) < EPS)
            return false;

        double t = qp.cross(s) / rxs;
        double u = qpxr / rxs;

        if (t >= -EPS && t <= 1.0 + EPS && u >= -EPS && u <= 1.0 + EPS)
        {
            out = p1 + r * t;
            return true;
        }
        return false;
    }

    // polygon intersection logic
    __device__ void compute_intersection_polygon(const Polygon &C1, const Polygon &C2, Polygon &out)
    {
        out.clear();

        for (int i = 0; i < C1.count; ++i)
        {
            if (is_inside(C2, C1.pts[i]))
            {
                out.push_back(C1.pts[i]);
            }
        }

        for (int i = 0; i < C2.count; ++i)
        {
            if (is_inside(C1, C2.pts[i]))
            {
                out.push_back(C2.pts[i]);
            }
        }

        Point inter;
        for (int i = 0; i < C1.count; ++i)
        {
            for (int j = 0; j < C2.count; ++j)
            {
                if (get_intersection(C1.pts[i], C1.pts[(i + 1) % C1.count],
                                     C2.pts[j], C2.pts[(j + 1) % C2.count], inter))
                {
                    out.push_back(inter);
                }
            }
        }

        if (out.count > 0)
        {
            Point center(0, 0);
            for (int i = 0; i < out.count; ++i)
                center = center + out.pts[i];
            center = center * (1.0 / out.count);

            for (int i = 0; i < out.count - 1; ++i)
            {
                for (int j = 0; j < out.count - i - 1; ++j)
                {
                    double ang1 = atan2(out.pts[j].y - center.y, out.pts[j].x - center.x);
                    double ang2 = atan2(out.pts[j + 1].y - center.y, out.pts[j + 1].x - center.x);
                    if (ang1 > ang2)
                    {
                        Point temp = out.pts[j];
                        out.pts[j] = out.pts[j + 1];
                        out.pts[j + 1] = temp;
                    }
                }
            }
        }
    }
    // polygon tracing on ellipses
    __device__ void approximate_ellipse(double cx, double cy, double a, double b, double angle_rad, int num_points, Polygon &poly)
    {
        poly.clear();
        double cos_rot = cos(angle_rad);
        double sin_rot = sin(angle_rad);

        int n = (num_points < 3) ? 3 : (num_points > MAX_VERTS ? MAX_VERTS : num_points);

        for (int i = 0; i < n; ++i)
        {
            double phi = 2.0 * M_PI * i / n;
            double dx = a * cos(phi);
            double dy = b * sin(phi);

            double x = cx + dx * cos_rot - dy * sin_rot;
            double y = cy + dx * sin_rot + dy * cos_rot;

            poly.push_back(Point(x, y));
        }
    }
}

#endif
