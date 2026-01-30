// scene_types.h
#pragma once

#ifdef __OPENCL_VERSION__
#define HOST_DEVICE
typedef float4 cl_float4;
typedef float cl_float;
typedef int cl_int;

#else
#include <CL/cl.h>
#define HOST_DEVICE
#endif

typedef struct {
  cl_float4 pos;
  cl_float4 dir;
  cl_float fov;
  cl_float padding[3];
} t_camera;

typedef struct {
  cl_float reflection;
  cl_float transparency;
  cl_float ior;
  cl_float roughness;
  cl_float metallic;
  cl_float4 color;
  cl_float4 albedo;
  cl_float4 absorption;
} t_material;

typedef enum : cl_int {
  TYPE_SPHERE,
  TYPE_PLANE,
  TYPE_QUADRIC,
  TYPE_BOX,
  TYPE_TORUS,
  TYPE_MOBIUS,
} t_object_type;

typedef struct { cl_float radius; } t_sphere;
typedef struct { cl_float4 normal; } t_plane;
typedef struct { cl_float coeffs[10]; } t_quadric;
typedef struct { cl_float4 half_size; } t_box;
typedef struct { cl_float major, minor; } t_torus;
typedef struct { cl_float radius, width; } t_mobius;

typedef struct {
  cl_int type; // 0: Sphere, 1: Plane
  cl_int material_id;
  cl_float4 pos;
  cl_float4 rot;
  t_material mat;
  union {
    t_sphere sphere;
    t_plane plane;
    t_quadric quadric;
    t_box box;
    t_torus torus;
    t_mobius mobius;
  };
} t_object;

typedef enum : cl_int {
  LIGHT_POINT,
  LIGHT_DIRECTIONAL,
  LIGHT_SPOT,
} t_light_type;

typedef struct {
  cl_float4 pos;
  cl_float4 dir;      // for directional/spot
  cl_float4 color;    // RGB color
  cl_float intensity;
  cl_float angle;     // spot angle (radians)
  cl_int type;
  cl_float padding;
} t_light;

typedef struct {
  cl_int max_bounces;
  cl_float gamma;
  cl_float exposure;
  cl_float epsilon;
} t_render;

// Axis-Aligned Bounding Box
typedef struct {
  cl_float4 min;  // xyz = min corner, w unused
  cl_float4 max;  // xyz = max corner, w unused
} t_aabb;

// BVH Node (flat array layout for GPU)
// If left == -1, it's a leaf node with obj_idx and obj_count
// Otherwise it's an internal node with left/right children
typedef struct {
  t_aabb bounds;
  cl_int left;       // left child index, or -1 for leaf
  cl_int right;      // right child index
  cl_int obj_start;  // first object index (leaf only)
  cl_int obj_count;  // number of objects (leaf only)
} t_bvh_node;
