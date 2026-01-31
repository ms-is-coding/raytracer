// scene_types.h
#pragma once

#include <CL/cl.h>

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

// GPU-optimized BVH node (64 bytes, cache-line aligned)
// Uses child indices packed with leaf flag for branchless traversal
typedef struct {
  cl_float4 bbox_min;    // xyz = min, w = unused
  cl_float4 bbox_max;    // xyz = max, w = unused
  cl_int child[2];       // child[0] = left, child[1] = right (negative = leaf: ~idx = first prim)
  cl_int prim_count[2];  // number of primitives in each child (0 for internal)
  cl_int parent;         // parent index for stackless traversal (optional)
  cl_int axis;           // split axis (0=X, 1=Y, 2=Z) for child ordering
  cl_int pad[2];         // padding to 64 bytes
} t_bvh_node_gpu;

// SoA primitive data for coalesced GPU access
typedef struct {
  cl_float4 *pos;        // positions (xyz) + type (w as int bits)
  cl_float4 *param0;     // first param vec (sphere: radius,0,0,mat_id; box: half_size,mat_id; etc)
  cl_float4 *param1;     // second param vec (plane: normal; quadric: coeffs[0-3]; etc)
  cl_float4 *param2;     // third param vec (quadric: coeffs[4-7]; etc)
  cl_float4 *param3;     // fourth param vec (quadric: coeffs[8-9],0,0; etc)
  cl_float4 *mat_color;  // material color
  cl_float4 *mat_props;  // reflection, transparency, ior, roughness
  cl_int count;
} t_primitives_soa;

#ifndef __OPENCL_VERSION__
// GPU buffer structure for optimized BVH (CPU-side only)
typedef struct t_gpu_buffers {
  cl_mem bvh_gpu;        // GPU-optimized BVH nodes
  cl_mem prim_pos;       // SoA: positions + type
  cl_mem prim_param0;    // SoA: param0
  cl_mem prim_param1;    // SoA: param1
  cl_mem prim_param2;    // SoA: param2
  cl_mem prim_param3;    // SoA: param3
  cl_mem prim_mat_color; // SoA: material color
  cl_mem prim_mat_props; // SoA: material properties
} t_gpu_buffers;
#endif
