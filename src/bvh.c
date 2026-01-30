#include "minirt.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <immintrin.h>

// Compute AABB for an object
static t_aabb compute_object_aabb(const t_object *obj) {
  t_aabb box;
  float px = obj->pos.s[0], py = obj->pos.s[1], pz = obj->pos.s[2];

  switch (obj->type) {
    case TYPE_SPHERE: {
      float r = obj->sphere.radius;
      box.min = (cl_float4){{px - r, py - r, pz - r, 0}};
      box.max = (cl_float4){{px + r, py + r, pz + r, 0}};
      break;
    }
    case TYPE_BOX: {
      float hx = obj->box.half_size.s[0];
      float hy = obj->box.half_size.s[1];
      float hz = obj->box.half_size.s[2];
      box.min = (cl_float4){{px - hx, py - hy, pz - hz, 0}};
      box.max = (cl_float4){{px + hx, py + hy, pz + hz, 0}};
      break;
    }
    case TYPE_TORUS: {
      float R = obj->torus.major;
      float r = obj->torus.minor;
      float extent = R + r;
      box.min = (cl_float4){{px - extent, py - r, pz - extent, 0}};
      box.max = (cl_float4){{px + extent, py + r, pz + extent, 0}};
      break;
    }
    case TYPE_MOBIUS: {
      float R = obj->mobius.radius;
      float w = obj->mobius.width;
      float extent = R + w + 0.1f;
      box.min = (cl_float4){{px - extent, py - w, pz - extent, 0}};
      box.max = (cl_float4){{px + extent, py + w, pz + extent, 0}};
      break;
    }
    case TYPE_QUADRIC: {
      // Conservative bound for cylinders/cones - use large box
      float extent = 10.0f;
      box.min = (cl_float4){{px - extent, py - extent, pz - extent, 0}};
      box.max = (cl_float4){{px + extent, py + extent, pz + extent, 0}};
      break;
    }
    case TYPE_PLANE:
    default: {
      // Planes are infinite - use huge box
      float inf = 1e6f;
      box.min = (cl_float4){{-inf, -inf, -inf, 0}};
      box.max = (cl_float4){{inf, inf, inf, 0}};
      break;
    }
  }
  return box;
}

// Merge two AABBs
static t_aabb aabb_merge(t_aabb a, t_aabb b) {
  t_aabb result;
  __m128 a_min = _mm_loadu_ps(a.min.s);
  __m128 b_min = _mm_loadu_ps(b.min.s);
  __m128 a_max = _mm_loadu_ps(a.max.s);
  __m128 b_max = _mm_loadu_ps(b.max.s);
  _mm_storeu_ps(result.min.s, _mm_min_ps(a_min, b_min));
  _mm_storeu_ps(result.max.s, _mm_max_ps(a_max, b_max));
  return result;
}

// Surface area of AABB (for SAH)
static float aabb_surface_area(const t_aabb *box) {
  float dx = box->max.s[0] - box->min.s[0];
  float dy = box->max.s[1] - box->min.s[1];
  float dz = box->max.s[2] - box->min.s[2];
  return 2.0f * (dx*dy + dy*dz + dz*dx);
}

typedef struct {
  t_bvh_node *nodes;
  int node_count;
  int node_cap;
  int *obj_indices;  // Reordered object indices
  t_aabb *obj_aabbs; // Precomputed AABBs
  int total_objects; // For progress tracking
  void (*progress_cb)(int current, int total, void *ctx);
  void *progress_ctx;
} bvh_builder;

static int alloc_node(bvh_builder *b) {
  if (b->node_count >= b->node_cap) {
    b->node_cap *= 2;
    b->nodes = realloc(b->nodes, b->node_cap * sizeof(t_bvh_node));
  }
  int idx = b->node_count++;
  // Report progress (max nodes ≈ 2 * obj_count - 1)
  if (b->progress_cb && (idx % 50 == 0 || idx < 10)) {
    b->progress_cb(idx, b->total_objects * 2, b->progress_ctx);
  }
  return idx;
}

// Sort context for qsort
static __thread const t_aabb *g_sort_aabbs;
static __thread int g_sort_axis;

static int compare_by_center(const void *a, const void *b) {
  int ia = *(const int *)a;
  int ib = *(const int *)b;
  float ca = (g_sort_aabbs[ia].min.s[g_sort_axis] + g_sort_aabbs[ia].max.s[g_sort_axis]);
  float cb = (g_sort_aabbs[ib].min.s[g_sort_axis] + g_sort_aabbs[ib].max.s[g_sort_axis]);
  return (ca > cb) - (ca < cb);
}

// Recursive BVH build with SAH (optimized)
static int build_recursive(bvh_builder *b, int start, int end) {
  int node_idx = alloc_node(b);
  t_bvh_node *node = &b->nodes[node_idx];
  int count = end - start;

  // Compute bounds for this node
  t_aabb bounds = b->obj_aabbs[b->obj_indices[start]];
  for (int i = start + 1; i < end; i++) {
    bounds = aabb_merge(bounds, b->obj_aabbs[b->obj_indices[i]]);
  }
  node->bounds = bounds;

  // Leaf node if few objects
  if (count <= 2) {
    node->left = -1;
    node->right = -1;
    node->obj_start = start;
    node->obj_count = count;
    return node_idx;
  }

  // Allocate prefix bounds array (suffix computed on-the-fly from right)
  t_aabb *prefix = malloc(count * sizeof(t_aabb));
  int *temp_indices = malloc(count * sizeof(int));
  memcpy(temp_indices, &b->obj_indices[start], count * sizeof(int));

  float best_cost = FLT_MAX;
  int best_split = count / 2;
  int *best_order = malloc(count * sizeof(int));
  memcpy(best_order, temp_indices, count * sizeof(int));

  float parent_area = aabb_surface_area(&bounds);
  float inv_parent = 1.0f / parent_area;

  // Setup sort context
  g_sort_aabbs = b->obj_aabbs;

  for (int axis = 0; axis < 3; axis++) {
    // Sort by center using qsort - O(n log n)
    g_sort_axis = axis;
    qsort(temp_indices, count, sizeof(int), compare_by_center);

    // Build prefix bounds: prefix[i] = bounds of objects [0..i]
    prefix[0] = b->obj_aabbs[temp_indices[0]];
    for (int i = 1; i < count; i++) {
      prefix[i] = aabb_merge(prefix[i - 1], b->obj_aabbs[temp_indices[i]]);
    }

    // Sweep from right, computing suffix bounds and SAH cost - O(n)
    t_aabb suffix = b->obj_aabbs[temp_indices[count - 1]];
    for (int i = count - 2; i >= 0; i--) {
      // Split after index i: left = [0..i], right = [i+1..count-1]
      int left_count = i + 1;
      int right_count = count - left_count;

      float left_area = aabb_surface_area(&prefix[i]);
      float right_area = aabb_surface_area(&suffix);
      float cost = 1.0f + (left_area * left_count + right_area * right_count) * inv_parent;

      if (cost < best_cost) {
        best_cost = cost;
        best_split = left_count;
        memcpy(best_order, temp_indices, count * sizeof(int));
      }
      suffix = aabb_merge(suffix, b->obj_aabbs[temp_indices[i]]);
    }
  }

  // Apply best ordering
  memcpy(&b->obj_indices[start], best_order, count * sizeof(int));
  free(prefix);
  free(temp_indices);
  free(best_order);

  // Build children
  int left_idx = build_recursive(b, start, start + best_split);
  int right_idx = build_recursive(b, start + best_split, end);

  // Update node (pointer may have moved due to realloc)
  b->nodes[node_idx].bounds = bounds;
  b->nodes[node_idx].left = left_idx;
  b->nodes[node_idx].right = right_idx;
  b->nodes[node_idx].obj_start = 0;
  b->nodes[node_idx].obj_count = 0;

  return node_idx;
}

void build_bvh_with_progress(t_scene *scene,
                             void (*progress_cb)(int current, int total, void *ctx),
                             void *ctx) {
  if (scene->obj_count == 0) {
    scene->bvh_nodes = NULL;
    scene->bvh_node_count = 0;
    scene->obj_order = NULL;
    return;
  }

  bvh_builder b = {0};
  b.node_cap = scene->obj_count * 2;
  b.nodes = malloc(b.node_cap * sizeof(t_bvh_node));
  b.obj_indices = malloc(scene->obj_count * sizeof(int));
  b.obj_aabbs = malloc(scene->obj_count * sizeof(t_aabb));
  b.total_objects = scene->obj_count;
  b.progress_cb = progress_cb;
  b.progress_ctx = ctx;

  // Initialize indices and compute AABBs
  for (int i = 0; i < scene->obj_count; i++) {
    b.obj_indices[i] = i;
    b.obj_aabbs[i] = compute_object_aabb(&scene->objects[i]);
  }

  // Build tree
  build_recursive(&b, 0, scene->obj_count);

  // Final progress update
  if (progress_cb) {
    progress_cb(b.node_count, b.node_count, ctx);
  }

  // Reorder objects according to BVH order
  t_object *reordered = malloc(scene->obj_count * sizeof(t_object));
  for (int i = 0; i < scene->obj_count; i++) {
    reordered[i] = scene->objects[b.obj_indices[i]];
  }
  free(scene->objects);
  scene->objects = reordered;

  scene->bvh_nodes = b.nodes;
  scene->bvh_node_count = b.node_count;
  scene->obj_order = b.obj_indices;

  free(b.obj_aabbs);
}

void build_bvh(t_scene *scene) {
  build_bvh_with_progress(scene, NULL, NULL);
}

void free_bvh(t_scene *scene) {
  free(scene->bvh_nodes);
  free(scene->obj_order);
  scene->bvh_nodes = NULL;
  scene->obj_order = NULL;
  scene->bvh_node_count = 0;
}
