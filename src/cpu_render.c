#define _GNU_SOURCE
#include "minirt.h"
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Thread pool
typedef struct {
  pthread_t *threads;
  int num_threads;
  volatile int shutdown;

  // Work queue
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  pthread_cond_t done_cond;

  // Current render job
  t_data *data;
  volatile int next_row;
  volatile int rows_done;
  int total_rows;
} thread_pool_t;

static thread_pool_t g_pool = {0};

// AVX2 vector types (8 floats)
typedef __m256 v8f;
typedef __m256i v8i;

#define V8F_SET1(x) _mm256_set1_ps(x)
#define V8F_ZERO() _mm256_setzero_ps()
#define V8F_ADD(a, b) _mm256_add_ps(a, b)
#define V8F_SUB(a, b) _mm256_sub_ps(a, b)
#define V8F_MUL(a, b) _mm256_mul_ps(a, b)
#define V8F_DIV(a, b) _mm256_div_ps(a, b)
#define V8F_MIN(a, b) _mm256_min_ps(a, b)
#define V8F_MAX(a, b) _mm256_max_ps(a, b)
#define V8F_SQRT(a) _mm256_sqrt_ps(a)
#define V8F_AND(a, b) _mm256_and_ps(a, b)
#define V8F_OR(a, b) _mm256_or_ps(a, b)
#define V8F_XOR(a, b) _mm256_xor_ps(a, b)
#define V8F_CMP_GT(a, b) _mm256_cmp_ps(a, b, _CMP_GT_OQ)
#define V8F_CMP_LT(a, b) _mm256_cmp_ps(a, b, _CMP_LT_OQ)
#define V8F_BLEND(a, b, m) _mm256_blendv_ps(a, b, m)

// 8-wide ray structure
typedef struct {
  v8f ox, oy, oz;  // origin
  v8f dx, dy, dz;  // direction
  v8f inv_dx, inv_dy, inv_dz;  // inverse direction
} ray8_t;

// 8-wide hit result
typedef struct {
  v8f t;
  v8i obj_idx;
  v8f nx, ny, nz;  // normal
} hit8_t;

// Scalar helpers
static inline float dot3(float ax, float ay, float az, float bx, float by, float bz) {
  return ax*bx + ay*by + az*bz;
}

static inline void normalize3(float *x, float *y, float *z) {
  float len = sqrtf(*x * *x + *y * *y + *z * *z);
  if (len > 1e-6f) { *x /= len; *y /= len; *z /= len; }
}

// AVX2 dot product (for future 8-wide ray tracing)
__attribute__((unused))
static inline v8f dot8(v8f ax, v8f ay, v8f az, v8f bx, v8f by, v8f bz) {
  return V8F_ADD(V8F_ADD(V8F_MUL(ax, bx), V8F_MUL(ay, by)), V8F_MUL(az, bz));
}

// AVX2 AABB intersection (slab method) - for future 8-wide ray tracing
__attribute__((unused))
static inline v8f intersect_aabb8(const ray8_t *r, const t_aabb *box) {
  v8f bmin_x = V8F_SET1(box->min.s[0]);
  v8f bmin_y = V8F_SET1(box->min.s[1]);
  v8f bmin_z = V8F_SET1(box->min.s[2]);
  v8f bmax_x = V8F_SET1(box->max.s[0]);
  v8f bmax_y = V8F_SET1(box->max.s[1]);
  v8f bmax_z = V8F_SET1(box->max.s[2]);

  v8f t1x = V8F_MUL(V8F_SUB(bmin_x, r->ox), r->inv_dx);
  v8f t2x = V8F_MUL(V8F_SUB(bmax_x, r->ox), r->inv_dx);
  v8f t1y = V8F_MUL(V8F_SUB(bmin_y, r->oy), r->inv_dy);
  v8f t2y = V8F_MUL(V8F_SUB(bmax_y, r->oy), r->inv_dy);
  v8f t1z = V8F_MUL(V8F_SUB(bmin_z, r->oz), r->inv_dz);
  v8f t2z = V8F_MUL(V8F_SUB(bmax_z, r->oz), r->inv_dz);

  v8f tmin_x = V8F_MIN(t1x, t2x);
  v8f tmax_x = V8F_MAX(t1x, t2x);
  v8f tmin_y = V8F_MIN(t1y, t2y);
  v8f tmax_y = V8F_MAX(t1y, t2y);
  v8f tmin_z = V8F_MIN(t1z, t2z);
  v8f tmax_z = V8F_MAX(t1z, t2z);

  v8f tmin = V8F_MAX(V8F_MAX(tmin_x, tmin_y), tmin_z);
  v8f tmax = V8F_MIN(V8F_MIN(tmax_x, tmax_y), tmax_z);

  // Return tmin if hit, else infinity
  v8f hit_mask = V8F_CMP_GT(tmax, V8F_MAX(tmin, V8F_ZERO()));
  return V8F_BLEND(V8F_SET1(1e30f), tmin, hit_mask);
}

// Scalar sphere intersection
static float intersect_sphere_scalar(float ox, float oy, float oz,
                                     float dx, float dy, float dz,
                                     const t_object *s) {
  float cx = s->pos.s[0], cy = s->pos.s[1], cz = s->pos.s[2];
  float ocx = ox - cx, ocy = oy - cy, ocz = oz - cz;
  float r = s->sphere.radius;

  float b = dot3(ocx, ocy, ocz, dx, dy, dz);
  float c = dot3(ocx, ocy, ocz, ocx, ocy, ocz) - r*r;
  float h = b*b - c;
  if (h < 0.0f) return -1.0f;

  h = sqrtf(h);
  float t = -b - h;
  if (t > 0.001f) return t;
  t = -b + h;
  return (t > 0.001f) ? t : -1.0f;
}

// Scalar plane intersection
static float intersect_plane_scalar(float ox, float oy, float oz,
                                    float dx, float dy, float dz,
                                    const t_object *p) {
  float nx = p->plane.normal.s[0];
  float ny = p->plane.normal.s[1];
  float nz = p->plane.normal.s[2];
  float denom = dot3(nx, ny, nz, dx, dy, dz);

  if (fabsf(denom) > 1e-6f) {
    float px = p->pos.s[0] - ox;
    float py = p->pos.s[1] - oy;
    float pz = p->pos.s[2] - oz;
    float t = dot3(px, py, pz, nx, ny, nz) / denom;
    return (t > 0.001f) ? t : -1.0f;
  }
  return -1.0f;
}

// Scalar box intersection
static float intersect_box_scalar(float ox, float oy, float oz,
                                  float dx, float dy, float dz,
                                  const t_object *b) {
  float px = ox - b->pos.s[0];
  float py = oy - b->pos.s[1];
  float pz = oz - b->pos.s[2];
  float hx = b->box.half_size.s[0];
  float hy = b->box.half_size.s[1];
  float hz = b->box.half_size.s[2];

  float inv_dx = 1.0f / dx, inv_dy = 1.0f / dy, inv_dz = 1.0f / dz;

  float t1x = (-hx - px) * inv_dx, t2x = (hx - px) * inv_dx;
  float t1y = (-hy - py) * inv_dy, t2y = (hy - py) * inv_dy;
  float t1z = (-hz - pz) * inv_dz, t2z = (hz - pz) * inv_dz;

  float tmin = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
  float tmax = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));

  if (tmax < 0 || tmin > tmax) return -1.0f;
  return (tmin > 0.001f) ? tmin : ((tmax > 0.001f) ? tmax : -1.0f);
}

// Scalar object intersection
static float intersect_object_scalar(float ox, float oy, float oz,
                                     float dx, float dy, float dz,
                                     const t_object *obj,
                                     float *nx, float *ny, float *nz) {
  float t = -1.0f;

  switch (obj->type) {
    case TYPE_SPHERE:
      t = intersect_sphere_scalar(ox, oy, oz, dx, dy, dz, obj);
      if (t > 0.0f) {
        float hx = ox + dx*t - obj->pos.s[0];
        float hy = oy + dy*t - obj->pos.s[1];
        float hz = oz + dz*t - obj->pos.s[2];
        normalize3(&hx, &hy, &hz);
        *nx = hx; *ny = hy; *nz = hz;
      }
      break;

    case TYPE_PLANE:
      t = intersect_plane_scalar(ox, oy, oz, dx, dy, dz, obj);
      if (t > 0.0f) {
        *nx = obj->plane.normal.s[0];
        *ny = obj->plane.normal.s[1];
        *nz = obj->plane.normal.s[2];
        normalize3(nx, ny, nz);
      }
      break;

    case TYPE_BOX:
      t = intersect_box_scalar(ox, oy, oz, dx, dy, dz, obj);
      if (t > 0.0f) {
        float hx = ox + dx*t - obj->pos.s[0];
        float hy = oy + dy*t - obj->pos.s[1];
        float hz = oz + dz*t - obj->pos.s[2];
        float eps = 0.001f;
        float hsx = obj->box.half_size.s[0];
        float hsy = obj->box.half_size.s[1];
        float hsz = obj->box.half_size.s[2];
        *nx = *ny = *nz = 0.0f;
        if (fabsf(fabsf(hx) - hsx) < eps) *nx = (hx > 0) ? 1.0f : -1.0f;
        else if (fabsf(fabsf(hy) - hsy) < eps) *ny = (hy > 0) ? 1.0f : -1.0f;
        else if (fabsf(fabsf(hz) - hsz) < eps) *nz = (hz > 0) ? 1.0f : -1.0f;
      }
      break;

    default:
      // Skip complex types for CPU (torus, mobius, quadric)
      break;
  }

  return t;
}

// Traverse BVH with single ray (scalar fallback)
static float trace_bvh_scalar(const t_scene *scene,
                              float ox, float oy, float oz,
                              float dx, float dy, float dz,
                              int *hit_idx, float *hit_nx, float *hit_ny, float *hit_nz) {
  if (!scene->bvh_nodes || scene->bvh_node_count == 0) {
    // Linear fallback
    float closest_t = 1e30f;
    *hit_idx = -1;
    for (int i = 0; i < scene->obj_count; i++) {
      float nx, ny, nz;
      float t = intersect_object_scalar(ox, oy, oz, dx, dy, dz, &scene->objects[i], &nx, &ny, &nz);
      if (t > 0.001f && t < closest_t) {
        closest_t = t;
        *hit_idx = i;
        *hit_nx = nx; *hit_ny = ny; *hit_nz = nz;
      }
    }
    return closest_t;
  }

  // BVH traversal stack
  int stack[64];
  int stack_ptr = 0;
  stack[stack_ptr++] = 0;  // Start with root

  float inv_dx = 1.0f / dx, inv_dy = 1.0f / dy, inv_dz = 1.0f / dz;
  float closest_t = 1e30f;
  *hit_idx = -1;

  while (stack_ptr > 0) {
    int node_idx = stack[--stack_ptr];
    const t_bvh_node *node = &scene->bvh_nodes[node_idx];

    // Test AABB
    float bmin_x = node->bounds.min.s[0], bmin_y = node->bounds.min.s[1], bmin_z = node->bounds.min.s[2];
    float bmax_x = node->bounds.max.s[0], bmax_y = node->bounds.max.s[1], bmax_z = node->bounds.max.s[2];

    float t1x = (bmin_x - ox) * inv_dx, t2x = (bmax_x - ox) * inv_dx;
    float t1y = (bmin_y - oy) * inv_dy, t2y = (bmax_y - oy) * inv_dy;
    float t1z = (bmin_z - oz) * inv_dz, t2z = (bmax_z - oz) * inv_dz;

    float tmin = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
    float tmax = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));

    if (tmax < 0 || tmin > tmax || tmin > closest_t) continue;

    if (node->left == -1) {
      // Leaf node - test objects
      for (int i = 0; i < node->obj_count; i++) {
        int obj_i = node->obj_start + i;
        float nx, ny, nz;
        float t = intersect_object_scalar(ox, oy, oz, dx, dy, dz, &scene->objects[obj_i], &nx, &ny, &nz);
        if (t > 0.001f && t < closest_t) {
          closest_t = t;
          *hit_idx = obj_i;
          *hit_nx = nx; *hit_ny = ny; *hit_nz = nz;
        }
      }
    } else {
      // Internal node - push children
      stack[stack_ptr++] = node->right;
      stack[stack_ptr++] = node->left;
    }
  }

  return closest_t;
}

// Shade a single hit point
static void shade_pixel(const t_scene *scene,
                        float ox, float oy, float oz,
                        float dx, float dy, float dz,
                        float *r, float *g, float *b) {
  *r = *g = *b = 0.0f;
  float mask_r = 1.0f, mask_g = 1.0f, mask_b = 1.0f;
  float eps = scene->render.epsilon;

  for (int bounce = 0; bounce < scene->render.max_bounces; bounce++) {
    int hit_idx;
    float nx, ny, nz;
    float t = trace_bvh_scalar(scene, ox, oy, oz, dx, dy, dz, &hit_idx, &nx, &ny, &nz);

    if (hit_idx == -1) {
      // Background
      *r += scene->background_color.s[0] * mask_r;
      *g += scene->background_color.s[1] * mask_g;
      *b += scene->background_color.s[2] * mask_b;
      break;
    }

    const t_object *obj = &scene->objects[hit_idx];
    float hx = ox + dx*t, hy = oy + dy*t, hz = oz + dz*t;

    // Flip normal if inside
    float ndotd = dot3(nx, ny, nz, dx, dy, dz);
    if (ndotd > 0) { nx = -nx; ny = -ny; nz = -nz; }

    // Lighting
    float diff_r = 0, diff_g = 0, diff_b = 0;
    for (int li = 0; li < scene->light_count; li++) {
      const t_light *light = &scene->lights[li];
      float lx, ly, lz, l_dist;

      if (light->type == LIGHT_DIRECTIONAL) {
        lx = -light->dir.s[0]; ly = -light->dir.s[1]; lz = -light->dir.s[2];
        l_dist = 1e30f;
      } else {
        lx = light->pos.s[0] - hx;
        ly = light->pos.s[1] - hy;
        lz = light->pos.s[2] - hz;
        l_dist = sqrtf(lx*lx + ly*ly + lz*lz);
        lx /= l_dist; ly /= l_dist; lz /= l_dist;
      }

      // Shadow test
      int shadow_hit;
      float snx, sny, snz;
      float shadow_t = trace_bvh_scalar(scene,
                                        hx + nx*eps, hy + ny*eps, hz + nz*eps,
                                        lx, ly, lz,
                                        &shadow_hit, &snx, &sny, &snz);

      if (shadow_t > l_dist || shadow_hit == -1) {
        float ndotl = fmaxf(0.0f, dot3(nx, ny, nz, lx, ly, lz));
        float atten = (light->type == LIGHT_DIRECTIONAL) ? 1.0f : 1.0f / (1.0f + 0.01f * l_dist * l_dist);

        diff_r += obj->mat.color.s[0] * light->color.s[0] * ndotl * light->intensity * atten;
        diff_g += obj->mat.color.s[1] * light->color.s[1] * ndotl * light->intensity * atten;
        diff_b += obj->mat.color.s[2] * light->color.s[2] * ndotl * light->intensity * atten;
      }
    }

    // Ambient
    float amb_r = scene->ambient_color.s[0] * obj->mat.color.s[0];
    float amb_g = scene->ambient_color.s[1] * obj->mat.color.s[1];
    float amb_b = scene->ambient_color.s[2] * obj->mat.color.s[2];

    float local_r = diff_r + amb_r;
    float local_g = diff_g + amb_g;
    float local_b = diff_b + amb_b;

    if (obj->mat.reflection > 0.0f) {
      *r += local_r * mask_r * (1.0f - obj->mat.reflection);
      *g += local_g * mask_g * (1.0f - obj->mat.reflection);
      *b += local_b * mask_b * (1.0f - obj->mat.reflection);

      mask_r *= obj->mat.reflection;
      mask_g *= obj->mat.reflection;
      mask_b *= obj->mat.reflection;

      // Reflect
      float rdotn = dot3(dx, dy, dz, nx, ny, nz);
      dx = dx - 2.0f * rdotn * nx;
      dy = dy - 2.0f * rdotn * ny;
      dz = dz - 2.0f * rdotn * nz;
      ox = hx + nx * eps;
      oy = hy + ny * eps;
      oz = hz + nz * eps;
    } else {
      *r += local_r * mask_r;
      *g += local_g * mask_g;
      *b += local_b * mask_b;
      break;
    }
  }

  // Tone mapping and gamma
  float exposure = scene->render.exposure;
  float gamma = 1.0f / scene->render.gamma;
  *r = powf(1.0f - expf(-(*r) * exposure), gamma);
  *g = powf(1.0f - expf(-(*g) * exposure), gamma);
  *b = powf(1.0f - expf(-(*b) * exposure), gamma);
}

// Render a row of pixels
static void render_row(t_data *data, int y) {
  t_scene *scene = &data->scene;
  int width = scene->width;
  int height = scene->height;
  float aspect = (float)width / height;
  float scale = tanf(scene->camera.fov * 0.5f * 3.14159265f / 180.0f);

  float cam_x = scene->camera.pos.s[0];
  float cam_y = scene->camera.pos.s[1];
  float cam_z = scene->camera.pos.s[2];

  float fwd_x = scene->camera.dir.s[0];
  float fwd_y = scene->camera.dir.s[1];
  float fwd_z = scene->camera.dir.s[2];
  normalize3(&fwd_x, &fwd_y, &fwd_z);

  // Right = forward x (0,1,0)
  float right_x = fwd_z, right_y = 0, right_z = -fwd_x;
  float len = sqrtf(right_x*right_x + right_z*right_z);
  if (len > 0.001f) { right_x /= len; right_z /= len; }
  else { right_x = 1; right_z = 0; }

  // Up = right x forward
  float up_x = right_y * fwd_z - right_z * fwd_y;
  float up_y = right_z * fwd_x - right_x * fwd_z;
  float up_z = right_x * fwd_y - right_y * fwd_x;

  float py = (1.0f - 2.0f * (y + 0.5f) / height) * scale;

  for (int x = 0; x < width; x++) {
    float px = (2.0f * (x + 0.5f) / width - 1.0f) * aspect * scale;

    float dx = px * right_x + py * up_x + fwd_x;
    float dy = px * right_y + py * up_y + fwd_y;
    float dz = px * right_z + py * up_z + fwd_z;
    normalize3(&dx, &dy, &dz);

    float r, g, b;
    shade_pixel(scene, cam_x, cam_y, cam_z, dx, dy, dz, &r, &g, &b);

    int idx = (y * width + x) * 3;
    data->cl.host_image_buffer[idx] = r;
    data->cl.host_image_buffer[idx + 1] = g;
    data->cl.host_image_buffer[idx + 2] = b;
  }
}

// Worker thread function
static void *worker_thread(void *arg) {
  (void)arg;

  while (1) {
    pthread_mutex_lock(&g_pool.mutex);

    while (g_pool.next_row >= g_pool.total_rows && !g_pool.shutdown) {
      pthread_cond_wait(&g_pool.cond, &g_pool.mutex);
    }

    if (g_pool.shutdown) {
      pthread_mutex_unlock(&g_pool.mutex);
      break;
    }

    int row = g_pool.next_row++;
    pthread_mutex_unlock(&g_pool.mutex);

    if (row < g_pool.total_rows) {
      render_row(g_pool.data, row);

      pthread_mutex_lock(&g_pool.mutex);
      g_pool.rows_done++;
      if (g_pool.rows_done >= g_pool.total_rows) {
        pthread_cond_signal(&g_pool.done_cond);
      }
      pthread_mutex_unlock(&g_pool.mutex);
    }
  }

  return NULL;
}

void init_cpu_renderer(int num_threads) {
  if (num_threads <= 0) {
    num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_threads <= 0) num_threads = 4;
  }

  g_pool.num_threads = num_threads;
  g_pool.threads = malloc(num_threads * sizeof(pthread_t));
  g_pool.shutdown = 0;

  pthread_mutex_init(&g_pool.mutex, NULL);
  pthread_cond_init(&g_pool.cond, NULL);
  pthread_cond_init(&g_pool.done_cond, NULL);

  for (int i = 0; i < num_threads; i++) {
    pthread_create(&g_pool.threads[i], NULL, worker_thread, NULL);
  }
}

void render_cpu(t_data *data) {
  pthread_mutex_lock(&g_pool.mutex);

  g_pool.data = data;
  g_pool.next_row = 0;
  g_pool.rows_done = 0;
  g_pool.total_rows = data->scene.height;

  // Wake all workers
  pthread_cond_broadcast(&g_pool.cond);

  // Wait for completion
  while (g_pool.rows_done < g_pool.total_rows) {
    pthread_cond_wait(&g_pool.done_cond, &g_pool.mutex);
  }

  pthread_mutex_unlock(&g_pool.mutex);
}

void cleanup_cpu_renderer(void) {
  pthread_mutex_lock(&g_pool.mutex);
  g_pool.shutdown = 1;
  pthread_cond_broadcast(&g_pool.cond);
  pthread_mutex_unlock(&g_pool.mutex);

  for (int i = 0; i < g_pool.num_threads; i++) {
    pthread_join(g_pool.threads[i], NULL);
  }

  pthread_mutex_destroy(&g_pool.mutex);
  pthread_cond_destroy(&g_pool.cond);
  pthread_cond_destroy(&g_pool.done_cond);

  free(g_pool.threads);
  memset(&g_pool, 0, sizeof(g_pool));
}

// Hybrid render: use CPU when GPU is busy
void render_hybrid(t_data *data) {
  // For now, just use GPU
  // TODO: implement actual hybrid scheduling
  render_opencl(data);
}
