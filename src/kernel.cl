typedef struct {
  float4 pos;
  float4 dir;
  float fov;
  float padding[3];
} t_camera;

typedef struct {
  float reflection;
  float transparency;
  float ior;
  float roughness;
  float metallic;
  float4 color;
  float4 albedo;
  float4 absorption;
} t_material;

typedef enum : int {
  TYPE_SPHERE,
  TYPE_PLANE,
  TYPE_QUADRIC,
  TYPE_BOX,
  TYPE_TORUS,
  TYPE_MOBIUS,
} t_object_type;

typedef struct { float radius; } t_sphere;
typedef struct { float4 normal; } t_plane;
typedef struct { float coeffs[10]; } t_quadric;
typedef struct { float4 half_size; } t_box;
typedef struct { float major, minor; } t_torus;
typedef struct { float radius, width; } t_mobius;

typedef struct {
  t_object_type type;
  int material_id;
  float4 pos;
  float4 rot;
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

typedef enum : int {
  LIGHT_POINT,
  LIGHT_DIRECTIONAL,
  LIGHT_SPOT,
} t_light_type;

typedef struct {
  float4 pos;
  float4 dir;
  float4 color;
  float intensity;
  float angle;
  t_light_type type;
  float padding;
} t_light;

typedef struct {
  int max_bounces;
  float gamma;
  float exposure;
  float epsilon;
} t_render;

typedef struct {
  float4 min;
  float4 max;
} t_aabb;

typedef struct {
  t_aabb bounds;
  int left;
  int right;
  int obj_start;
  int obj_count;
} t_bvh_node;

// AABB-ray intersection (returns tmin, or INFINITY if no hit)
float intersect_aabb(float3 ro, float3 inv_rd, t_aabb box) {
  float3 t1 = (box.min.xyz - ro) * inv_rd;
  float3 t2 = (box.max.xyz - ro) * inv_rd;

  float3 tmin3 = fmin(t1, t2);
  float3 tmax3 = fmax(t1, t2);

  float tmin = fmax(fmax(tmin3.x, tmin3.y), tmin3.z);
  float tmax = fmin(fmin(tmax3.x, tmax3.y), tmax3.z);

  if (tmax < 0.0f || tmin > tmax) return INFINITY;
  return tmin;
}

// GPU-optimized BVH node (matches t_bvh_node_gpu)
typedef struct {
  float4 bbox_min;
  float4 bbox_max;
  int child[2];
  int prim_count[2];
  int parent;
  int axis;
  int pad[2];
} bvh_node_gpu;

// Branchless AABB intersection with precomputed inv_dir and dir_sign
// Returns (tmin, tmax) packed in float2, use tmax < 0 || tmin > tmax for miss
float2 intersect_aabb_gpu(float3 ro, float3 inv_rd, int3 dir_sign,
                          float3 bbox_min, float3 bbox_max) {
  // Select bounds based on ray direction sign (branchless)
  float3 bounds0 = select(bbox_min, bbox_max, dir_sign);
  float3 bounds1 = select(bbox_max, bbox_min, dir_sign);

  float3 t0 = (bounds0 - ro) * inv_rd;
  float3 t1 = (bounds1 - ro) * inv_rd;

  float tmin = fmax(fmax(t0.x, t0.y), t0.z);
  float tmax = fmin(fmin(t1.x, t1.y), t1.z);

  return (float2)(tmin, tmax);
}

// SoA primitive buffers for coalesced access
// pos.w contains type as int bits, param0.w contains material_id
float intersect_prim_soa(float3 ro, float3 rd,
                         __global float4 *prim_pos,
                         __global float4 *prim_param0,
                         __global float4 *prim_param1,
                         int idx) {
  float4 pos4 = prim_pos[idx];
  float3 p = pos4.xyz;
  int type = as_int(pos4.w);
  float4 param0 = prim_param0[idx];

  float3 oc = ro - p;

  if (type == TYPE_SPHERE) {
    float r = param0.x;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - r * r;
    float h = b * b - c;
    if (h < 0.0f) return -1.0f;
    h = sqrt(h);
    float t = -b - h;
    return (t > 0.001f) ? t : ((-b + h > 0.001f) ? -b + h : -1.0f);
  }
  else if (type == TYPE_PLANE) {
    float4 param1 = prim_param1[idx];
    float3 n = param1.xyz;
    float denom = dot(n, rd);
    if (fabs(denom) > 1e-6f) {
      float t = dot(p - ro, n) / denom;
      return (t > 0.001f) ? t : -1.0f;
    }
    return -1.0f;
  }
  else if (type == TYPE_BOX) {
    float3 hs = param0.xyz;
    float3 t1 = (-hs - oc) / rd;
    float3 t2 = (hs - oc) / rd;
    float3 tmin3 = fmin(t1, t2);
    float3 tmax3 = fmax(t1, t2);
    float tmin = fmax(fmax(tmin3.x, tmin3.y), tmin3.z);
    float tmax = fmin(fmin(tmax3.x, tmax3.y), tmax3.z);
    if (tmax < 0.0f || tmin > tmax) return -1.0f;
    return (tmin > 0.001f) ? tmin : ((tmax > 0.001f) ? tmax : -1.0f);
  }
  else if (type == TYPE_TORUS) {
    float R = param0.x;  // major radius
    float r = param0.y;  // minor radius

    // Bounding sphere early-out
    float bound = R + r;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - bound * bound;
    if (b*b - c < 0.0f) return -1.0f;

    // Quartic coefficients for torus
    float Ra2 = R * R;
    float ra2 = r * r;
    float m = dot(oc, oc);
    float n = dot(oc, rd);
    float k = (m - ra2 - Ra2) / 2.0f;
    float k3 = n;
    float k2 = n*n + Ra2*rd.y*rd.y + k;
    float k1 = k*n + Ra2*oc.y*rd.y;
    float k0 = k*k + Ra2*oc.y*oc.y - Ra2*ra2;

    // Solve quartic via resolvent cubic
    float po = 1.0f;
    if (fabs(k3*(k3*k3 - k2) + k1) < 0.01f) {
      po = -1.0f;
      float tmp = k1; k1 = k3; k3 = tmp;
      k0 = 1.0f / k0;
      k1 = k1 * k0;
      k2 = k2 * k0;
      k3 = k3 * k0;
    }

    float c2 = 2.0f*k2 - 3.0f*k3*k3;
    float c1 = k3*(k3*k3 - k2) + k1;
    float c0 = k3*(k3*(-3.0f*k3*k3 + 4.0f*k2) - 8.0f*k1) + 4.0f*k0;

    c2 /= 3.0f;
    c1 *= 2.0f;
    c0 /= 3.0f;

    float Q = c2*c2 + c0;
    float R2 = 3.0f*c0*c2 - c2*c2*c2 - c1*c1;

    float h = R2*R2 - Q*Q*Q;
    float z;

    if (h < 0.0f) {
      float sQ = sqrt(Q);
      z = 2.0f * sQ * cos(acos(clamp(R2/(sQ*Q), -1.0f, 1.0f)) / 3.0f);
    } else {
      float sQ = pow(sqrt(h) + fabs(R2), 1.0f/3.0f);
      z = sign(R2) * fabs(sQ + Q/sQ);
    }
    z = c2 - z;

    float d1 = z - 3.0f*c2;
    float d2 = z*z - c0;
    if (fabs(d1) < 1.0e-4f) {
      if (d2 < 0.0f) return -1.0f;
      d2 = sqrt(d2);
    } else {
      if (d1 < 0.0f) return -1.0f;
      d1 = sqrt(d1 / 2.0f);
      d2 = c1 / d1;
    }

    float result = 1e30f;
    h = d1*d1 - z + d2;
    if (h > 0.0f) {
      h = sqrt(h);
      float t1 = -d1 - h - k3; t1 = (po < 0.0f) ? 2.0f/t1 : t1;
      float t2 = -d1 + h - k3; t2 = (po < 0.0f) ? 2.0f/t2 : t2;
      if (t1 > 0.001f) result = fmin(result, t1);
      if (t2 > 0.001f) result = fmin(result, t2);
    }
    h = d1*d1 - z - d2;
    if (h > 0.0f) {
      h = sqrt(h);
      float t1 = d1 - h - k3; t1 = (po < 0.0f) ? 2.0f/t1 : t1;
      float t2 = d1 + h - k3; t2 = (po < 0.0f) ? 2.0f/t2 : t2;
      if (t1 > 0.001f) result = fmin(result, t1);
      if (t2 > 0.001f) result = fmin(result, t2);
    }

    return (result < 1e29f) ? result : -1.0f;
  }

  return -1.0f;
}

// GPU-optimized BVH traversal with branchless child ordering
// Uses direction sign to order near/far children for early termination
float trace_bvh_gpu(float3 ro, float3 rd, float eps,
                    __global bvh_node_gpu *bvh, int bvh_count,
                    __global float4 *prim_pos,
                    __global float4 *prim_param0,
                    __global float4 *prim_param1,
                    int *out_hit_idx) {
  *out_hit_idx = -1;
  if (bvh_count <= 0) return INFINITY;

  float closest_t = INFINITY;

  // Precompute ray properties (done once per ray)
  float3 inv_rd = native_recip(rd);
  int3 dir_sign = (int3)(rd.x < 0.0f, rd.y < 0.0f, rd.z < 0.0f);

  // Stack-based traversal (32 entries handles trees up to 2^32 nodes)
  int stack[16];
  int stack_ptr = 0;
  stack[stack_ptr++] = 0;

  while (stack_ptr > 0) {
    int node_idx = stack[--stack_ptr];
    __global bvh_node_gpu *node = &bvh[node_idx];

    // AABB test
    float2 t_box = intersect_aabb_gpu(ro, inv_rd, dir_sign,
                                       node->bbox_min.xyz, node->bbox_max.xyz);
    // Early reject: miss or farther than current hit
    if (t_box.y < 0.0f || t_box.x > t_box.y || t_box.x > closest_t)
      continue;

    int left = node->child[0];
    int right = node->child[1];

    // Check if leaf (negative child index encodes leaf)
    if (left < 0) {
      // Leaf: decode primitive range
      int prim_start = ~left;  // bitwise NOT to get start index
      int prim_count = node->prim_count[0];

      for (int i = 0; i < prim_count; i++) {
        int prim_idx = prim_start + i;
        float t = intersect_prim_soa(ro, rd, prim_pos, prim_param0, prim_param1, prim_idx);
        if (t > eps && t < closest_t) {
          closest_t = t;
          *out_hit_idx = prim_idx;
        }
      }
    } else {
      // Internal node: order children by ray direction for better culling
      // If ray goes in positive direction along split axis, visit left (near) first
      int axis = node->axis;
      int near_child, far_child;

      // Branchless selection: dir_sign[axis] ? right : left
      int swap = (axis == 0) ? dir_sign.x : ((axis == 1) ? dir_sign.y : dir_sign.z);
      near_child = swap ? right : left;
      far_child = swap ? left : right;

      // Push far child first (will be popped last)
      stack[stack_ptr++] = far_child;
      stack[stack_ptr++] = near_child;
    }
  }

  return closest_t;
}

// GPU shadow ray test - returns true if occluded (early exit on first hit)
bool trace_shadow_gpu(float3 ro, float3 rd, float max_dist, float eps,
                      __global bvh_node_gpu *bvh, int bvh_count,
                      __global float4 *prim_pos,
                      __global float4 *prim_param0,
                      __global float4 *prim_param1) {
  if (bvh_count <= 0) return false;

  float3 inv_rd = native_recip(rd);
  int3 dir_sign = (int3)(rd.x < 0.0f, rd.y < 0.0f, rd.z < 0.0f);

  int stack[16];
  int stack_ptr = 0;
  stack[stack_ptr++] = 0;

  while (stack_ptr > 0) {
    int node_idx = stack[--stack_ptr];
    __global bvh_node_gpu *node = &bvh[node_idx];

    float2 t_box = intersect_aabb_gpu(ro, inv_rd, dir_sign,
                                       node->bbox_min.xyz, node->bbox_max.xyz);
    if (t_box.y < 0.0f || t_box.x > t_box.y || t_box.x > max_dist)
      continue;

    int left = node->child[0];

    if (left < 0) {
      int prim_start = ~left;
      int prim_count = node->prim_count[0];

      for (int i = 0; i < prim_count; i++) {
        int prim_idx = prim_start + i;
        float t = intersect_prim_soa(ro, rd, prim_pos, prim_param0, prim_param1, prim_idx);
        if (t > eps && t < max_dist) return true;  // Early exit!
      }
    } else {
      int right = node->child[1];
      stack[stack_ptr++] = right;
      stack[stack_ptr++] = left;
    }
  }

  return false;
}

float3 reflect_vec(float3 I, float3 N) {
  return I - 2.0f * dot(N, I) * N;
}

// Rotate vector by Euler angles (in radians) - X, Y, Z order
float3 rotate_euler(float3 v, float3 rot) {
  if (rot.x == 0.0f && rot.y == 0.0f && rot.z == 0.0f) return v;

  float cx = cos(rot.x), sx = sin(rot.x);
  float cy = cos(rot.y), sy = sin(rot.y);
  float cz = cos(rot.z), sz = sin(rot.z);

  // Rotation around X axis
  float y1 = v.y * cx - v.z * sx;
  float z1 = v.y * sx + v.z * cx;
  v.y = y1; v.z = z1;

  // Rotation around Y axis
  float x2 = v.x * cy + v.z * sy;
  float z2 = -v.x * sy + v.z * cy;
  v.x = x2; v.z = z2;

  // Rotation around Z axis
  float x3 = v.x * cz - v.y * sz;
  float y3 = v.x * sz + v.y * cz;
  v.x = x3; v.y = y3;

  return v;
}

// Inverse rotation (for transforming points to local space)
float3 rotate_euler_inv(float3 v, float3 rot) {
  if (rot.x == 0.0f && rot.y == 0.0f && rot.z == 0.0f) return v;

  float cx = cos(rot.x), sx = sin(rot.x);
  float cy = cos(rot.y), sy = sin(rot.y);
  float cz = cos(rot.z), sz = sin(rot.z);

  // Inverse rotation around Z axis
  float x1 = v.x * cz + v.y * sz;
  float y1 = -v.x * sz + v.y * cz;
  v.x = x1; v.y = y1;

  // Inverse rotation around Y axis
  float x2 = v.x * cy - v.z * sy;
  float z2 = v.x * sy + v.z * cy;
  v.x = x2; v.z = z2;

  // Inverse rotation around X axis
  float y3 = v.y * cx + v.z * sx;
  float z3 = -v.y * sx + v.z * cx;
  v.y = y3; v.z = z3;

  return v;
}

float3 refract_vec(float3 I, float3 N, float eta) {
  float k = 1.0f - eta * eta * (1.0f - dot(N, I) * dot(N, I));
  if (k < 0.0f) return (float3)(0, 0, 0);
  return eta * I - (eta * dot(N, I) + sqrt(k)) * N;
}

float fresnel(float3 I, float3 N, float ior) {
  float r0 = (1.0f - ior) / (1.0f + ior);
  r0 = r0 * r0;
  float cosX = -dot(N, I);
  if (ior > 1.0f && cosX < 0.0f) cosX = dot(N, I);
  float x = 1.0f - cosX;
  float x2 = x * x;
  return r0 + (1.0f - r0) * (x2 * x2 * x);  // x^5 = x^2 * x^2 * x
}

float intersect_sphere(float3 ro, float3 rd, __global t_object *s) {
  float3 oc = ro - s->pos.xyz;
  float b = dot(oc, rd);
  float c = dot(oc, oc) - (s->sphere.radius * s->sphere.radius);
  float h = b*b - c;
  if (h < 0.0) return -1.0;
  h = sqrt(h);
  float t = -b - h;
  return (t > 0.001) ? t : -b + h;
}

float intersect_plane(float3 ro, float3 rd, __global t_object *p) {
  float denom = dot(p->plane.normal.xyz, rd);
  if (fabs(denom) > 1e-6) {
    float t = dot(p->pos.xyz - ro, p->plane.normal.xyz) / denom;
    return (t > 0.001) ? t : -1.0;
  }
  return -1.0;
}

float intersect_box(float3 ro, float3 rd, __global t_object *b) {
  float3 p = ro - b->pos.xyz;
  float3 hs = b->box.half_size.xyz;

  float3 t1 = (-hs - p) / rd;
  float3 t2 = (hs - p) / rd;

  float3 tmin3 = fmin(t1, t2);
  float3 tmax3 = fmax(t1, t2);

  float tmin = fmax(fmax(tmin3.x, tmin3.y), tmin3.z);
  float tmax = fmin(fmin(tmax3.x, tmax3.y), tmax3.z);

  if (tmax < 0 || tmin > tmax) return -1.0f;
  return (tmin > 0.001f) ? tmin : ((tmax > 0.001f) ? tmax : -1.0f);
}

float3 box_normal(float3 p, float3 half_size) {
  float3 d = fabs(p) - half_size;
  float3 n = (float3)(0, 0, 0);
  float eps = 0.001f;
  if (fabs(d.x) < eps) n.x = sign(p.x);
  else if (fabs(d.y) < eps) n.y = sign(p.y);
  else if (fabs(d.z) < eps) n.z = sign(p.z);
  return normalize(n);
}

// Optimized cylinder intersection: x² + z² = r² (infinite along Y)
float intersect_cylinder(float3 p, float3 rd, float r2) {
  float a = rd.x*rd.x + rd.z*rd.z;
  if (a < 1e-6f) return -1.0f;  // Ray parallel to cylinder

  float b = p.x*rd.x + p.z*rd.z;
  float c = p.x*p.x + p.z*p.z - r2;

  float disc = b*b - a*c;
  if (disc < 0.0f) return -1.0f;

  float sq = sqrt(disc);
  float t1 = (-b - sq) / a;
  float t2 = (-b + sq) / a;

  if (t1 > 0.001f) return t1;
  if (t2 > 0.001f) return t2;
  return -1.0f;
}

// Optimized cone intersection: x² + z² = y² (apex at origin, along Y)
float intersect_cone(float3 p, float3 rd) {
  float a = rd.x*rd.x + rd.z*rd.z - rd.y*rd.y;
  float b = p.x*rd.x + p.z*rd.z - p.y*rd.y;
  float c = p.x*p.x + p.z*p.z - p.y*p.y;

  if (fabs(a) < 1e-6f) {
    if (fabs(b) < 1e-6f) return -1.0f;
    float t = -c / (2.0f * b);
    return (t > 0.001f) ? t : -1.0f;
  }

  float disc = b*b - a*c;
  if (disc < 0.0f) return -1.0f;

  float sq = sqrt(disc);
  float t1 = (-b - sq) / a;
  float t2 = (-b + sq) / a;

  if (t1 > 0.001f) return t1;
  if (t2 > 0.001f) return t2;
  return -1.0f;
}

// Quadric: Ax² + By² + Cz² + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
// coeffs[0-9] = A,B,C,D,E,F,G,H,I,J
float intersect_quadric(float3 ro, float3 rd, __global t_object *q) {
  float3 p = ro - q->pos.xyz;
  __global float *c = q->quadric.coeffs;

  // Fast path for cylinder: A=1, B=0, C=1, D-I=0, J=-r²
  if (c[1] == 0.0f && c[3] == 0.0f && c[4] == 0.0f && c[5] == 0.0f &&
      c[6] == 0.0f && c[7] == 0.0f && c[8] == 0.0f && c[0] == c[2]) {
    return intersect_cylinder(p, rd, -c[9]);
  }

  // Fast path for cone: A=1, B=-1, C=1, rest=0
  if (c[0] == 1.0f && c[1] == -1.0f && c[2] == 1.0f &&
      c[3] == 0.0f && c[4] == 0.0f && c[5] == 0.0f &&
      c[6] == 0.0f && c[7] == 0.0f && c[8] == 0.0f && c[9] == 0.0f) {
    return intersect_cone(p, rd);
  }

  // General quadric
  float A = c[0]*rd.x*rd.x + c[1]*rd.y*rd.y + c[2]*rd.z*rd.z +
            c[3]*rd.x*rd.y + c[4]*rd.x*rd.z + c[5]*rd.y*rd.z;
  float B = 2*(c[0]*p.x*rd.x + c[1]*p.y*rd.y + c[2]*p.z*rd.z) +
            c[3]*(p.x*rd.y + p.y*rd.x) + c[4]*(p.x*rd.z + p.z*rd.x) +
            c[5]*(p.y*rd.z + p.z*rd.y) +
            c[6]*rd.x + c[7]*rd.y + c[8]*rd.z;
  float C = c[0]*p.x*p.x + c[1]*p.y*p.y + c[2]*p.z*p.z +
            c[3]*p.x*p.y + c[4]*p.x*p.z + c[5]*p.y*p.z +
            c[6]*p.x + c[7]*p.y + c[8]*p.z + c[9];

  if (fabs(A) < 1e-6f) {
    if (fabs(B) < 1e-6f) return -1.0f;
    float t = -C / B;
    return (t > 0.001f) ? t : -1.0f;
  }

  float disc = B*B - 4*A*C;
  if (disc < 0) return -1.0f;

  float sq = sqrt(disc);
  float t1 = (-B - sq) / (2*A);
  float t2 = (-B + sq) / (2*A);

  if (t1 > 0.001f) return t1;
  if (t2 > 0.001f) return t2;
  return -1.0f;
}

float3 quadric_normal(float3 p, __global float *c) {
  // Gradient of Ax² + By² + Cz² + Dxy + Exz + Fyz + Gx + Hy + Iz + J
  float3 n;
  n.x = 2*c[0]*p.x + c[3]*p.y + c[4]*p.z + c[6];
  n.y = 2*c[1]*p.y + c[3]*p.x + c[5]*p.z + c[7];
  n.z = 2*c[2]*p.z + c[4]*p.x + c[5]*p.y + c[8];
  return normalize(n);
}

// Torus with Y-axis orientation (lying flat in XZ plane)
float intersect_torus(float3 ro, float3 rd, __global t_object *tor) {
  float3 p = ro - tor->pos.xyz;
  float R = tor->torus.major;
  float r = tor->torus.minor;

  // Bounding sphere early-out
  float bound = R + r;
  float3 oc = p;
  float b = dot(oc, rd);
  float c = dot(oc, oc) - bound * bound;
  if (b*b - c < 0.0f) return -1.0f;

  // Quartic coefficients for torus: (x² + y² + z² + R² - r²)² = 4R²(x² + z²)
  // Using Y as the axis (ring in XZ plane)
  float Ra2 = R * R;
  float ra2 = r * r;

  float m = dot(p, p);
  float n = dot(p, rd);
  float k = (m - ra2 - Ra2) / 2.0f;
  float k3 = n;
  float k2 = n*n + Ra2*rd.y*rd.y + k;
  float k1 = k*n + Ra2*p.y*rd.y;
  float k0 = k*k + Ra2*p.y*p.y - Ra2*ra2;

  // Solve quartic via resolvent cubic
  float po = 1.0f;
  if (fabs(k3*(k3*k3 - k2) + k1) < 0.01f) {
    po = -1.0f;
    float tmp = k1; k1 = k3; k3 = tmp;
    k0 = 1.0f / k0;
    k1 = k1 * k0;
    k2 = k2 * k0;
    k3 = k3 * k0;
  }

  float c2 = 2.0f*k2 - 3.0f*k3*k3;
  float c1 = k3*(k3*k3 - k2) + k1;
  float c0 = k3*(k3*(-3.0f*k3*k3 + 4.0f*k2) - 8.0f*k1) + 4.0f*k0;

  c2 /= 3.0f;
  c1 *= 2.0f;
  c0 /= 3.0f;

  float Q = c2*c2 + c0;
  float R2 = 3.0f*c0*c2 - c2*c2*c2 - c1*c1;

  float h = R2*R2 - Q*Q*Q;
  float z;

  if (h < 0.0f) {
    float sQ = sqrt(Q);
    z = 2.0f * sQ * cos(acos(clamp(R2/(sQ*Q), -1.0f, 1.0f)) / 3.0f);
  } else {
    float sQ = pow(sqrt(h) + fabs(R2), 1.0f/3.0f);
    z = sign(R2) * fabs(sQ + Q/sQ);
  }
  z = c2 - z;

  float d1 = z - 3.0f*c2;
  float d2 = z*z - c0;
  if (fabs(d1) < 1.0e-4f) {
    if (d2 < 0.0f) return -1.0f;
    d2 = sqrt(d2);
  } else {
    if (d1 < 0.0f) return -1.0f;
    d1 = sqrt(d1 / 2.0f);
    d2 = c1 / d1;
  }

  float result = 1e30f;
  h = d1*d1 - z + d2;
  if (h > 0.0f) {
    h = sqrt(h);
    float t1 = -d1 - h - k3; t1 = (po < 0.0f) ? 2.0f/t1 : t1;
    float t2 = -d1 + h - k3; t2 = (po < 0.0f) ? 2.0f/t2 : t2;
    if (t1 > 0.001f) result = fmin(result, t1);
    if (t2 > 0.001f) result = fmin(result, t2);
  }
  h = d1*d1 - z - d2;
  if (h > 0.0f) {
    h = sqrt(h);
    float t1 = d1 - h - k3; t1 = (po < 0.0f) ? 2.0f/t1 : t1;
    float t2 = d1 + h - k3; t2 = (po < 0.0f) ? 2.0f/t2 : t2;
    if (t1 > 0.001f) result = fmin(result, t1);
    if (t2 > 0.001f) result = fmin(result, t2);
  }

  return (result < 1e29f) ? result : -1.0f;
}

float3 torus_normal(float3 p, float R) {
  // Y-axis torus: ring in XZ plane
  float k = sqrt(p.x*p.x + p.z*p.z);
  float3 n;
  n.x = p.x * (1.0f - R / k);
  n.y = p.y;
  n.z = p.z * (1.0f - R / k);
  return normalize(n);
}

// Mobius strip SDF
float sdf_mobius(float3 q, float R, float w) {
  float u = atan2(q.z, q.x);
  float cu = cos(u), su = sin(u);
  float hu = u * 0.5f;
  float chu = cos(hu), shu = sin(hu);

  // Center of the strip at angle u
  float3 center = (float3)(R * cu, 0.0f, R * su);

  // Local coordinate frame: tangent and twisted normal
  float3 tangent = (float3)(-su, 0.0f, cu);
  float3 up = (float3)(cu * shu, chu, su * shu);

  float3 d = q - center;
  float v = dot(d, up);      // height on strip
  float dist_v = fabs(v) - w;  // distance from strip edge

  // Distance from center line in the plane perpendicular to tangent
  float3 proj = d - dot(d, tangent) * tangent;
  float dist_u = length(proj - v * up);

  return fmax(dist_u - 0.02f, dist_v);
}

float intersect_mobius(float3 ro, float3 rd, __global t_object *m) {
  float3 p = ro - m->pos.xyz;
  float R = m->mobius.radius;
  float w = m->mobius.width;

  // Bounding sphere early-out
  float bound = R + w + 0.1f;
  float b = dot(p, rd);
  float c = dot(p, p) - bound * bound;
  if (b > 0.0f && c > 0.0f) return -1.0f;
  if (b*b - c < 0.0f) return -1.0f;

  // Sphere tracing
  float t = fmax(0.001f, -b - sqrt(b*b - c));
  for (int i = 0; i < 48; i++) {
    float3 q = p + rd * t;
    float d = sdf_mobius(q, R, w);
    if (d < 0.0005f) return t;
    t += d * 0.8f;
    if (t > 100.0f) break;
  }
  return -1.0f;
}

float3 mobius_normal(float3 p, float R, float w) {
  // Numerical gradient of SDF
  float eps = 0.001f;
  float3 n;
  n.x = sdf_mobius(p + (float3)(eps, 0, 0), R, w) - sdf_mobius(p - (float3)(eps, 0, 0), R, w);
  n.y = sdf_mobius(p + (float3)(0, eps, 0), R, w) - sdf_mobius(p - (float3)(0, eps, 0), R, w);
  n.z = sdf_mobius(p + (float3)(0, 0, eps), R, w) - sdf_mobius(p - (float3)(0, 0, eps), R, w);
  return normalize(n);
}

// Single object intersection dispatcher
float intersect_object(float3 ro, float3 rd, __global t_object *obj) {
  switch (obj->type) {
    case TYPE_SPHERE:  return intersect_sphere(ro, rd, obj);
    case TYPE_PLANE:   return intersect_plane(ro, rd, obj);
    case TYPE_BOX:     return intersect_box(ro, rd, obj);
    case TYPE_QUADRIC: return intersect_quadric(ro, rd, obj);
    case TYPE_TORUS:   return intersect_torus(ro, rd, obj);
    case TYPE_MOBIUS:  return intersect_mobius(ro, rd, obj);
    default:           return -1.0f;
  }
}

// BVH traversal - returns closest hit distance and index
float trace_bvh(float3 ro, float3 rd, float eps,
                __global t_object *objects, int obj_count,
                __global t_bvh_node *bvh, int bvh_count,
                int *out_hit_idx) {
  *out_hit_idx = -1;
  float closest_t = 1e30f;

  // If no BVH, use linear search
  if (bvh_count <= 0) {
    for (int i = 0; i < obj_count; i++) {
      float t = intersect_object(ro, rd, &objects[i]);
      if (t > eps && t < closest_t) {
        closest_t = t;
        *out_hit_idx = i;
      }
    }
    return closest_t;
  }

  // Precompute inverse ray direction
  float3 inv_rd = (float3)(1.0f / rd.x, 1.0f / rd.y, 1.0f / rd.z);

  // Stack-based BVH traversal
  int stack[16];
  int stack_ptr = 0;
  stack[stack_ptr++] = 0;

  while (stack_ptr > 0) {
    int node_idx = stack[--stack_ptr];
    __global t_bvh_node *node = &bvh[node_idx];

    // Test AABB
    float box_t = intersect_aabb(ro, inv_rd, node->bounds);
    if (box_t > closest_t || box_t == INFINITY) continue;

    if (node->left == -1) {
      // Leaf node - test objects
      for (int i = 0; i < node->obj_count; i++) {
        int obj_i = node->obj_start + i;
        float t = intersect_object(ro, rd, &objects[obj_i]);
        if (t > eps && t < closest_t) {
          closest_t = t;
          *out_hit_idx = obj_i;
        }
      }
    } else {
      // Internal node - push children (far first for better culling)
      stack[stack_ptr++] = node->right;
      stack[stack_ptr++] = node->left;
    }
  }

  return closest_t;
}

// Shadow ray test - returns true if occluded (early exit on first hit)
bool trace_shadow(float3 ro, float3 rd, float max_dist, float eps,
                  __global t_object *objects, int obj_count,
                  __global t_bvh_node *bvh, int bvh_count) {
  // If no BVH, use linear search with early exit
  if (bvh_count <= 0) {
    for (int i = 0; i < obj_count; i++) {
      float t = intersect_object(ro, rd, &objects[i]);
      if (t > eps && t < max_dist) return true;  // Early exit!
    }
    return false;
  }

  float3 inv_rd = (float3)(1.0f / rd.x, 1.0f / rd.y, 1.0f / rd.z);

  int stack[16];
  int stack_ptr = 0;
  stack[stack_ptr++] = 0;

  while (stack_ptr > 0) {
    int node_idx = stack[--stack_ptr];
    __global t_bvh_node *node = &bvh[node_idx];

    float box_t = intersect_aabb(ro, inv_rd, node->bounds);
    if (box_t > max_dist || box_t == INFINITY) continue;

    if (node->left == -1) {
      for (int i = 0; i < node->obj_count; i++) {
        int obj_i = node->obj_start + i;
        float t = intersect_object(ro, rd, &objects[obj_i]);
        if (t > eps && t < max_dist) return true;  // Early exit!
      }
    } else {
      stack[stack_ptr++] = node->right;
      stack[stack_ptr++] = node->left;
    }
  }

  return false;
}

float3 get_color(
    float3 ro,
    float3 rd,
    __global t_object* objects,
    int obj_count,
    __global t_light* lights,
    int light_count,
    float3 ambient,
    float3 background,
    t_render render,
    __global t_bvh_node* bvh,
    int bvh_count) {
  float3 accum_color = (float3)(0, 0, 0);
  float3 mask = (float3)(1, 1, 1);
  float eps = render.epsilon;

  for (int bounce = 0; bounce < render.max_bounces; bounce++) {
    int hit_idx;
    float closest_t = trace_bvh(ro, rd, eps, objects, obj_count, bvh, bvh_count, &hit_idx);

    if (hit_idx == -1) {
      accum_color += background * mask;
      break;
    }

    __global t_object* obj = &objects[hit_idx];
    float3 hit_p = ro + rd * closest_t;
    float3 local_p = hit_p - obj->pos.xyz;
    float3 n;

    switch (obj->type) {
      case TYPE_SPHERE:  n = normalize(local_p); break;
      case TYPE_PLANE:   n = normalize(obj->plane.normal.xyz); break;
      case TYPE_BOX:     n = box_normal(local_p, obj->box.half_size.xyz); break;
      case TYPE_QUADRIC: n = quadric_normal(local_p, obj->quadric.coeffs); break;
      case TYPE_TORUS:   n = torus_normal(local_p, obj->torus.major); break;
      case TYPE_MOBIUS:  n = mobius_normal(local_p, obj->mobius.radius, obj->mobius.width); break;
      default:           n = (float3)(0, 1, 0); break;
    }

    bool outside = dot(rd, n) < 0;
    if (!outside) n = -n;

    float3 diffuse = (float3)(0, 0, 0);
    for (int i = 0; i < light_count; i++) {
      __global t_light* light = &lights[i];
      float3 l_dir;
      float l_dist;
      float attenuation = 1.0f;

      if (light->type == LIGHT_DIRECTIONAL) {
        l_dir = -normalize(light->dir.xyz);
        l_dist = 1e30f;
      } else {
        l_dir = light->pos.xyz - hit_p;
        l_dist = length(l_dir);
        l_dir = normalize(l_dir);
        // Distance attenuation for point/spot lights
        attenuation = 1.0f; // / (1.0f + 0.01f * l_dist * l_dist);
      }

      // Spot light cone
      if (light->type == LIGHT_SPOT) {
        float spot_cos = dot(-l_dir, normalize(light->dir.xyz));
        float cone_cos = cos(light->angle);
        if (spot_cos < cone_cos) continue; // Outside cone
        // Soft edge falloff
        float outer_cos = cos(light->angle * 1.2f);
        attenuation *= smoothstep(outer_cos, cone_cos, spot_cos);
      }

      // Shadow ray with early exit
      float3 shadow_ro = hit_p + n * eps;
      bool in_shadow = trace_shadow(shadow_ro, l_dir, l_dist, eps, objects, obj_count, bvh, bvh_count);

      if (!in_shadow) {
        float ndotl = max(dot(n, l_dir), 0.0f);
        diffuse += obj->mat.color.xyz * light->color.xyz * ndotl * light->intensity * attenuation;
      }
    }

    // Apply Ambient + Diffuse
    float3 local_col = (diffuse + ambient * obj->mat.color.xyz);

    if (obj->mat.transparency > 0.0f) {
      // Accumulate surface contribution before refraction/reflection
      accum_color += local_col * mask * (1.0f - obj->mat.transparency);
      float eta = outside ? (1.0f / obj->mat.ior) : obj->mat.ior;
      float fr = fresnel(rd, n, obj->mat.ior);
      if (fr > 0.5f) {
        rd = reflect_vec(rd, n);
        ro = hit_p + n * eps;
      } else {
        float3 refr = refract_vec(rd, n, eta);
        if (length(refr) < eps) rd = reflect_vec(rd, n);
        else rd = refr;
        ro = hit_p - n * eps;
      }
      mask *= obj->mat.transparency;
    } else if (obj->mat.reflection > 0.0f) {
      accum_color += local_col * mask * (1.0f - obj->mat.reflection);
      mask *= obj->mat.reflection;
      rd = reflect_vec(rd, n);
      ro = hit_p + n * eps;
    } else {
      accum_color += local_col * mask;
      break;
    }
  }
  return accum_color;
}

__kernel void render_kernel(
    __global uint *output,
    int width,
    int height,
    t_camera cam,
    __global t_object *objects,
    int obj_count,
    __global t_light *lights,
    int light_count,
    float4 ambient,
    float4 background,
    t_render render,
    __global t_bvh_node *bvh,
    int bvh_count) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= width || y >= height) return;

  float aspect = (float)width / height;
  float scale = tan(cam.fov * 0.5f * M_PI_F / 180.0f);
  float px = (2.0f * (x + 0.5f) / width - 1.0f) * aspect * scale;
  float py = (1.0f - 2.0f * (y + 0.5f) / height) * scale;

  float3 forward = normalize(cam.dir.xyz);
  float3 right = normalize(cross(forward, (float3)(0, 1, 0)));
  if (length(right) < 0.001) right = (float3)(1, 0, 0);
  float3 up = cross(right, forward);
  float3 rd = normalize(px * right + py * up + forward);

  float3 col = get_color(cam.pos.xyz, rd, objects, obj_count, lights, light_count, ambient.xyz, background.xyz, render, bvh, bvh_count);

  // Apply exposure and gamma correction
  col = (float3)(1.0f) - exp(-col * render.exposure);
  col = pow(col, (float3)(1.0f / render.gamma));

  // Pack directly to 0x00RRGGBB (clamp and convert on GPU)
  uint r = min((uint)(col.x * 255.99f), 255u);
  uint g = min((uint)(col.y * 255.99f), 255u);
  uint b = min((uint)(col.z * 255.99f), 255u);
  output[y * width + x] = (r << 16) | (g << 8) | b;
}

// =============================================================================
// GPU-OPTIMIZED BATCHED RAY TRACING KERNEL
// =============================================================================

// Ray buffer structure for batched processing
typedef struct {
  float4 origin;     // xyz = origin, w = tmin
  float4 direction;  // xyz = direction, w = tmax
} ray_data;

// Hit result structure
typedef struct {
  float t;           // Hit distance (INFINITY if miss)
  int prim_idx;      // Primitive index (-1 if miss)
  int pad[2];
} hit_data;

// Get normal for SoA primitive at hit point
float3 get_normal_soa(float3 hit_p, float3 local_p, int type,
                      __global float4 *prim_param0,
                      __global float4 *prim_param1,
                      int idx) {
  float4 param0 = prim_param0[idx];

  if (type == TYPE_SPHERE) {
    return normalize(local_p);
  }
  else if (type == TYPE_PLANE) {
    float4 param1 = prim_param1[idx];
    return normalize(param1.xyz);
  }
  else if (type == TYPE_BOX) {
    float3 hs = param0.xyz;
    float3 d = fabs(local_p) - hs;
    float eps = 0.001f;
    float3 n = (float3)(0, 0, 0);
    if (fabs(d.x) < eps) n.x = sign(local_p.x);
    else if (fabs(d.y) < eps) n.y = sign(local_p.y);
    else if (fabs(d.z) < eps) n.z = sign(local_p.z);
    return normalize(n);
  }
  else if (type == TYPE_TORUS) {
    float R = param0.x;  // major radius
    float k = sqrt(local_p.x*local_p.x + local_p.z*local_p.z);
    float3 n;
    n.x = local_p.x * (1.0f - R / k);
    n.y = local_p.y;
    n.z = local_p.z * (1.0f - R / k);
    return normalize(n);
  }

  return (float3)(0, 1, 0);
}

// Batched ray-BVH intersection kernel
// Processes rays in parallel, outputs hit results
// Designed for minimal divergence and coalesced memory access
__kernel void trace_rays_gpu(
    __global ray_data *rays,
    __global hit_data *hits,
    int ray_count,
    __global bvh_node_gpu *bvh,
    int bvh_count,
    __global float4 *prim_pos,
    __global float4 *prim_param0,
    __global float4 *prim_param1,
    float epsilon) {

  int ray_idx = get_global_id(0);
  if (ray_idx >= ray_count) return;

  // Load ray data (coalesced read)
  ray_data ray = rays[ray_idx];
  float3 ro = ray.origin.xyz;
  float3 rd = ray.direction.xyz;
  float tmin = ray.origin.w;
  float tmax = ray.direction.w;

  // Initialize result
  hit_data hit;
  hit.t = INFINITY;
  hit.prim_idx = -1;

  if (bvh_count <= 0) {
    hits[ray_idx] = hit;
    return;
  }

  // Precompute ray properties
  float3 inv_rd = native_recip(rd);
  int3 dir_sign = (int3)(rd.x < 0.0f, rd.y < 0.0f, rd.z < 0.0f);

  // Stack-based traversal
  int stack[16];
  int stack_ptr = 0;
  stack[stack_ptr++] = 0;

  float closest_t = tmax;

  while (stack_ptr > 0) {
    int node_idx = stack[--stack_ptr];
    __global bvh_node_gpu *node = &bvh[node_idx];

    // Branchless AABB test
    float2 t_box = intersect_aabb_gpu(ro, inv_rd, dir_sign,
                                       node->bbox_min.xyz, node->bbox_max.xyz);

    // Skip if miss or farther than closest hit
    float box_tmin = fmax(t_box.x, tmin);
    float box_tmax = fmin(t_box.y, closest_t);
    if (box_tmax < box_tmin) continue;

    int left = node->child[0];

    if (left < 0) {
      // Leaf node: test primitives
      int prim_start = ~left;
      int prim_count = node->prim_count[0];

      for (int i = 0; i < prim_count; i++) {
        int prim_idx = prim_start + i;
        float t = intersect_prim_soa(ro, rd, prim_pos, prim_param0, prim_param1, prim_idx);

        if (t > epsilon && t < closest_t) {
          closest_t = t;
          hit.t = t;
          hit.prim_idx = prim_idx;
        }
      }
    } else {
      // Internal node: order children by ray direction
      int right = node->child[1];
      int axis = node->axis;

      // Branchless near/far selection
      int swap = (axis == 0) ? dir_sign.x : ((axis == 1) ? dir_sign.y : dir_sign.z);
      int near_child = swap ? right : left;
      int far_child = swap ? left : right;

      // Push far first (processed last)
      stack[stack_ptr++] = far_child;
      stack[stack_ptr++] = near_child;
    }
  }

  // Write result (coalesced write)
  hits[ray_idx] = hit;
}

// Full GPU render with optimized BVH (SoA layout)
// This kernel uses the GPU-optimized structures for better performance
__kernel void render_kernel_gpu(
    __global uint *output,
    int width,
    int height,
    t_camera cam,
    __global bvh_node_gpu *bvh,
    int bvh_count,
    __global float4 *prim_pos,
    __global float4 *prim_param0,
    __global float4 *prim_param1,
    __global float4 *prim_mat_color,
    __global float4 *prim_mat_props,
    int prim_count,
    __global t_light *lights,
    int light_count,
    float4 ambient,
    float4 background,
    t_render render) {

  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= width || y >= height) return;

  // Generate primary ray
  float aspect = (float)width / height;
  float scale = tan(cam.fov * 0.5f * M_PI_F / 180.0f);
  float px = (2.0f * (x + 0.5f) / width - 1.0f) * aspect * scale;
  float py = (1.0f - 2.0f * (y + 0.5f) / height) * scale;

  float3 forward = normalize(cam.dir.xyz);
  float3 right = normalize(cross(forward, (float3)(0, 1, 0)));
  if (length(right) < 0.001f) right = (float3)(1, 0, 0);
  float3 up = cross(right, forward);

  float3 ro = cam.pos.xyz;
  float3 rd = normalize(px * right + py * up + forward);

  // Precompute ray properties (reused for all bounces)
  float eps = render.epsilon;

  float3 accum_color = (float3)(0, 0, 0);
  float3 mask = (float3)(1, 1, 1);

  for (int bounce = 0; bounce < render.max_bounces; bounce++) {
    // Trace ray using GPU-optimized BVH
    int hit_idx;
    float closest_t = trace_bvh_gpu(ro, rd, eps, bvh, bvh_count,
                                     prim_pos, prim_param0, prim_param1, &hit_idx);

    if (hit_idx < 0) {
      accum_color += background.xyz * mask;
      break;
    }

    // Get hit info from SoA buffers
    float4 pos4 = prim_pos[hit_idx];
    float3 obj_pos = pos4.xyz;
    int type = as_int(pos4.w);

    float3 hit_p = ro + rd * closest_t;
    float3 local_p = hit_p - obj_pos;

    float3 n = get_normal_soa(hit_p, local_p, type, prim_param0, prim_param1, hit_idx);

    bool outside = dot(rd, n) < 0.0f;
    if (!outside) n = -n;

    // Get material from SoA
    float4 mat_color = prim_mat_color[hit_idx];
    float4 mat_props = prim_mat_props[hit_idx];
    float reflection = mat_props.x;
    float transparency = mat_props.y;
    float ior = mat_props.z;

    // Compute diffuse lighting
    float3 diffuse = (float3)(0, 0, 0);
    for (int i = 0; i < light_count; i++) {
      __global t_light *light = &lights[i];
      float3 l_dir;
      float l_dist;
      float attenuation = 1.0f;

      if (light->type == LIGHT_DIRECTIONAL) {
        l_dir = -normalize(light->dir.xyz);
        l_dist = 1e30f;
      } else {
        l_dir = light->pos.xyz - hit_p;
        l_dist = length(l_dir);
        l_dir = normalize(l_dir);
      }

      // Shadow ray with early exit
      float3 shadow_ro = hit_p + n * eps;
      bool in_shadow = trace_shadow_gpu(shadow_ro, l_dir, l_dist, eps, bvh, bvh_count,
                                        prim_pos, prim_param0, prim_param1);

      if (!in_shadow) {
        float ndotl = fmax(dot(n, l_dir), 0.0f);
        diffuse += mat_color.xyz * light->color.xyz * ndotl * light->intensity * attenuation;
      }
    }

    float3 local_col = diffuse + ambient.xyz * mat_color.xyz;

    if (transparency > 0.0f) {
      // Accumulate surface contribution before refraction/reflection
      accum_color += local_col * mask * (1.0f - transparency);
      float eta = outside ? (1.0f / ior) : ior;
      float fr = fresnel(rd, n, ior);
      if (fr > 0.5f) {
        rd = reflect_vec(rd, n);
        ro = hit_p + n * eps;
      } else {
        float3 refr = refract_vec(rd, n, eta);
        if (length(refr) < eps) rd = reflect_vec(rd, n);
        else rd = refr;
        ro = hit_p - n * eps;
      }
      mask *= transparency;
    } else if (reflection > 0.0f) {
      accum_color += local_col * mask * (1.0f - reflection);
      mask *= reflection;
      rd = reflect_vec(rd, n);
      ro = hit_p + n * eps;
    } else {
      accum_color += local_col * mask;
      break;
    }
  }

  // Tonemap and gamma
  accum_color = (float3)(1.0f) - exp(-accum_color * render.exposure);
  accum_color = pow(accum_color, (float3)(1.0f / render.gamma));

  uint r = min((uint)(accum_color.x * 255.99f), 255u);
  uint g = min((uint)(accum_color.y * 255.99f), 255u);
  uint b_val = min((uint)(accum_color.z * 255.99f), 255u);
  output[y * width + x] = (r << 16) | (g << 8) | b_val;
}
