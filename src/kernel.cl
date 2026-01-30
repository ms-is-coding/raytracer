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
  int type; // 0: Sphere, 1: Plane
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
  int type;
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

float3 reflect_vec(float3 I, float3 N) {
  return I - 2.0f * dot(N, I) * N;
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
  return r0 + (1.0f - r0) * pow(1.0f - cosX, 5.0f);
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
  int stack[32];
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

      // Shadow ray
      float3 shadow_ro = hit_p + n * eps;
      int shadow_idx;
      float shadow_t = trace_bvh(shadow_ro, l_dir, eps, objects, obj_count, bvh, bvh_count, &shadow_idx);

      if (shadow_t > l_dist) {
        float ndotl = max(dot(n, l_dir), 0.0f);
        diffuse += obj->mat.color.xyz * light->color.xyz * ndotl * light->intensity * attenuation;
      }
    }

    // Apply Ambient + Diffuse
    float3 local_col = (diffuse + ambient * obj->mat.color.xyz);

    if (obj->mat.transparency > 0.0f) {
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
