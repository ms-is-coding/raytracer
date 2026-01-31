#include "minirt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static char *load_kernel_source(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  rewind(f);
  char *source = malloc(len + 1);
  fread(source, 1, len, f);
  source[len] = '\0';
  fclose(f);
  return source;
}

static void check_error(cl_int err, const char *msg) {
  if (err != CL_SUCCESS) {
    fprintf(stderr, "OpenCL Error %d: %s\n", err, msg);
    exit(1);
  }
}

void init_opencl(t_data *data) {
  cl_int err;

  // 1. Setup Platform & Device
  clGetPlatformIDs(1, &data->cl.platform, NULL);
  clGetDeviceIDs(data->cl.platform, CL_DEVICE_TYPE_DEFAULT, 1, &data->cl.device, NULL);

  // 2. Context & Queue
  data->cl.context = clCreateContext(NULL, 1, &data->cl.device, NULL, NULL, &err);
  check_error(err, "Context Creation");

  // Fix for deprecated clCreateCommandQueue
  // Using clCreateCommandQueueWithProperties (OpenCL 2.0+)
  cl_queue_properties props[] = {0};
  data->cl.queue = clCreateCommandQueueWithProperties(data->cl.context, data->cl.device, props, &err);

  // Fallback for older OpenCL versions if necessary
  if (err != CL_SUCCESS) {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    data->cl.queue = clCreateCommandQueue(data->cl.context, data->cl.device, 0, &err);
    #pragma GCC diagnostic pop
  }
  check_error(err, "Queue Creation");

  // 3. Build Program
  char *source = load_kernel_source("src/kernel.cl");
  if (!source) { fprintf(stderr, "Error: Could not load kernel.cl\n"); exit(1); }

  data->cl.program = clCreateProgramWithSource(data->cl.context, 1, (const char **)&source, NULL, &err);
  free(source);

  err = clBuildProgram(data->cl.program, 1, &data->cl.device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    char log[4096];
    clGetProgramBuildInfo(data->cl.program, data->cl.device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
    fprintf(stderr, "Build Log:\n%s\n", log);
    exit(1);
  }

  data->cl.kernel = clCreateKernel(data->cl.program, "render_kernel", &err);
  check_error(err, "Kernel Creation");

  // 4. Create Buffers (packed uint per pixel - no CPU conversion needed)
  size_t pixel_count = data->scene.width * data->scene.height;
  size_t img_size = sizeof(cl_uint) * pixel_count;

  data->cl.host_image_buffer = malloc(img_size);

  data->cl.output_buffer = clCreateBuffer(data->cl.context, CL_MEM_WRITE_ONLY, img_size, NULL, &err);
  data->cl.obj_buffer = clCreateBuffer(data->cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(t_object) * data->scene.obj_count, data->scene.objects, &err);
  data->cl.light_buffer = clCreateBuffer(data->cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(t_light) * data->scene.light_count, data->scene.lights, &err);

  // BVH buffer
  if (data->scene.bvh_node_count > 0) {
    data->cl.bvh_buffer = clCreateBuffer(data->cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(t_bvh_node) * data->scene.bvh_node_count, data->scene.bvh_nodes, &err);
  } else {
    // Create dummy buffer
    t_bvh_node dummy = {0};
    data->cl.bvh_buffer = clCreateBuffer(data->cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(t_bvh_node), &dummy, &err);
  }
}

void render_opencl(t_data *data) {
  cl_int err;

  clSetKernelArg(data->cl.kernel, 0, sizeof(cl_mem), &data->cl.output_buffer);
  clSetKernelArg(data->cl.kernel, 1, sizeof(int), &data->scene.width);
  clSetKernelArg(data->cl.kernel, 2, sizeof(int), &data->scene.height);
  clSetKernelArg(data->cl.kernel, 3, sizeof(t_camera), &data->scene.camera);
  clSetKernelArg(data->cl.kernel, 4, sizeof(cl_mem), &data->cl.obj_buffer);
  clSetKernelArg(data->cl.kernel, 5, sizeof(int), &data->scene.obj_count);
  clSetKernelArg(data->cl.kernel, 6, sizeof(cl_mem), &data->cl.light_buffer);
  clSetKernelArg(data->cl.kernel, 7, sizeof(int), &data->scene.light_count);
  clSetKernelArg(data->cl.kernel, 8, sizeof(cl_float4), &data->scene.ambient_color);
  clSetKernelArg(data->cl.kernel, 9, sizeof(cl_float4), &data->scene.background_color);
  clSetKernelArg(data->cl.kernel, 10, sizeof(t_render), &data->scene.render);
  clSetKernelArg(data->cl.kernel, 11, sizeof(cl_mem), &data->cl.bvh_buffer);
  clSetKernelArg(data->cl.kernel, 12, sizeof(int), &data->scene.bvh_node_count);

  // Enqueue Kernel with explicit work group size for better GPU occupancy
  size_t local_work[2] = { 16, 16 };  // 256 threads per work group
  // Round up global work size to be divisible by local work size
  size_t global_work[2] = {
    ((data->scene.width + local_work[0] - 1) / local_work[0]) * local_work[0],
    ((data->scene.height + local_work[1] - 1) / local_work[1]) * local_work[1]
  };
  err = clEnqueueNDRangeKernel(data->cl.queue, data->cl.kernel, 2, NULL, global_work, local_work, 0, NULL, NULL);
  check_error(err, "Enqueue Kernel");

  // Read Result (packed uint - direct copy to MLX image)
  size_t img_size = sizeof(cl_uint) * data->scene.width * data->scene.height;
  err = clEnqueueReadBuffer(data->cl.queue, data->cl.output_buffer, CL_TRUE, 0, img_size, data->cl.host_image_buffer, 0, NULL, NULL);
  check_error(err, "Read Buffer");
}

void cleanup_opencl(t_data *data) {
  free(data->cl.host_image_buffer);
  clReleaseMemObject(data->cl.output_buffer);
  clReleaseMemObject(data->cl.obj_buffer);
  clReleaseMemObject(data->cl.light_buffer);
  clReleaseMemObject(data->cl.bvh_buffer);
  clReleaseKernel(data->cl.kernel);
  clReleaseProgram(data->cl.program);
  clReleaseCommandQueue(data->cl.queue);
  clReleaseContext(data->cl.context);
}

// Initialize OpenCL with pre-allocated buffers for incremental loading
void init_opencl_incremental(t_data *data, int max_objects, int max_lights) {
  cl_int err;

  // 1. Setup Platform & Device
  clGetPlatformIDs(1, &data->cl.platform, NULL);
  clGetDeviceIDs(data->cl.platform, CL_DEVICE_TYPE_DEFAULT, 1, &data->cl.device, NULL);

  // 2. Context & Queue
  data->cl.context = clCreateContext(NULL, 1, &data->cl.device, NULL, NULL, &err);
  check_error(err, "Context Creation");

  cl_queue_properties props[] = {0};
  data->cl.queue = clCreateCommandQueueWithProperties(data->cl.context, data->cl.device, props, &err);
  if (err != CL_SUCCESS) {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    data->cl.queue = clCreateCommandQueue(data->cl.context, data->cl.device, 0, &err);
    #pragma GCC diagnostic pop
  }
  check_error(err, "Queue Creation");

  // 3. Build Program
  char *source = load_kernel_source("src/kernel.cl");
  if (!source) { fprintf(stderr, "Error: Could not load kernel.cl\n"); exit(1); }

  data->cl.program = clCreateProgramWithSource(data->cl.context, 1, (const char **)&source, NULL, &err);
  free(source);

  err = clBuildProgram(data->cl.program, 1, &data->cl.device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    char log[4096];
    clGetProgramBuildInfo(data->cl.program, data->cl.device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
    fprintf(stderr, "Build Log:\n%s\n", log);
    exit(1);
  }

  data->cl.kernel = clCreateKernel(data->cl.program, "render_kernel", &err);
  check_error(err, "Kernel Creation");

  // 4. Create Buffers with max capacity (packed uint - no CPU conversion)
  size_t pixel_count = data->scene.width * data->scene.height;
  size_t img_size = sizeof(cl_uint) * pixel_count;

  data->cl.host_image_buffer = malloc(img_size);

  data->cl.output_buffer = clCreateBuffer(data->cl.context, CL_MEM_WRITE_ONLY, img_size, NULL, &err);

  // Pre-allocate object buffer at max capacity
  if (max_objects > 0) {
    data->cl.obj_buffer = clCreateBuffer(data->cl.context, CL_MEM_READ_ONLY,
                                         sizeof(t_object) * max_objects, NULL, &err);
  } else {
    t_object dummy = {0};
    data->cl.obj_buffer = clCreateBuffer(data->cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(t_object), &dummy, &err);
  }

  // Pre-allocate light buffer at max capacity
  if (max_lights > 0) {
    data->cl.light_buffer = clCreateBuffer(data->cl.context, CL_MEM_READ_ONLY,
                                           sizeof(t_light) * max_lights, NULL, &err);
  } else {
    t_light dummy = {0};
    data->cl.light_buffer = clCreateBuffer(data->cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(t_light), &dummy, &err);
  }

  // Create dummy BVH buffer (will be properly created after loading)
  t_bvh_node dummy_bvh = {0};
  data->cl.bvh_buffer = clCreateBuffer(data->cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(t_bvh_node), &dummy_bvh, &err);
}

// Upload a single object at the specified index
void upload_object(t_data *data, int index) {
  if (index < 0 || index >= data->scene.obj_count) return;

  cl_int err = clEnqueueWriteBuffer(data->cl.queue, data->cl.obj_buffer, CL_FALSE,
                                    sizeof(t_object) * index, sizeof(t_object),
                                    &data->scene.objects[index], 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error uploading object %d: %d\n", index, err);
  }
}

// Upload a single light at the specified index
void upload_light(t_data *data, int index) {
  if (index < 0 || index >= data->scene.light_count) return;

  cl_int err = clEnqueueWriteBuffer(data->cl.queue, data->cl.light_buffer, CL_FALSE,
                                    sizeof(t_light) * index, sizeof(t_light),
                                    &data->scene.lights[index], 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error uploading light %d: %d\n", index, err);
  }
}

// Upload BVH after all objects are loaded
void upload_bvh(t_data *data) {
  cl_int err;

  if (data->cl.obj_buffer) clReleaseMemObject(data->cl.obj_buffer);
  if (data->cl.light_buffer) clReleaseMemObject(data->cl.light_buffer);
  if (data->cl.bvh_buffer) clReleaseMemObject(data->cl.bvh_buffer);

  // Create new buffers with updated data
  data->cl.obj_buffer = clCreateBuffer(data->cl.context,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(t_object) * data->scene.obj_count, data->scene.objects, &err);

  data->cl.light_buffer = clCreateBuffer(data->cl.context,
                                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(t_light) * data->scene.light_count, data->scene.lights, &err);

  // Create new BVH buffer
  if (data->scene.bvh_node_count > 0) {
    data->cl.bvh_buffer = clCreateBuffer(data->cl.context,
                                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(t_bvh_node) * data->scene.bvh_node_count,
                                         data->scene.bvh_nodes, &err);
  } else {
    t_bvh_node dummy = {0};
    data->cl.bvh_buffer = clCreateBuffer(data->cl.context,
                                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(t_bvh_node), &dummy, &err);
  }
}

// =============================================================================
// GPU-OPTIMIZED BVH AND SOA PRIMITIVE SUPPORT
// =============================================================================

// Determine split axis from existing BVH node bounds (largest extent)
static int compute_split_axis(const t_bvh_node *node, const t_bvh_node *nodes) {
  if (node->left < 0) return 0;  // Leaf node

  // Get child bounds to determine split axis
  const t_bvh_node *left = &nodes[node->left];
  const t_bvh_node *right = &nodes[node->right];

  // Use centroid difference to determine axis
  float left_cx = (left->bounds.min.s[0] + left->bounds.max.s[0]) * 0.5f;
  float left_cy = (left->bounds.min.s[1] + left->bounds.max.s[1]) * 0.5f;
  float left_cz = (left->bounds.min.s[2] + left->bounds.max.s[2]) * 0.5f;

  float right_cx = (right->bounds.min.s[0] + right->bounds.max.s[0]) * 0.5f;
  float right_cy = (right->bounds.min.s[1] + right->bounds.max.s[1]) * 0.5f;
  float right_cz = (right->bounds.min.s[2] + right->bounds.max.s[2]) * 0.5f;

  float dx = fabsf(right_cx - left_cx);
  float dy = fabsf(right_cy - left_cy);
  float dz = fabsf(right_cz - left_cz);

  if (dx >= dy && dx >= dz) return 0;
  if (dy >= dz) return 1;
  return 2;
}

// Flatten BVH to GPU-optimized format with branchless traversal support
t_bvh_node_gpu *flatten_bvh_for_gpu(const t_bvh_node *nodes, int node_count,
                                     int *out_gpu_node_count) {
  if (node_count <= 0) {
    *out_gpu_node_count = 0;
    return NULL;
  }

  t_bvh_node_gpu *gpu_nodes = calloc(node_count, sizeof(t_bvh_node_gpu));
  *out_gpu_node_count = node_count;

  // Build parent pointers (for optional stackless traversal)
  int *parents = calloc(node_count, sizeof(int));
  parents[0] = -1;  // Root has no parent

  for (int i = 0; i < node_count; i++) {
    const t_bvh_node *src = &nodes[i];
    if (src->left >= 0) {
      parents[src->left] = i;
      parents[src->right] = i;
    }
  }

  // Convert each node
  for (int i = 0; i < node_count; i++) {
    const t_bvh_node *src = &nodes[i];
    t_bvh_node_gpu *dst = &gpu_nodes[i];

    // Copy bounds
    dst->bbox_min = src->bounds.min;
    dst->bbox_max = src->bounds.max;
    dst->parent = parents[i];

    if (src->left < 0) {
      // Leaf node: encode primitive start as ~index (bitwise NOT)
      dst->child[0] = ~src->obj_start;  // Negative signals leaf
      dst->child[1] = -1;
      dst->prim_count[0] = src->obj_count;
      dst->prim_count[1] = 0;
      dst->axis = 0;
    } else {
      // Internal node
      dst->child[0] = src->left;
      dst->child[1] = src->right;
      dst->prim_count[0] = 0;
      dst->prim_count[1] = 0;
      dst->axis = compute_split_axis(src, nodes);
    }
  }

  free(parents);
  return gpu_nodes;
}

// Convert AoS objects to SoA layout for coalesced GPU memory access
void convert_objects_to_soa(const t_object *objects, int count, t_primitives_soa *soa) {
  soa->count = count;
  soa->pos = malloc(count * sizeof(cl_float4));
  soa->param0 = malloc(count * sizeof(cl_float4));
  soa->param1 = malloc(count * sizeof(cl_float4));
  soa->param2 = malloc(count * sizeof(cl_float4));
  soa->param3 = malloc(count * sizeof(cl_float4));
  soa->mat_color = malloc(count * sizeof(cl_float4));
  soa->mat_props = malloc(count * sizeof(cl_float4));

  for (int i = 0; i < count; i++) {
    const t_object *obj = &objects[i];

    // Position with type encoded in w
    soa->pos[i] = obj->pos;
    soa->pos[i].s[3] = *(float *)&obj->type;  // Type as float bits

    // Material properties
    soa->mat_color[i] = obj->mat.color;
    soa->mat_props[i] = (cl_float4){{
      obj->mat.reflection,
      obj->mat.transparency,
      obj->mat.ior,
      obj->mat.roughness
    }};

    // Zero-initialize params
    soa->param0[i] = (cl_float4){{0, 0, 0, 0}};
    soa->param1[i] = (cl_float4){{0, 0, 0, 0}};
    soa->param2[i] = (cl_float4){{0, 0, 0, 0}};
    soa->param3[i] = (cl_float4){{0, 0, 0, 0}};

    // Pack type-specific parameters
    switch (obj->type) {
      case TYPE_SPHERE:
        soa->param0[i].s[0] = obj->sphere.radius;
        soa->param0[i].s[3] = *(float *)&obj->material_id;
        break;

      case TYPE_PLANE:
        soa->param0[i].s[3] = *(float *)&obj->material_id;
        soa->param1[i] = obj->plane.normal;
        break;

      case TYPE_BOX:
        soa->param0[i] = obj->box.half_size;
        soa->param0[i].s[3] = *(float *)&obj->material_id;
        break;

      case TYPE_QUADRIC:
        // Spread 10 coefficients across param0-param3
        soa->param0[i] = (cl_float4){{
          obj->quadric.coeffs[0], obj->quadric.coeffs[1],
          obj->quadric.coeffs[2], *(float *)&obj->material_id
        }};
        soa->param1[i] = (cl_float4){{
          obj->quadric.coeffs[3], obj->quadric.coeffs[4],
          obj->quadric.coeffs[5], obj->quadric.coeffs[6]
        }};
        soa->param2[i] = (cl_float4){{
          obj->quadric.coeffs[7], obj->quadric.coeffs[8],
          obj->quadric.coeffs[9], 0
        }};
        break;

      case TYPE_TORUS:
        soa->param0[i].s[0] = obj->torus.major;
        soa->param0[i].s[1] = obj->torus.minor;
        soa->param0[i].s[3] = *(float *)&obj->material_id;
        break;

      case TYPE_MOBIUS:
        soa->param0[i].s[0] = obj->mobius.radius;
        soa->param0[i].s[1] = obj->mobius.width;
        soa->param0[i].s[3] = *(float *)&obj->material_id;
        break;
    }
  }
}

// Free SoA primitive arrays
void free_primitives_soa(t_primitives_soa *soa) {
  free(soa->pos);
  free(soa->param0);
  free(soa->param1);
  free(soa->param2);
  free(soa->param3);
  free(soa->mat_color);
  free(soa->mat_props);
  soa->count = 0;
}

// Upload GPU-optimized BVH and SoA primitives
t_gpu_buffers *create_gpu_bvh_buffers(t_data *data) {
  cl_int err;
  t_gpu_buffers *bufs = calloc(1, sizeof(t_gpu_buffers));

  // Flatten BVH to GPU format
  int gpu_node_count;
  t_bvh_node_gpu *gpu_nodes = flatten_bvh_for_gpu(
    data->scene.bvh_nodes, data->scene.bvh_node_count, &gpu_node_count);

  if (gpu_node_count > 0) {
    bufs->bvh_gpu = clCreateBuffer(data->cl.context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(t_bvh_node_gpu) * gpu_node_count,
                                   gpu_nodes, &err);
    check_error(err, "BVH GPU buffer");
    free(gpu_nodes);
  }

  // Convert objects to SoA
  t_primitives_soa soa;
  convert_objects_to_soa(data->scene.objects, data->scene.obj_count, &soa);

  bufs->prim_pos = clCreateBuffer(data->cl.context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(cl_float4) * soa.count, soa.pos, &err);
  bufs->prim_param0 = clCreateBuffer(data->cl.context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(cl_float4) * soa.count, soa.param0, &err);
  bufs->prim_param1 = clCreateBuffer(data->cl.context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(cl_float4) * soa.count, soa.param1, &err);
  bufs->prim_param2 = clCreateBuffer(data->cl.context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(cl_float4) * soa.count, soa.param2, &err);
  bufs->prim_param3 = clCreateBuffer(data->cl.context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(cl_float4) * soa.count, soa.param3, &err);
  bufs->prim_mat_color = clCreateBuffer(data->cl.context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        sizeof(cl_float4) * soa.count, soa.mat_color, &err);
  bufs->prim_mat_props = clCreateBuffer(data->cl.context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        sizeof(cl_float4) * soa.count, soa.mat_props, &err);

  free_primitives_soa(&soa);
  return bufs;
}

// Release GPU buffers
void release_gpu_bvh_buffers(t_gpu_buffers *bufs) {
  if (!bufs) return;
  if (bufs->bvh_gpu) clReleaseMemObject(bufs->bvh_gpu);
  if (bufs->prim_pos) clReleaseMemObject(bufs->prim_pos);
  if (bufs->prim_param0) clReleaseMemObject(bufs->prim_param0);
  if (bufs->prim_param1) clReleaseMemObject(bufs->prim_param1);
  if (bufs->prim_param2) clReleaseMemObject(bufs->prim_param2);
  if (bufs->prim_param3) clReleaseMemObject(bufs->prim_param3);
  if (bufs->prim_mat_color) clReleaseMemObject(bufs->prim_mat_color);
  if (bufs->prim_mat_props) clReleaseMemObject(bufs->prim_mat_props);
  free(bufs);
}
