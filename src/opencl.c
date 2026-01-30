#include "minirt.h"
#include <stdio.h>

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

  // Set Arguments
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

  // Enqueue Kernel
  size_t global_work[2] = { (size_t)data->scene.width, (size_t)data->scene.height };
  err = clEnqueueNDRangeKernel(data->cl.queue, data->cl.kernel, 2, NULL, global_work, NULL, 0, NULL, NULL);
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
