#ifndef MINIRT_H
#define MINIRT_H

#include <CL/cl_platform.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

// OpenCL & MLX
#include <CL/cl.h>
#include <mlx.h>

#include "types.h"
#include "scene/scene.h"

// OpenCL Context Container
typedef struct {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;
  cl_mem output_buffer;
  cl_mem obj_buffer;
  cl_mem light_buffer;
  cl_mem bvh_buffer;
  cl_uint *host_image_buffer; // Packed ARGB from GPU (no CPU conversion)
} t_cl;

typedef struct {
  bool w;
  bool s;
  bool a;
  bool d;
  bool q;
  bool e;
  bool left;
  bool right;
  bool up;
  bool down;
  bool shift;
  bool space;
} t_keys;

#define KEY_MASK_W 0x0001
#define KEY_MASK_A 0x0002
#define KEY_MASK_S 0x0004
#define KEY_MASK_D 0x0008

#define KEY_MASK_SP 0x0010
#define KEY_MASK_SH 0x0020
#define KEY_MASK_Q 0x0040
#define KEY_MASK_E 0x0080

#define KEY_MASK_LEFT 0x0100
#define KEY_MASK_RIGHT 0x0200
#define KEY_MASK_UP 0x0400
#define KEY_MASK_DOWN 0x0800

// Main Application Struct
typedef struct {
  void *mlx;
  void *win;
  void *img;
  char *img_addr;
  int bpp;
  int line_len;
  int endian;
  bool should_render;
  bool loading;
  int load_total;
  int load_current;
  t_scene scene;
  t_cl cl;
  uint32_t keys;
} t_data;

// --- Prototypes ---

// OpenCL rendering
void init_opencl(t_data *data);
void init_opencl_incremental(t_data *data, int max_objects, int max_lights);
void upload_object(t_data *data, int index);
void upload_light(t_data *data, int index);
void upload_bvh(t_data *data);
void render_opencl(t_data *data);
void cleanup_opencl(t_data *data);

// BVH acceleration
void build_bvh(t_scene *scene);
void build_bvh_with_progress(t_scene *scene,
                             void (*progress_cb)(int current, int total, void *ctx),
                             void *ctx);
void free_bvh(t_scene *scene);
void debug_print_bvh(t_scene *scene);

// CPU rendering (AVX2)
void init_cpu_renderer(int num_threads);
void render_cpu(t_data *data);
void cleanup_cpu_renderer(void);

// Hybrid rendering
void render_hybrid(t_data *data);

// Config watcher
void init_config_watcher(const char *filename);
void check_for_scene_updates(t_data *data);
void cleanup_config_watcher(void);

#endif
