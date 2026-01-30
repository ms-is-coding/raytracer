#include "minirt.h"
#include "mlx.h"
#include "scene/scene.h"
#include <CL/cl_platform.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <X11/keysym.h>
#include <X11/X.h>

// Global incremental parser state
static t_incremental_parser *g_parser = NULL;

// FPS tracking
static int g_frame_count = 0;
static double g_last_fps_time = 0;
static double g_fps = 0;

static double get_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static void update_fps(void) {
  g_frame_count++;
  double now = get_time_ms();
  double elapsed = now - g_last_fps_time;
  if (elapsed >= 1000.0) {
    g_fps = g_frame_count * 1000.0 / elapsed;
    printf("FPS: %.1f\n", g_fps);
    g_frame_count = 0;
    g_last_fps_time = now;
  }
}

static void update_mlx_image(t_data *data) {
  // OpenCL outputs packed ARGB directly - just memcpy
  memcpy(data->img_addr, data->cl.host_image_buffer,
         sizeof(uint32_t) * data->scene.width * data->scene.height);
}

// Hook to close window
static int close_window(t_data *data) {
  if (g_parser) {
    incremental_parser_cleanup(g_parser);
    g_parser = NULL;
  }
  cleanup_config_watcher();
  mlx_destroy_window(data->mlx, data->win);
  // Note: mlx_destroy_image and mlx_destroy_display might be needed depending on MLX version
  // but standard 42 minilibx is minimal.
  cleanup_opencl(data);
  free_bvh(&data->scene);
  free(data->scene.objects);
  free(data->scene.lights);
  exit(0);
  return (0);
}

struct s_key { int keycode; int mask; } masks[] = {
  { XK_w, KEY_MASK_W },
  { XK_a, KEY_MASK_A },
  { XK_s, KEY_MASK_S },
  { XK_d, KEY_MASK_D },
  { XK_space, KEY_MASK_SP },
  { XK_Shift_L, KEY_MASK_SH },
  { XK_Shift_R, KEY_MASK_SH },
  { XK_q, KEY_MASK_Q },
  { XK_e, KEY_MASK_E },
  { XK_Left, KEY_MASK_LEFT },
  { XK_Right, KEY_MASK_RIGHT },
  { XK_Up, KEY_MASK_UP },
  { XK_Down, KEY_MASK_DOWN },
  { 0, 0 }
};

static int key_press(int keycode, t_data *data) {
  if (keycode == XK_Escape) close_window(data);
  struct s_key key;
  int i = 0;
  while ((key = masks[i++]).mask)
    if (key.keycode == keycode) data->keys |= key.mask;
  return (0);
}

static int key_release(int keycode, t_data *data) {
  struct s_key key;
  int i = 0;
  while ((key = masks[i++]).mask)
    if (key.keycode == keycode) data->keys &= ~key.mask;
  return (0);
}

static void update_camera(t_data *data) {
  t_camera *c = &data->scene.camera;
  float m_spd = 0.2f;
  float r_spd = 0.05f;

  // Forward vector (from camera dir)
  float fx = c->dir.s[0];
  float fy = c->dir.s[1];
  float fz = c->dir.s[2];

  // Assuming a world UP for basic movement, but for full 6DOF we'd track a camera UP
  // Right vector (forward x world_up)
  float rx =  fz;
  __attribute__((unused)) float ry = 0;
  float rz = -fx;

  // Normalize Right vector
  float mag = sqrt(rx*rx + rz*rz);
  if (mag > 0) { rx /= mag; rz /= mag; }

  // Movement logic (WASD + vertical)
  if (data->keys & KEY_MASK_W) {
    c->pos.s[0] += fx * m_spd;
    c->pos.s[1] += fy * m_spd;
    c->pos.s[2] += fz * m_spd;
  }
  if (data->keys & KEY_MASK_S) {
    c->pos.s[0] -= fx * m_spd;
    c->pos.s[1] -= fy * m_spd;
    c->pos.s[2] -= fz * m_spd;
  }
  if (data->keys & KEY_MASK_A) {
    c->pos.s[0] += rx * m_spd;
    c->pos.s[2] += rz * m_spd;
  }
  if (data->keys & KEY_MASK_D) {
    c->pos.s[0] -= rx * m_spd;
    c->pos.s[2] -= rz * m_spd;
  }
  if (data->keys & KEY_MASK_SP) { // move up
    c->pos.s[1] += m_spd;
  }
  if (data->keys & KEY_MASK_SH) { // move down
    c->pos.s[1] -= m_spd;
  }

  // Q/E now used for Roll rotation, so we use a different logic for vertical if needed, 
  // but typically people use Space/Shift or similar. 
  // Keeping Q/E for Roll as requested.

  // Yaw (Left/Right Arrows) - Fixed direction
  if (data->keys & (KEY_MASK_LEFT | KEY_MASK_RIGHT)) {
    float angle = (data->keys & KEY_MASK_RIGHT) ? r_spd : -r_spd;
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    float ox = c->dir.s[0];
    float oz = c->dir.s[2];
    c->dir.s[0] = ox * cos_a - oz * sin_a;
    c->dir.s[2] = ox * sin_a + oz * cos_a;
  }

  // Pitch (Up/Down Arrows)
  if (data->keys & (KEY_MASK_UP | KEY_MASK_DOWN)) {
    float angle = (data->keys & KEY_MASK_UP) ? r_spd : -r_spd;
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    float oy = c->dir.s[1];
    float oz = sqrt(c->dir.s[0] * c->dir.s[0] + c->dir.s[2] * c->dir.s[2]);
    // This is a simplified pitch, for true 6DOF you would rotate around the 'right' vector
    c->dir.s[1] = oy * cos_a + oz * sin_a;
    float ratio = sqrt(1.0f - c->dir.s[1] * c->dir.s[1]) / oz;
    c->dir.s[0] *= ratio;
    c->dir.s[2] *= ratio;
  }

  // Roll (Q/E) - Note: The simple kernel logic might need an 'up' vector argument to show roll.
  // For now, these keys are mapped to movement or rotation state.
}

// Draw progress bar during loading (color: 0=green loading, 1=blue BVH)
static void draw_progress_bar_color(t_data *data, float pct, int color_mode) {
  const int width = data->scene.width;
  const int height = data->scene.height;
  const int bar_width = width - 40;
  const int filled = bar_width * pct;
  int *pixels = (int *)data->img_addr;
  const int y_base = height - 30;
  const int fill_color = (color_mode == 1) ? 0x0066CC : 0x00AA00;

  const int y_start = (y_base < 0) ? 0 : y_base;
  const int y_end = (y_base + 6 > height) ? height : y_base + 6;
  const int x_start = 20;
  const int x_end = (20 + bar_width > width) ? width : 20 + bar_width;
  const int filled_x_end = (20 + filled < x_end) ? 20 + filled : x_end;

  const __m256i border_vec = _mm256_set1_epi32(0x444444);
  const __m256i fill_vec = _mm256_set1_epi32(fill_color);
  const __m256i unfill_vec = _mm256_set1_epi32(0x222222);

  for (int y = y_start; y < y_end; y++) {
    int *row_ptr = pixels + y * width;
    int row = y - y_base;
    int x;

    if (row == 0 || row == 5) {
      for (x = x_start; x + 8 <= x_end; x += 8)
        _mm256_storeu_si256((__m256i *)(row_ptr + x), border_vec);
      for (; x < x_end; x++)
        row_ptr[x] = 0x444444;
    } else {
      for (x = x_start; x + 8 <= filled_x_end; x += 8)
        _mm256_storeu_si256((__m256i *)(row_ptr + x), fill_vec);
      for (; x < filled_x_end; x++)
        row_ptr[x] = fill_color;
      for (; x + 8 <= x_end; x += 8)
        _mm256_storeu_si256((__m256i *)(row_ptr + x), unfill_vec);
      for (; x < x_end; x++)
        row_ptr[x] = 0x222222;
    }
  }
}

static void draw_progress_bar(t_data *data) {
  if (data->load_total <= 0) return;
  float pct = (float)data->load_current / data->load_total;
  draw_progress_bar_color(data, pct, 0);
}

// BVH progress callback
static void bvh_progress_callback(int current, int total, void *ctx) {
  printf("Progress...\n");
  t_data *data = (t_data *)ctx;
  float pct = (float)current / total;
  draw_progress_bar_color(data, pct, 1);
  mlx_put_image_to_window(data->mlx, data->win, data->img, 0, 0);
}

static int render_loop(t_data *data) {
  // Handle incremental loading
  if (data->loading && g_parser) {
    if (incremental_parse_next(g_parser)) {
      // Upload the newly parsed object or light
      int obj_idx = data->scene.obj_count - 1;
      int light_idx = data->scene.light_count - 1;

      // Determine what was just parsed by checking counts
      if (obj_idx >= data->load_current) {
        upload_object(data, obj_idx);
        data->load_current = data->scene.obj_count + data->scene.light_count;
      } else if (light_idx >= 0 && (data->scene.obj_count + light_idx + 1) > data->load_current) {
        upload_light(data, light_idx);
        data->load_current = data->scene.obj_count + data->scene.light_count;
      }

      // Render with brute-force (bvh_node_count = 0 disables BVH)
      int saved_bvh = data->scene.bvh_node_count;
      data->scene.bvh_node_count = 0;
      // render_opencl(data);
      data->scene.bvh_node_count = saved_bvh;

      // update_mlx_image(data);
      draw_progress_bar(data);
      // update_fps();
    } else {
      // Loading complete - build BVH and switch to accelerated rendering
      printf("Loading complete. Building BVH (%d objects, %d lights)...\n",
             data->scene.obj_count, data->scene.light_count);
      memset(data->img_addr, 0, sizeof(uint32_t) * data->scene.width * data->scene.height);
      mlx_put_image_to_window(data->mlx, data->win, data->img, 0, 0);

      build_bvh_with_progress(&data->scene, bvh_progress_callback, data);
      debug_print_bvh(&data->scene);
      upload_bvh(data);

      printf("BVH: %d nodes\n", data->scene.bvh_node_count);

      incremental_parser_cleanup(g_parser);
      g_parser = NULL;
      data->loading = false;

      // Final render with BVH
      render_opencl(data);
      update_mlx_image(data);
      update_fps();
    }
  } else {
    // Normal rendering mode
    check_for_scene_updates(data);
    if (data->keys || data->should_render) {
      data->should_render = 0;
      update_camera(data);
      render_opencl(data);
      update_mlx_image(data);
      update_fps();
    }
  }

  mlx_put_image_to_window(data->mlx, data->win, data->img, 0, 0);
  return (0);
}

typedef int (*t_fn)(void);

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <scene.yml>\n", argv[0]);
    return 1;
  }

  t_data data;
  memset(&data, 0, sizeof(t_data));

  // 1. Count scene items for pre-allocation
  printf("Counting items in %s...\n", argv[1]);
  t_scene_counts counts = count_scene_items(argv[1]);
  printf("Found %d objects, %d lights\n", counts.obj_count, counts.light_count);

  // 2. Parse scene header (resolution, camera, render settings)
  printf("Parsing scene header...\n");
  parse_scene_header(argv[1], &data.scene);

  // 3. Init MLX first (so window appears immediately)
  printf("Starting MiniLibX...\n");
  data.mlx = mlx_init();
  if (!data.mlx) return (1);

  data.win = mlx_new_window(data.mlx, data.scene.width, data.scene.height, "MiniRT OpenCL");
  data.img = mlx_new_image(data.mlx, data.scene.width, data.scene.height);
  data.img_addr = mlx_get_data_addr(data.img, &data.bpp, &data.line_len, &data.endian);

  // 4. Init OpenCL with pre-allocated buffers
  printf("Initializing OpenCL (incremental mode)...\n");
  init_opencl_incremental(&data, counts.obj_count, counts.light_count);

  // 5. Setup incremental loading state
  data.loading = true;
  data.load_total = counts.obj_count + counts.light_count;
  data.load_current = 0;

  // 6. Initialize incremental parser
  g_parser = incremental_parser_init(argv[1], &data.scene, counts);

  // 7. Initialize watcher (will be used after loading completes)
  printf("Initializing Watcher...\n");
  init_config_watcher(argv[1]);

  // 8. Initialize FPS timer
  g_last_fps_time = get_time_ms();

  // 9. Event Loop (loading happens in render_loop)
  mlx_hook(data.win, KeyPress, KeyPressMask, (t_fn)(intptr_t)key_press, &data);
  mlx_hook(data.win, KeyRelease, KeyReleaseMask, (t_fn)(intptr_t)key_release, &data);
  mlx_hook(data.win, DestroyNotify, 0, (t_fn)(intptr_t)close_window, &data);

  mlx_loop_hook(data.mlx, (t_fn)(intptr_t)render_loop, &data);
  mlx_loop(data.mlx);

  return 0;
}
