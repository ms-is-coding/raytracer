#include "minirt.h"
#include "mlx.h"
#include "scene/scene.h"
#include "ui/ui.h"
#include <CL/cl_platform.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <X11/keysym.h>
#include <X11/X.h>
#include <X11/Xlib.h>

// Global incremental parser state
static t_incremental_parser *g_parser = NULL;

// FPS tracking
static int g_frame_count = 0;
static double g_last_fps_time = 0;
static double g_fps = 0;
static double g_frame_time_ms = 0;

// Debug mode and object selection
static bool g_debug_mode = false;
static int g_selected_object = -1;  // -1 = no selection
static float g_selected_distance = 0.0f;
static float g_hit_point[3] = {0, 0, 0};
static float g_hit_normal[3] = {0, 1, 0};

// UI System
static t_ui_system g_ui;
static int32_t g_mouse_x = 0;
static int32_t g_mouse_y = 0;
static bool g_mouse_pressed = false;

// UI state bindings
static bool g_ui_antialiasing = true;
static bool g_ui_shadows = true;
static float g_ui_exposure = 0.5f;
static float g_ui_roughness = 0.5f;

// UI Callback functions
static void ui_put_pixel(void *data, int32_t x, int32_t y, uint32_t color) {
  t_data *d = (t_data *)data;
  if (x < 0 || y < 0 || x >= d->scene.width || y >= d->scene.height) return;
  ((uint32_t *)d->img_addr)[y * d->scene.width + x] = color;
}

static uint32_t ui_get_color(void *data, int32_t x, int32_t y) {
  t_data *d = (t_data *)data;
  if (x < 0 || y < 0 || x >= d->scene.width || y >= d->scene.height) return 0;
  return ((uint32_t *)d->img_addr)[y * d->scene.width + x];
}

static int32_t ui_get_mouse_x(void *data) { (void)data; return g_mouse_x; }
static int32_t ui_get_mouse_y(void *data) { (void)data; return g_mouse_y; }
static int32_t ui_mouse_pressed(void *data) { (void)data; return g_mouse_pressed; }

static void init_ui_system(t_data *data) {
  ui_init(&g_ui, data, data->scene.width, data->scene.height);

  // Set up callbacks
  g_ui.ctx.put_pixel = ui_put_pixel;
  g_ui.ctx.get_color = ui_get_color;
  g_ui.ctx.get_mouse_x = ui_get_mouse_x;
  g_ui.ctx.get_mouse_y = ui_get_mouse_y;
  g_ui.ctx.mouse_pressed = ui_mouse_pressed;

  return;
  // Add a settings panel
  ui_add_component(&g_ui, UI_PANEL("Render Settings", 10, 10, 180, 160));

  // Add toggle switches
  ui_add_component(&g_ui, UI_LABEL("Antialiasing", 20, 35, 0xCCCCCC));
  ui_add_component(&g_ui, UI_SWITCH(&g_ui_antialiasing, 130, 32, 48, 24));

  ui_add_component(&g_ui, UI_LABEL("Shadows", 20, 65, 0xCCCCCC));
  ui_add_component(&g_ui, UI_SWITCH(&g_ui_shadows, 130, 62, 48, 24));

  // Add sliders
  ui_add_component(&g_ui, UI_LABEL("Exposure", 20, 95, 0xCCCCCC));
  ui_add_component(&g_ui, UI_SLIDER(&g_ui_exposure, 20, 110, 150, 12));

  ui_add_component(&g_ui, UI_LABEL("Roughness", 20, 130, 0xCCCCCC));
  ui_add_component(&g_ui, UI_SLIDER(&g_ui_roughness, 20, 145, 150, 12));
}

static double get_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static void update_fps(double render_time) {
  g_frame_count++;
  g_frame_time_ms = render_time;
  double now = get_time_ms();
  double elapsed = now - g_last_fps_time;
  if (elapsed >= 1000.0) {
    g_fps = g_frame_count * 1000.0 / elapsed;
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
  if (keycode == XK_Escape) {
    if (g_selected_object >= 0) {
      g_selected_object = -1;  // Deselect object
      data->should_render = true;
    } else {
      close_window(data);
    }
    return 0;
  }
  if (keycode == XK_F3 || keycode == XK_g) {
    g_debug_mode = !g_debug_mode;
    data->should_render = true;
    return 0;
  }
  if (keycode == XK_F5) {
    // Force scene reload
    data->should_render = true;
    return 0;
  }
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

// Ray-sphere intersection for picking
static float intersect_sphere(float ox, float oy, float oz, float dx, float dy, float dz,
                               float cx, float cy, float cz, float radius) {
  float ocx = ox - cx, ocy = oy - cy, ocz = oz - cz;
  float b = 2.0f * (ocx*dx + ocy*dy + ocz*dz);
  float c = ocx*ocx + ocy*ocy + ocz*ocz - radius*radius;
  float disc = b*b - 4*c;
  if (disc < 0) return -1.0f;
  float t = (-b - copysignf(sqrtf(disc), b)) * 0.5f;
  return t > 0 ? t : -1.0f;
}

// Ray-plane intersection for picking
static float intersect_plane(float ox, float oy, float oz, float dx, float dy, float dz,
                              float px, float py, float pz, float nx, float ny, float nz) {
  float denom = dx*nx + dy*ny + dz*nz;
  if (fabsf(denom) < 1e-6f) return -1.0f;
  float t = ((px-ox)*nx + (py-oy)*ny + (pz-oz)*nz) / denom;
  return t > 0 ? t : -1.0f;
}

// Ray-box intersection for picking
static float intersect_box(float ox, float oy, float oz, float dx, float dy, float dz,
                            float cx, float cy, float cz, float hx, float hy, float hz) {
  float invdx = 1.0f / (fabsf(dx) > 1e-6f ? dx : 1e-6f);
  float invdy = 1.0f / (fabsf(dy) > 1e-6f ? dy : 1e-6f);
  float invdz = 1.0f / (fabsf(dz) > 1e-6f ? dz : 1e-6f);

  float t1x = (cx - hx - ox) * invdx, t2x = (cx + hx - ox) * invdx;
  float t1y = (cy - hy - oy) * invdy, t2y = (cy + hy - oy) * invdy;
  float t1z = (cz - hz - oz) * invdz, t2z = (cz + hz - oz) * invdz;

  float tmin = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
  float tmax = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));

  if (tmax < 0 || tmin > tmax) return -1.0f;
  return tmin > 0 ? tmin : tmax;
}

// Pick object at screen coordinates
static int pick_object(t_data *data, int screen_x, int screen_y, float *out_distance) {
  t_camera *c = &data->scene.camera;
  int w = data->scene.width;
  int h = data->scene.height;

  // Match the kernel's ray generation exactly
  float aspect = (float)w / h;
  float scale = tanf(c->fov * 0.5f * M_PI / 180.0f);
  float px = (2.0f * (screen_x + 0.5f) / w - 1.0f) * aspect * scale;
  float py = (1.0f - 2.0f * (screen_y + 0.5f) / h) * scale;

  // Forward vector (normalized)
  float fx = c->dir.s[0], fy = c->dir.s[1], fz = c->dir.s[2];
  float fmag = sqrtf(fx*fx + fy*fy + fz*fz);
  if (fmag > 0) { fx /= fmag; fy /= fmag; fz /= fmag; }

  // Right vector = forward x (0, 1, 0)
  float rx = fy * 0 - fz * 1;  // = -fz
  float ry = fz * 0 - fx * 0;  // = 0
  float rz = fx * 1 - fy * 0;  // = fx
  // Simplify: right = (-fz, 0, fx), but cross(forward, up) is (fy*0 - fz*1, fz*0 - fx*0, fx*1 - fy*0)
  // Actually: cross((fx,fy,fz), (0,1,0)) = (fy*0 - fz*1, fz*0 - fx*0, fx*1 - fy*0) = (-fz, 0, fx)
  rx = -fz; ry = 0; rz = fx;
  float rmag = sqrtf(rx*rx + ry*ry + rz*rz);
  if (rmag < 0.001f) { rx = 1; ry = 0; rz = 0; }
  else { rx /= rmag; ry /= rmag; rz /= rmag; }

  // Up vector = right x forward
  float ux = ry * fz - rz * fy;
  float uy = rz * fx - rx * fz;
  float uz = rx * fy - ry * fx;

  // Ray direction in world space
  float dx = px * rx + py * ux + fx;
  float dy = px * ry + py * uy + fy;
  float dz = px * rz + py * uz + fz;
  float dmag = sqrtf(dx*dx + dy*dy + dz*dz);
  dx /= dmag; dy /= dmag; dz /= dmag;

  // Ray origin
  float ox = c->pos.s[0], oy = c->pos.s[1], oz = c->pos.s[2];

  int closest_idx = -1;
  float closest_t = 1e30f;

  for (int i = 0; i < data->scene.obj_count; i++) {
    t_object *obj = &data->scene.objects[i];
    float t = -1.0f;

    switch (obj->type) {
      case TYPE_SPHERE:
        t = intersect_sphere(ox, oy, oz, dx, dy, dz,
                             obj->pos.s[0], obj->pos.s[1], obj->pos.s[2],
                             obj->sphere.radius);
        break;
      case TYPE_PLANE:
        t = intersect_plane(ox, oy, oz, dx, dy, dz,
                            obj->pos.s[0], obj->pos.s[1], obj->pos.s[2],
                            obj->plane.normal.s[0], obj->plane.normal.s[1], obj->plane.normal.s[2]);
        break;
      case TYPE_BOX:
        t = intersect_box(ox, oy, oz, dx, dy, dz,
                          obj->pos.s[0], obj->pos.s[1], obj->pos.s[2],
                          obj->box.half_size.s[0], obj->box.half_size.s[1], obj->box.half_size.s[2]);
        break;
      default:
        // For other types, use a bounding sphere approximation
        t = intersect_sphere(ox, oy, oz, dx, dy, dz,
                             obj->pos.s[0], obj->pos.s[1], obj->pos.s[2], 1.0f);
        break;
    }

    if (t > 0 && t < closest_t) {
      closest_t = t;
      closest_idx = i;
    }
  }

  // Compute hit point and normal for selected object
  if (closest_idx >= 0) {
    g_hit_point[0] = ox + dx * closest_t;
    g_hit_point[1] = oy + dy * closest_t;
    g_hit_point[2] = oz + dz * closest_t;

    t_object *obj = &data->scene.objects[closest_idx];
    float lx = g_hit_point[0] - obj->pos.s[0];
    float ly = g_hit_point[1] - obj->pos.s[1];
    float lz = g_hit_point[2] - obj->pos.s[2];

    switch (obj->type) {
      case TYPE_SPHERE: {
        float mag = sqrtf(lx*lx + ly*ly + lz*lz);
        g_hit_normal[0] = lx / mag;
        g_hit_normal[1] = ly / mag;
        g_hit_normal[2] = lz / mag;
        break;
      }
      case TYPE_PLANE:
        g_hit_normal[0] = obj->plane.normal.s[0];
        g_hit_normal[1] = obj->plane.normal.s[1];
        g_hit_normal[2] = obj->plane.normal.s[2];
        break;
      case TYPE_BOX: {
        // Determine which face was hit
        float hx = obj->box.half_size.s[0];
        float hy = obj->box.half_size.s[1];
        float hz = obj->box.half_size.s[2];
        float eps = 0.001f;
        g_hit_normal[0] = g_hit_normal[1] = g_hit_normal[2] = 0;
        if (fabsf(fabsf(lx) - hx) < eps) g_hit_normal[0] = lx > 0 ? 1 : -1;
        else if (fabsf(fabsf(ly) - hy) < eps) g_hit_normal[1] = ly > 0 ? 1 : -1;
        else if (fabsf(fabsf(lz) - hz) < eps) g_hit_normal[2] = lz > 0 ? 1 : -1;
        break;
      }
      default:
        g_hit_normal[0] = 0; g_hit_normal[1] = 1; g_hit_normal[2] = 0;
        break;
    }
  }

  *out_distance = closest_t;
  return closest_idx;
}

// Mouse button handler - object selection
static int mouse_button(int button, int x, int y, t_data *data) {
  g_mouse_x = x;
  g_mouse_y = y;

  if (button == 1) {  // Left click
    g_mouse_pressed = true;
    // Check if clicking on UI - skip object selection if so
    if (x < 200 && y < 180) {
      data->should_render = true;
      return 0;
    }
    float dist;
    int idx = pick_object(data, x, y, &dist);
    if (idx >= 0) {
      g_selected_object = idx;
      g_selected_distance = dist;
      g_debug_mode = true;  // Auto-enable debug mode when selecting
    } else {
      g_selected_object = -1;
    }
    data->should_render = true;
  } else if (button == 3) {  // Right click - deselect
    g_selected_object = -1;
    data->should_render = true;
  } else if (button == 4) {  // Scroll up - increase FOV
    data->scene.camera.fov += 2.0f;
    if (data->scene.camera.fov > 120.0f) data->scene.camera.fov = 120.0f;
    data->should_render = true;
  } else if (button == 5) {  // Scroll down - decrease FOV
    data->scene.camera.fov -= 2.0f;
    if (data->scene.camera.fov < 10.0f) data->scene.camera.fov = 10.0f;
    data->should_render = true;
  }

  return 0;
}

// Mouse release handler
static int mouse_release(int button, int x, int y, t_data *data) {
  g_mouse_x = x;
  g_mouse_y = y;
  if (button == 1) {
    g_mouse_pressed = false;
    data->should_render = true;
  }
  return 0;
}

// Mouse motion handler
static int mouse_motion(int x, int y, t_data *data) {
  g_mouse_x = x;
  g_mouse_y = y;
  if (g_mouse_pressed) {
    data->should_render = true;
  }
  return 0;
}

// Draw a single character (simple 5x7 bitmap font)
static void draw_char(t_data *data, int x, int y, char c, int color) {
  // Simple bitmap font data for 0-9, A-Z, a-z, and some symbols
  static const uint8_t font[128][7] = {
    // ASCII 32-47 (space, punctuation)
    [' '] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    ['.'] = {0x00,0x00,0x00,0x00,0x00,0x00,0x04},
    [':'] = {0x00,0x04,0x00,0x00,0x04,0x00,0x00},
    ['-'] = {0x00,0x00,0x00,0x1F,0x00,0x00,0x00},
    ['/'] = {0x01,0x02,0x04,0x08,0x10,0x00,0x00},
    ['('] = {0x02,0x04,0x08,0x08,0x08,0x04,0x02},
    [')'] = {0x08,0x04,0x02,0x02,0x02,0x04,0x08},
    ['%'] = {0x18,0x19,0x02,0x04,0x08,0x13,0x03},
    // Numbers
    ['0'] = {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E},
    ['1'] = {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E},
    ['2'] = {0x0E,0x11,0x01,0x06,0x08,0x10,0x1F},
    ['3'] = {0x0E,0x11,0x01,0x06,0x01,0x11,0x0E},
    ['4'] = {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02},
    ['5'] = {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E},
    ['6'] = {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E},
    ['7'] = {0x1F,0x01,0x02,0x04,0x08,0x08,0x08},
    ['8'] = {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E},
    ['9'] = {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C},
    // Uppercase letters
    ['A'] = {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11},
    ['B'] = {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E},
    ['C'] = {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E},
    ['D'] = {0x1E,0x11,0x11,0x11,0x11,0x11,0x1E},
    ['E'] = {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F},
    ['F'] = {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10},
    ['G'] = {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F},
    ['H'] = {0x11,0x11,0x11,0x1F,0x11,0x11,0x11},
    ['I'] = {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E},
    ['J'] = {0x01,0x01,0x01,0x01,0x11,0x11,0x0E},
    ['K'] = {0x11,0x12,0x14,0x18,0x14,0x12,0x11},
    ['L'] = {0x10,0x10,0x10,0x10,0x10,0x10,0x1F},
    ['M'] = {0x11,0x1B,0x15,0x15,0x11,0x11,0x11},
    ['N'] = {0x11,0x19,0x15,0x13,0x11,0x11,0x11},
    ['O'] = {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E},
    ['P'] = {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10},
    ['Q'] = {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D},
    ['R'] = {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11},
    ['S'] = {0x0E,0x11,0x10,0x0E,0x01,0x11,0x0E},
    ['T'] = {0x1F,0x04,0x04,0x04,0x04,0x04,0x04},
    ['U'] = {0x11,0x11,0x11,0x11,0x11,0x11,0x0E},
    ['V'] = {0x11,0x11,0x11,0x11,0x0A,0x0A,0x04},
    ['W'] = {0x11,0x11,0x11,0x15,0x15,0x15,0x0A},
    ['X'] = {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11},
    ['Y'] = {0x11,0x11,0x0A,0x04,0x04,0x04,0x04},
    ['Z'] = {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F},
    // Lowercase (simplified - same as upper for most)
    ['a'] = {0x00,0x00,0x0E,0x01,0x0F,0x11,0x0F},
    ['b'] = {0x10,0x10,0x1E,0x11,0x11,0x11,0x1E},
    ['c'] = {0x00,0x00,0x0E,0x11,0x10,0x11,0x0E},
    ['d'] = {0x01,0x01,0x0F,0x11,0x11,0x11,0x0F},
    ['e'] = {0x00,0x00,0x0E,0x11,0x1F,0x10,0x0E},
    ['f'] = {0x06,0x08,0x1E,0x08,0x08,0x08,0x08},
    ['g'] = {0x00,0x0F,0x11,0x11,0x0F,0x01,0x0E},
    ['h'] = {0x10,0x10,0x1E,0x11,0x11,0x11,0x11},
    ['i'] = {0x04,0x00,0x0C,0x04,0x04,0x04,0x0E},
    ['j'] = {0x02,0x00,0x06,0x02,0x02,0x12,0x0C},
    ['k'] = {0x10,0x10,0x12,0x14,0x18,0x14,0x12},
    ['l'] = {0x0C,0x04,0x04,0x04,0x04,0x04,0x0E},
    ['m'] = {0x00,0x00,0x1A,0x15,0x15,0x11,0x11},
    ['n'] = {0x00,0x00,0x1E,0x11,0x11,0x11,0x11},
    ['o'] = {0x00,0x00,0x0E,0x11,0x11,0x11,0x0E},
    ['p'] = {0x00,0x1E,0x11,0x11,0x1E,0x10,0x10},
    ['q'] = {0x00,0x0F,0x11,0x11,0x0F,0x01,0x01},
    ['r'] = {0x00,0x00,0x16,0x19,0x10,0x10,0x10},
    ['s'] = {0x00,0x00,0x0F,0x10,0x0E,0x01,0x1E},
    ['t'] = {0x08,0x08,0x1E,0x08,0x08,0x09,0x06},
    ['u'] = {0x00,0x00,0x11,0x11,0x11,0x13,0x0D},
    ['v'] = {0x00,0x00,0x11,0x11,0x11,0x0A,0x04},
    ['w'] = {0x00,0x00,0x11,0x11,0x15,0x15,0x0A},
    ['x'] = {0x00,0x00,0x11,0x0A,0x04,0x0A,0x11},
    ['y'] = {0x00,0x11,0x11,0x11,0x0F,0x01,0x0E},
    ['z'] = {0x00,0x00,0x1F,0x02,0x04,0x08,0x1F},
  };

  unsigned char uc = (unsigned char)c;
  if (uc > 127) return;
  int *pixels = (int *)data->img_addr;
  int w = data->scene.width;
  int h = data->scene.height;

  for (int row = 0; row < 7; row++) {
    uint8_t bits = font[uc][row];
    for (int col = 0; col < 5; col++) {
      if (bits & (0x10 >> col)) {
        int px = x + col;
        int py = y + row;
        if (px >= 0 && px < w && py >= 0 && py < h) {
          pixels[py * w + px] = color;
        }
      }
    }
  }
}

// Draw string
static void draw_string(t_data *data, int x, int y, const char *str, int color) {
  while (*str) {
    draw_char(data, x, y, *str, color);
    x += 6;  // Character width + spacing
    str++;
  }
}

// Draw a line using Bresenham's algorithm
static void draw_line(t_data *data, int x0, int y0, int x1, int y1, int color) {
  int *pixels = (int *)data->img_addr;
  int w = data->scene.width;
  int h = data->scene.height;

  int dx = abs(x1 - x0);
  int dy = abs(y1 - y0);
  int sx = x0 < x1 ? 1 : -1;
  int sy = y0 < y1 ? 1 : -1;
  int err = dx - dy;

  while (1) {
    if (x0 >= 0 && x0 < w && y0 >= 0 && y0 < h) {
      pixels[y0 * w + x0] = color;
    }
    if (x0 == x1 && y0 == y1) break;
    int e2 = 2 * err;
    if (e2 > -dy) { err -= dy; x0 += sx; }
    if (e2 < dx) { err += dx; y0 += sy; }
  }
}

// Project 3D point to screen coordinates
static int project_point(t_data *data, float px, float py, float pz, int *sx, int *sy) {
  t_camera *c = &data->scene.camera;

  // Forward vector (normalized) - matches kernel
  float fx = c->dir.s[0], fy = c->dir.s[1], fz = c->dir.s[2];
  float fmag = sqrtf(fx*fx + fy*fy + fz*fz);
  if (fmag > 0) { fx /= fmag; fy /= fmag; fz /= fmag; }

  // Right vector = cross(forward, (0,1,0)) = (-fz, 0, fx)
  float rx = -fz, ry = 0, rz = fx;
  float rmag = sqrtf(rx*rx + ry*ry + rz*rz);
  if (rmag < 0.001f) { rx = 1; ry = 0; rz = 0; }
  else { rx /= rmag; ry /= rmag; rz /= rmag; }

  // Up vector = cross(right, forward)
  float ux = ry * fz - rz * fy;
  float uy = rz * fx - rx * fz;
  float uz = rx * fy - ry * fx;

  // Vector from camera to point
  float vx = px - c->pos.s[0];
  float vy = py - c->pos.s[1];
  float vz = pz - c->pos.s[2];

  // Project onto camera basis
  float depth = vx * fx + vy * fy + vz * fz;
  if (depth <= 0.1f) return 0;  // Behind camera

  float local_x = vx * rx + vy * ry + vz * rz;
  float local_y = vx * ux + vy * uy + vz * uz;

  // Perspective projection - matches kernel
  float scale = tanf(c->fov * 0.5f * M_PI / 180.0f);
  float aspect = (float)data->scene.width / data->scene.height;

  float ndc_x = (local_x / depth) / (aspect * scale);
  float ndc_y = (local_y / depth) / scale;

  *sx = (int)((ndc_x * 0.5f + 0.5f) * data->scene.width);
  *sy = (int)((0.5f - ndc_y * 0.5f) * data->scene.height);

  return 1;
}

// Draw 3D axis at world origin
static void draw_world_axis(t_data *data, float ox, float oy, float oz, float length) {
  int cx, cy;
  int ex, ey;

  // Origin
  if (!project_point(data, ox, oy, oz, &cx, &cy)) return;

  // X axis (red)
  if (project_point(data, ox + length, oy, oz, &ex, &ey)) {
    draw_line(data, cx, cy, ex, ey, 0xFF0000);
    draw_string(data, ex + 2, ey - 3, "X", 0xFF0000);
  }

  // Y axis (green)
  if (project_point(data, ox, oy + length, oz, &ex, &ey)) {
    draw_line(data, cx, cy, ex, ey, 0x00FF00);
    draw_string(data, ex + 2, ey - 3, "Y", 0x00FF00);
  }

  // Z axis (blue)
  if (project_point(data, ox, oy, oz + length, &ex, &ey)) {
    draw_line(data, cx, cy, ex, ey, 0x0000FF);
    draw_string(data, ex + 2, ey - 3, "Z", 0x0000FF);
  }
}

// Draw normal vector at a point
static void draw_normal(t_data *data, float px, float py, float pz,
                        float nx, float ny, float nz, float length, int color) {
  int sx0, sy0, sx1, sy1;
  if (!project_point(data, px, py, pz, &sx0, &sy0)) return;
  if (!project_point(data, px + nx * length, py + ny * length, pz + nz * length, &sx1, &sy1)) return;
  draw_line(data, sx0, sy0, sx1, sy1, color);
}

// Draw 3D box wireframe
static void draw_box_wireframe(t_data *data, float cx, float cy, float cz,
                               float hx, float hy, float hz, int color) {
  // 8 corners of the box
  float corners[8][3] = {
    {cx - hx, cy - hy, cz - hz}, {cx + hx, cy - hy, cz - hz},
    {cx + hx, cy + hy, cz - hz}, {cx - hx, cy + hy, cz - hz},
    {cx - hx, cy - hy, cz + hz}, {cx + hx, cy - hy, cz + hz},
    {cx + hx, cy + hy, cz + hz}, {cx - hx, cy + hy, cz + hz}
  };

  int screen[8][2];
  int valid[8];
  for (int i = 0; i < 8; i++) {
    valid[i] = project_point(data, corners[i][0], corners[i][1], corners[i][2],
                             &screen[i][0], &screen[i][1]);
  }

  // 12 edges of the box
  int edges[12][2] = {
    {0,1}, {1,2}, {2,3}, {3,0},  // Front face
    {4,5}, {5,6}, {6,7}, {7,4},  // Back face
    {0,4}, {1,5}, {2,6}, {3,7}   // Connecting edges
  };

  for (int i = 0; i < 12; i++) {
    int a = edges[i][0], b = edges[i][1];
    if (valid[a] && valid[b]) {
      draw_line(data, screen[a][0], screen[a][1], screen[b][0], screen[b][1], color);
    }
  }
}

// Get object type name
static const char *get_object_type_name(int type) {
  switch (type) {
    case TYPE_SPHERE: return "Sphere";
    case TYPE_PLANE: return "Plane";
    case TYPE_QUADRIC: return "Quadric";
    case TYPE_BOX: return "Box";
    case TYPE_TORUS: return "Torus";
    case TYPE_MOBIUS: return "Mobius";
    default: return "Unknown";
  }
}

// Draw debug overlay
static void draw_debug_overlay(t_data *data) {
  if (!g_debug_mode) return;

  char buf[128];
  int y = 10;
  int color_white = 0xFFFFFF;
  int color_yellow = 0xFFFF00;
  int color_green = 0x00FF00;
  int color_cyan = 0x00FFFF;
  int color_magenta = 0xFF00FF;

  // FPS and frame time
  snprintf(buf, sizeof(buf), "FPS: %.1f (%.2f ms)", g_fps, g_frame_time_ms);
  draw_string(data, 10, y, buf, color_green);
  y += 10;

  // Resolution and ray count
  int ray_count = data->scene.width * data->scene.height * data->scene.render.max_bounces;
  snprintf(buf, sizeof(buf), "%dx%d  Rays: %d", data->scene.width, data->scene.height, ray_count);
  draw_string(data, 10, y, buf, color_white);
  y += 10;

  // Camera info
  t_camera *c = &data->scene.camera;
  snprintf(buf, sizeof(buf), "Cam: (%.1f %.1f %.1f) FOV:%.0f",
           c->pos.s[0], c->pos.s[1], c->pos.s[2], c->fov);
  draw_string(data, 10, y, buf, color_cyan);
  y += 10;

  // Scene info
  snprintf(buf, sizeof(buf), "Obj: %d  Lights: %d  BVH: %d",
           data->scene.obj_count, data->scene.light_count, data->scene.bvh_node_count);
  draw_string(data, 10, y, buf, color_yellow);
  y += 10;

  // Selected object info
  y += 5;
  if (g_selected_object >= 0 && g_selected_object < data->scene.obj_count) {
    t_object *obj = &data->scene.objects[g_selected_object];

    draw_string(data, 10, y, "--- SELECTED OBJECT ---", color_magenta);
    y += 10;

    snprintf(buf, sizeof(buf), "ID: %d  Type: %s", g_selected_object, get_object_type_name(obj->type));
    draw_string(data, 10, y, buf, color_magenta);
    y += 10;

    snprintf(buf, sizeof(buf), "Pos: %.2f %.2f %.2f", obj->pos.s[0], obj->pos.s[1], obj->pos.s[2]);
    draw_string(data, 10, y, buf, color_white);
    y += 10;

    snprintf(buf, sizeof(buf), "Distance: %.2f", g_selected_distance);
    draw_string(data, 10, y, buf, color_white);
    y += 10;

    // Hit point and normal
    snprintf(buf, sizeof(buf), "Hit: %.2f %.2f %.2f", g_hit_point[0], g_hit_point[1], g_hit_point[2]);
    draw_string(data, 10, y, buf, color_white);
    y += 10;

    snprintf(buf, sizeof(buf), "Normal: %.2f %.2f %.2f", g_hit_normal[0], g_hit_normal[1], g_hit_normal[2]);
    draw_string(data, 10, y, buf, color_cyan);
    y += 10;

    // Type-specific properties
    switch (obj->type) {
      case TYPE_SPHERE:
        snprintf(buf, sizeof(buf), "Radius: %.2f", obj->sphere.radius);
        draw_string(data, 10, y, buf, color_white);
        y += 10;
        draw_box_wireframe(data, obj->pos.s[0], obj->pos.s[1], obj->pos.s[2],
                           obj->sphere.radius, obj->sphere.radius, obj->sphere.radius,
                           0xFFFF00);
        break;
      case TYPE_BOX:
        snprintf(buf, sizeof(buf), "Size: %.2f x %.2f x %.2f",
                 obj->box.half_size.s[0]*2, obj->box.half_size.s[1]*2, obj->box.half_size.s[2]*2);
        draw_string(data, 10, y, buf, color_white);
        y += 10;
        // Draw bounding box wireframe
        draw_box_wireframe(data, obj->pos.s[0], obj->pos.s[1], obj->pos.s[2],
                           obj->box.half_size.s[0], obj->box.half_size.s[1], obj->box.half_size.s[2],
                           0xFFFF00);
        break;
      case TYPE_PLANE:
        snprintf(buf, sizeof(buf), "Stored Normal: %.2f %.2f %.2f",
                 obj->plane.normal.s[0], obj->plane.normal.s[1], obj->plane.normal.s[2]);
        draw_string(data, 10, y, buf, color_white);
        y += 10;
        break;
      case TYPE_TORUS:
        snprintf(buf, sizeof(buf), "Major: %.2f  Minor: %.2f", obj->torus.major, obj->torus.minor);
        draw_string(data, 10, y, buf, color_white);
        y += 10;
        break;
      case TYPE_MOBIUS:
        snprintf(buf, sizeof(buf), "Radius: %.2f  Width: %.2f", obj->mobius.radius, obj->mobius.width);
        draw_string(data, 10, y, buf, color_white);
        y += 10;
        break;
      default:
        break;
    }

    // Draw normal vector at hit point
    draw_normal(data, g_hit_point[0], g_hit_point[1], g_hit_point[2],
                g_hit_normal[0], g_hit_normal[1], g_hit_normal[2], 1.0f, 0x00FFFF);

    // Material properties
    y += 5;
    draw_string(data, 10, y, "Material:", color_yellow);
    y += 10;

    snprintf(buf, sizeof(buf), "Color: %.2f %.2f %.2f",
             obj->mat.color.s[0], obj->mat.color.s[1], obj->mat.color.s[2]);
    draw_string(data, 10, y, buf, color_white);
    y += 10;

    snprintf(buf, sizeof(buf), "Refl: %.2f  Trans: %.2f  IOR: %.2f",
             obj->mat.reflection, obj->mat.transparency, obj->mat.ior);
    draw_string(data, 10, y, buf, color_white);
    y += 10;

    snprintf(buf, sizeof(buf), "Rough: %.2f  Metal: %.2f", obj->mat.roughness, obj->mat.metallic);
    draw_string(data, 10, y, buf, color_white);
    y += 10;
  } else {
    draw_string(data, 10, y, "Click on object to select", 0x888888);
    y += 10;
  }

  // Controls help
  y += 5;
  if (g_selected_object >= 0) {
    draw_string(data, 10, y, "WASD/Space/Shift: Move object  ESC: Deselect", 0x666666);
  } else {
    draw_string(data, 10, y, "G/F3: Toggle  Click: Select  Scroll: FOV", 0x666666);
  }

  // Draw world axis at origin
  draw_world_axis(data, 0, 0, 0, 2.0f);

  // Draw axis at selected object's position
  if (g_selected_object >= 0 && g_selected_object < data->scene.obj_count) {
    t_object *obj = &data->scene.objects[g_selected_object];
    draw_world_axis(data, obj->pos.s[0], obj->pos.s[1], obj->pos.s[2], 1.0f);
  }
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

// Camera rotation only (arrow keys) - used when object is selected
static void update_camera_rotation(t_data *data) {
  t_camera *c = &data->scene.camera;
  float r_spd = 0.05f;

  // Yaw (Left/Right Arrows)
  if (data->keys & (KEY_MASK_LEFT | KEY_MASK_RIGHT)) {
    float angle = (data->keys & KEY_MASK_RIGHT) ? r_spd : -r_spd;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);
    float ox = c->dir.s[0];
    float oz = c->dir.s[2];
    c->dir.s[0] = ox * cos_a - oz * sin_a;
    c->dir.s[2] = ox * sin_a + oz * cos_a;
  }

  // Pitch (Up/Down Arrows)
  if (data->keys & (KEY_MASK_UP | KEY_MASK_DOWN)) {
    float angle = (data->keys & KEY_MASK_UP) ? r_spd : -r_spd;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);
    float oy = c->dir.s[1];
    float oz = sqrtf(c->dir.s[0] * c->dir.s[0] + c->dir.s[2] * c->dir.s[2]);
    c->dir.s[1] = oy * cos_a + oz * sin_a;
    float ratio = sqrtf(1.0f - c->dir.s[1] * c->dir.s[1]) / oz;
    c->dir.s[0] *= ratio;
    c->dir.s[2] *= ratio;
  }
}

// Move selected object using same keys as camera movement
static void update_selected_object(t_data *data) {
  if (!g_debug_mode || g_selected_object < 0 || g_selected_object >= data->scene.obj_count)
    return;

  t_object *obj = &data->scene.objects[g_selected_object];
  t_camera *c = &data->scene.camera;
  float m_spd = 0.1f;

  // Forward vector (from camera dir)
  float fx = c->dir.s[0];
  float fy = c->dir.s[1];
  float fz = c->dir.s[2];
  float fmag = sqrtf(fx*fx + fy*fy + fz*fz);
  if (fmag > 0) { fx /= fmag; fy /= fmag; fz /= fmag; }

  // Right vector (forward x world_up)
  float rx = -fz;
  float rz = fx;
  float rmag = sqrtf(rx*rx + rz*rz);
  if (rmag > 0) { rx /= rmag; rz /= rmag; }

  bool moved = false;

  // Movement logic (WASD + vertical) - moves object relative to camera view
  if (data->keys & KEY_MASK_W) {
    obj->pos.s[0] += fx * m_spd;
    obj->pos.s[1] += fy * m_spd;
    obj->pos.s[2] += fz * m_spd;
    moved = true;
  }
  if (data->keys & KEY_MASK_S) {
    obj->pos.s[0] -= fx * m_spd;
    obj->pos.s[1] -= fy * m_spd;
    obj->pos.s[2] -= fz * m_spd;
    moved = true;
  }
  if (data->keys & KEY_MASK_A) {
    obj->pos.s[0] += rx * m_spd;
    obj->pos.s[2] += rz * m_spd;
    moved = true;
  }
  if (data->keys & KEY_MASK_D) {
    obj->pos.s[0] -= rx * m_spd;
    obj->pos.s[2] -= rz * m_spd;
    moved = true;
  }
  if (data->keys & KEY_MASK_SP) {  // Move up
    obj->pos.s[1] += m_spd;
    moved = true;
  }
  if (data->keys & KEY_MASK_SH) {  // Move down
    obj->pos.s[1] -= m_spd;
    moved = true;
  }

  // If object moved, update GPU buffer and rebuild BVH
  if (moved) {
    data->should_render = true;
    // Update the object in GPU buffer
    cl_int err = clEnqueueWriteBuffer(data->cl.queue, data->cl.obj_buffer, CL_FALSE,
                                      sizeof(t_object) * g_selected_object, sizeof(t_object),
                                      obj, 0, NULL, NULL);
    (void)err;
  }
}


// Helper to fill a horizontal span using AVX2
// labeled "always_inline" to ensure the compiler embeds it directly
static inline __attribute__((always_inline)) void avx_memset_row(
  int *dest, 
  int x_start, 
  int x_end, 
  __m256i vec_color, 
  int scalar_color
) {
  int x = x_start;

  // Main AVX Loop (32 bytes / 8 pixels at a time)
  for (; x + 8 <= x_end; x += 8) {
    _mm256_storeu_si256((__m256i *)(dest + x), vec_color);
  }

  // Cleanup Scalar Loop (remaining pixels)
  for (; x < x_end; x++) {
    dest[x] = scalar_color;
  }
}

static void draw_progress_bar_color(t_data *data, float pct, int color_mode) {
  // --- 1. Calculate Geometry Early ---
  const int width = data->scene.width;
  const int bar_width = width - 40;

  // Safety: Handle minimized window or invalid geometry
  if (bar_width <= 0) return;

  const int filled_width = (int)(bar_width * pct);

  // --- 2. Optimization: Check Previous State ---
  // specific to this function instance
  static int prev_filled_width = -1;
  static int prev_color_mode = -1;
  static int prev_scene_width = -1;
  static void *prev_img_addr = NULL;

  // If NOTHING visual has changed, skip the draw entirely
  if (filled_width == prev_filled_width && 
    color_mode == prev_color_mode && 
    width == prev_scene_width &&
    data->img_addr == prev_img_addr) {
    return; 
  }

  // --- 3. Update State for Next Time ---
  prev_filled_width = filled_width;
  prev_color_mode = color_mode;
  prev_scene_width = width;
  prev_img_addr = data->img_addr;

  // --- 4. Drawing Logic (Same as before) ---
  const int height = data->scene.height;
  const int y_base = height - 30;
  const int y_start = (y_base < 0) ? 0 : y_base;
  const int y_end = (y_base + 6 > height) ? height : y_base + 6;

  if (y_start >= y_end) return; 

  const int x_start = 20;
  const int x_end = (20 + bar_width > width) ? width : 20 + bar_width;
  // Clamp filled_x_end to ensure it doesn't exceed the bar's container
  const int filled_x_end = (20 + filled_width > x_end) ? x_end : 20 + filled_width;

  const int color_hex = (color_mode == 1) ? 0x0066CC : 0x00AA00;
  const __m256i border_vec = _mm256_set1_epi32(0x444444);
  const __m256i fill_vec   = _mm256_set1_epi32(color_hex);
  const __m256i unfill_vec = _mm256_set1_epi32(0x222222);

  int *line_ptr = (int *)data->img_addr + (y_start * width);

  for (int y = y_start; y < y_end; y++) {
    int relative_row = y - y_base;

    if (relative_row == 0 || relative_row == 5) {
      avx_memset_row(line_ptr, x_start, x_end, border_vec, 0x444444);
    } else {
      avx_memset_row(line_ptr, x_start, filled_x_end, fill_vec, color_hex);
      avx_memset_row(line_ptr, filled_x_end, x_end, unfill_vec, 0x222222);
    }
    line_ptr += width;
  }
}

// Draw progress bar during loading (color: 0=green loading, 1=blue BVH)
static void draw_progress_bar(t_data *data) {
  if (data->load_total <= 0) return;
  float pct = (float)data->load_current / data->load_total;
  draw_progress_bar_color(data, pct, 0);
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
      // int saved_bvh = data->scene.bvh_node_count;
      // data->scene.bvh_node_count = 0;
      // render_opencl(data);
      // data->scene.bvh_node_count = saved_bvh;

      // update_mlx_image(data);
      draw_progress_bar(data);
      // update_fps(0);
    } else {
      // Loading complete - build BVH and switch to accelerated rendering
      printf("Loading complete. Building BVH (%d objects, %d lights)...\n",
             data->scene.obj_count, data->scene.light_count);
      memset(data->img_addr, 0, sizeof(uint32_t) * data->scene.width * data->scene.height);
      mlx_put_image_to_window(data->mlx, data->win, data->img, 0, 0);

      build_bvh(&data->scene);
      // debug_print_bvh(&data->scene);
      upload_bvh(data);

      printf("BVH: %d nodes\n", data->scene.bvh_node_count);

      incremental_parser_cleanup(g_parser);
      g_parser = NULL;
      data->loading = false;

      // Final render with BVH
      double t0 = get_time_ms();
      render_opencl(data);
      double render_time = get_time_ms() - t0;
      update_mlx_image(data);
      draw_debug_overlay(data);
      update_fps(render_time);
    }
  } else {
    // Normal rendering mode
    check_for_scene_updates(data);
    if (data->keys || data->should_render || g_debug_mode) {
      data->should_render = 0;
      // Move selected object or camera based on debug mode state
      if (g_debug_mode && g_selected_object >= 0) {
        update_selected_object(data);
        // Still allow camera rotation with arrow keys while moving object
        update_camera_rotation(data);
      } else {
        update_camera(data);
      }

      double t0 = get_time_ms();
      render_opencl(data);
      double render_time = get_time_ms() - t0;

      update_mlx_image(data);
      draw_debug_overlay(data);
      ui_render(&g_ui);
      ui_update_mouse_state(&g_ui);
      update_fps(render_time);
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

  // Mouse hooks for object selection and UI interaction
  mlx_hook(data.win, ButtonPress, ButtonPressMask, (t_fn)(intptr_t)mouse_button, &data);
  mlx_hook(data.win, ButtonRelease, ButtonReleaseMask, (t_fn)(intptr_t)mouse_release, &data);
  mlx_hook(data.win, MotionNotify, PointerMotionMask, (t_fn)(intptr_t)mouse_motion, &data);

  // Initialize UI system
  init_ui_system(&data);

  mlx_loop_hook(data.mlx, (t_fn)(intptr_t)render_loop, &data);
  mlx_loop(data.mlx);

  return 0;
}
