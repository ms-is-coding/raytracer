#ifndef UI_H
#define UI_H

#include <stdbool.h>
#include <stdint.h>

// ============================================================================
// Color Utilities
// ============================================================================

uint8_t  color_get_a(uint32_t c);
uint8_t  color_get_r(uint32_t c);
uint8_t  color_get_g(uint32_t c);
uint8_t  color_get_b(uint32_t c);

uint32_t color_from_rgb(uint8_t r, uint8_t g, uint8_t b);
uint32_t color_from_argb(uint8_t a, uint8_t r, uint8_t g, uint8_t b);
uint32_t color_from_hsv(float h, float s, float v);

uint32_t color_blend(uint32_t fg, uint32_t bg);
uint32_t color_lerp(uint32_t c0, uint32_t c1, float t);

// ============================================================================
// Inline Helpers
// ============================================================================

static inline uint32_t lerp_channel(uint8_t shift, uint32_t a, uint32_t b, float t) {
  const uint8_t ca = (a >> shift) & 0xff;
  const uint8_t cb = (b >> shift) & 0xff;
  return ((ca + (uint32_t)((cb - ca) * t)) << shift);
}

static inline uint32_t lerp_u32(uint32_t a, uint32_t b, float t) {
  return (lerp_channel(0, a, b, t) | lerp_channel(8, a, b, t) | lerp_channel(16, a, b, t));
}

// ============================================================================
// UI Context - Callback-based renderer abstraction
// ============================================================================

typedef void     (*t_ui_put_pixel_fn)(void *data, int32_t x, int32_t y, uint32_t color);
typedef uint32_t (*t_ui_get_color_fn)(void *data, int32_t x, int32_t y);
typedef int32_t  (*t_ui_generic_fn)(void *data);

typedef struct s_ui_ctx {
  t_ui_put_pixel_fn put_pixel;
  t_ui_generic_fn   get_mouse_x;
  t_ui_generic_fn   get_mouse_y;
  t_ui_get_color_fn get_color;
  t_ui_generic_fn   mouse_pressed;
  void             *data;
  bool              prev_mouse_down;
  int32_t           width;
  int32_t           height;
} t_ui_ctx;

// ============================================================================
// Component Types
// ============================================================================

typedef enum e_ui_component_type {
  UI_COMPONENT_NONE,
  UI_COMPONENT_RECT,
  UI_COMPONENT_SWITCH,
  UI_COMPONENT_SLIDER,
  UI_COMPONENT_BUTTON,
  UI_COMPONENT_LABEL,
  UI_COMPONENT_PANEL,
  UI_COMPONENT_COLOR_PICKER,
} t_ui_component_type;

// ============================================================================
// Base Component Properties
// ============================================================================

typedef struct s_ui_base {
  int32_t  x;
  int32_t  y;
  int32_t  w;
  int32_t  h;
  uint32_t color;
  struct {
    int32_t  radius;
    int32_t  width;
    uint32_t color;
  } border;
} t_ui_base;

typedef struct s_ui_animation {
  float state;
  float velocity;
} t_ui_animation;

// ============================================================================
// Rectangle Component
// ============================================================================

typedef t_ui_base t_ui_rect;

void ui_rect(t_ui_ctx *ctx, t_ui_rect *rect);

// ============================================================================
// Switch (Toggle) Component
// ============================================================================

typedef struct s_ui_switch {
  bool          *value;
  t_ui_base      base;
  t_ui_animation anim;
  uint32_t       color;
  uint32_t       active_color;
  struct {
    uint32_t color;
    uint32_t active_color;
  } knob;
  bool           clicked;
} t_ui_switch;

void ui_switch(t_ui_ctx *ctx, t_ui_switch *sw);

// ============================================================================
// Slider Component
// ============================================================================

typedef struct s_ui_slider {
  float         *value;       // Pointer to float value (0.0 to 1.0)
  t_ui_base      base;
  t_ui_animation anim;
  uint32_t       track_color;
  uint32_t       fill_color;
  uint32_t       knob_color;
  int32_t        knob_radius;
  bool           dragging;
} t_ui_slider;

void ui_slider(t_ui_ctx *ctx, t_ui_slider *slider);

// ============================================================================
// Button Component
// ============================================================================

typedef struct s_ui_button {
  const char    *text;
  t_ui_base      base;
  t_ui_animation anim;
  uint32_t       color;
  uint32_t       hover_color;
  uint32_t       press_color;
  uint32_t       text_color;
  bool           clicked;     // True on frame when clicked
  bool           held;        // Currently being held
  bool           hovered;
} t_ui_button;

void ui_button(t_ui_ctx *ctx, t_ui_button *btn);

// ============================================================================
// Label Component
// ============================================================================

typedef struct s_ui_label {
  const char *text;
  int32_t     x;
  int32_t     y;
  uint32_t    color;
  int32_t     scale;  // 1 = normal, 2 = double size
} t_ui_label;

void ui_label(t_ui_ctx *ctx, t_ui_label *label);

// ============================================================================
// Panel Component (Container)
// ============================================================================

typedef struct s_ui_panel {
  const char *title;
  t_ui_base   base;
  uint32_t    bg_color;
  uint32_t    title_color;
  uint32_t    border_color;
  bool        collapsed;
  bool        draggable;
  int32_t     drag_offset_x;
  int32_t     drag_offset_y;
  bool        dragging;
} t_ui_panel;

void ui_panel(t_ui_ctx *ctx, t_ui_panel *panel);

// ============================================================================
// Color Picker Component
// ============================================================================

typedef struct s_ui_color_picker {
  uint32_t *value;
  t_ui_base base;
  float     hue;
  float     saturation;
  float     brightness;
  bool      dragging_hue;
  bool      dragging_sv;
} t_ui_color_picker;

void ui_color_picker(t_ui_ctx *ctx, t_ui_color_picker *cp);

// ============================================================================
// Component Union (for generic handling)
// ============================================================================

typedef struct s_ui_component {
  t_ui_component_type type;
  union {
    t_ui_rect         rect;
    t_ui_switch       switch_;
    t_ui_slider       slider;
    t_ui_button       button;
    t_ui_label        label;
    t_ui_panel        panel;
    t_ui_color_picker color_picker;
  } as;
} t_ui_component;

// ============================================================================
// UI System Management
// ============================================================================

#define UI_MAX_COMPONENTS 64

typedef struct s_ui_system {
  t_ui_ctx       ctx;
  t_ui_component components[UI_MAX_COMPONENTS];
  int32_t        component_count;
} t_ui_system;

void ui_init(t_ui_system *ui, void *render_data, int32_t width, int32_t height);
int  ui_add_component(t_ui_system *ui, t_ui_component component);
void ui_render(t_ui_system *ui);
void ui_update_mouse_state(t_ui_system *ui);

// ============================================================================
// Convenience Macros for Creating Components
// ============================================================================

#define UI_SWITCH(val_ptr, x_, y_, w_, h_) \
  (t_ui_component){ \
    .type = UI_COMPONENT_SWITCH, \
    .as.switch_ = { \
      .value = (val_ptr), \
      .base = {.x = (x_), .y = (y_), .w = (w_), .h = (h_), .border = {.radius = (h_)/3}}, \
      .anim = {0, 0.15f}, \
      .color = 0x505050, \
      .active_color = 0x00AA00, \
      .knob = {.color = 0xCCCCCC, .active_color = 0xFFFFFF} \
    } \
  }

#define UI_SLIDER(val_ptr, x_, y_, w_, h_) \
  (t_ui_component){ \
    .type = UI_COMPONENT_SLIDER, \
    .as.slider = { \
      .value = (val_ptr), \
      .base = {.x = (x_), .y = (y_), .w = (w_), .h = (h_), .border = {.radius = (h_)/2}}, \
      .anim = {0, 0.2f}, \
      .track_color = 0x404040, \
      .fill_color = 0x0088FF, \
      .knob_color = 0xFFFFFF, \
      .knob_radius = (h_)/2 + 2 \
    } \
  }

#define UI_BUTTON(text_, x_, y_, w_, h_) \
  (t_ui_component){ \
    .type = UI_COMPONENT_BUTTON, \
    .as.button = { \
      .text = (text_), \
      .base = {.x = (x_), .y = (y_), .w = (w_), .h = (h_), .border = {.radius = 4}}, \
      .anim = {0, 0.25f}, \
      .color = 0x404040, \
      .hover_color = 0x505050, \
      .press_color = 0x303030, \
      .text_color = 0xFFFFFF \
    } \
  }

#define UI_LABEL(text_, x_, y_, color_) \
  (t_ui_component){ \
    .type = UI_COMPONENT_LABEL, \
    .as.label = { \
      .text = (text_), \
      .x = (x_), \
      .y = (y_), \
      .color = (color_), \
      .scale = 1 \
    } \
  }

#define UI_PANEL(title_, x_, y_, w_, h_) \
  (t_ui_component){ \
    .type = UI_COMPONENT_PANEL, \
    .as.panel = { \
      .title = (title_), \
      .base = {.x = (x_), .y = (y_), .w = (w_), .h = (h_), .border = {.radius = 6}}, \
      .bg_color = 0x202020, \
      .title_color = 0xFFFFFF, \
      .border_color = 0x404040 \
    } \
  }

#endif
