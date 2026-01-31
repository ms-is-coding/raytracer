#include "ui.h"
#include <string.h>

// ============================================================================
// Pixel Access Callbacks (to be connected to render context)
// ============================================================================

// Default implementations - user should override these via ui_init

static void default_put_pixel(void *data, int32_t x, int32_t y, uint32_t color) {
  (void)data; (void)x; (void)y; (void)color;
}

static uint32_t default_get_color(void *data, int32_t x, int32_t y) {
  (void)data; (void)x; (void)y;
  return 0;
}

static int32_t default_get_mouse_x(void *data) {
  (void)data;
  return 0;
}

static int32_t default_get_mouse_y(void *data) {
  (void)data;
  return 0;
}

static int32_t default_mouse_pressed(void *data) {
  (void)data;
  return 0;
}

// ============================================================================
// UI System Initialization
// ============================================================================

void ui_init(t_ui_system *ui, void *render_data, int32_t width, int32_t height) {
  memset(ui, 0, sizeof(t_ui_system));

  ui->ctx.data = render_data;
  ui->ctx.width = width;
  ui->ctx.height = height;

  // Set default callbacks (user should override these)
  ui->ctx.put_pixel = default_put_pixel;
  ui->ctx.get_color = default_get_color;
  ui->ctx.get_mouse_x = default_get_mouse_x;
  ui->ctx.get_mouse_y = default_get_mouse_y;
  ui->ctx.mouse_pressed = default_mouse_pressed;
}

// ============================================================================
// Component Management
// ============================================================================

int ui_add_component(t_ui_system *ui, t_ui_component component) {
  if (ui->component_count >= UI_MAX_COMPONENTS) {
    return -1;
  }
  ui->components[ui->component_count] = component;
  return ui->component_count++;
}

// ============================================================================
// Rendering
// ============================================================================

void ui_render(t_ui_system *ui) {
  for (int32_t i = 0; i < ui->component_count; i++) {
    t_ui_component *c = &ui->components[i];

    switch (c->type) {
      case UI_COMPONENT_RECT:
        ui_rect(&ui->ctx, &c->as.rect);
        break;
      case UI_COMPONENT_SWITCH:
        ui_switch(&ui->ctx, &c->as.switch_);
        break;
      case UI_COMPONENT_SLIDER:
        ui_slider(&ui->ctx, &c->as.slider);
        break;
      case UI_COMPONENT_BUTTON:
        ui_button(&ui->ctx, &c->as.button);
        break;
      case UI_COMPONENT_LABEL:
        ui_label(&ui->ctx, &c->as.label);
        break;
      case UI_COMPONENT_PANEL:
        ui_panel(&ui->ctx, &c->as.panel);
        break;
      case UI_COMPONENT_COLOR_PICKER:
        ui_color_picker(&ui->ctx, &c->as.color_picker);
        break;
      case UI_COMPONENT_NONE:
      default:
        break;
    }
  }
}

void ui_update_mouse_state(t_ui_system *ui) {
  ui->ctx.prev_mouse_down = ui->ctx.mouse_pressed(ui->ctx.data);
}
