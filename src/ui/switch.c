#include "ui.h"
#include <stdbool.h>
#include <stdint.h>

static inline bool is_inside_circle(int32_t x, int32_t y, int32_t r) {
  return (x * x + y * y <= r * r);
}

static bool is_hovering_switch(t_ui_ctx *ui, t_ui_switch *sw) {
  const int32_t mx = ui->get_mouse_x(ui->data);
  const int32_t my = ui->get_mouse_y(ui->data);
  const int32_t r = sw->base.border.radius;
  int32_t dx, dy;

  // Basic bounds check
  if (mx < sw->base.x || mx >= sw->base.x + sw->base.w ||
      my < sw->base.y || my >= sw->base.y + sw->base.h)
    return false;

  dx = mx - sw->base.x;
  dy = my - sw->base.y;

  // Check rounded corners
  if (dx < r && dy < r)
    return is_inside_circle(r - 1 - dx, r - 1 - dy, r);
  if (dx >= sw->base.w - r && dy < r)
    return is_inside_circle(dx - (sw->base.w - r), r - 1 - dy, r);
  if (dx < r && dy >= sw->base.h - r)
    return is_inside_circle(r - 1 - dx, dy - (sw->base.h - r), r);
  if (dx >= sw->base.w - r && dy >= sw->base.h - r)
    return is_inside_circle(dx - (sw->base.w - r), dy - (sw->base.h - r), r);

  return true;
}

static void draw_switch_knob(t_ui_ctx *ui, t_ui_switch *sw) {
  int32_t r = sw->base.border.radius;
  t_ui_base knob = {
    .w = sw->base.h - 4,
    .h = sw->base.h - 4,
    .x = (int32_t)((float)sw->base.x + 2 +
                   sw->anim.state * ((float)sw->base.w - (float)(sw->base.h - 1))),
    .y = sw->base.y + 2,
    .color = lerp_u32(sw->knob.color, sw->knob.active_color, sw->anim.state),
    .border = {.radius = r - 2}
  };
  ui_rect(ui, &knob);
}

void ui_switch(t_ui_ctx *ui, t_ui_switch *sw) {
  const bool hovering = is_hovering_switch(ui, sw);
  const bool pressed = ui->mouse_pressed(ui->data);
  const bool just_pressed = pressed && !ui->prev_mouse_down;
  const bool just_released = !pressed && ui->prev_mouse_down;

  // Click detection with press-and-release
  if (just_pressed && hovering)
    sw->clicked = true;
  if (just_released) {
    if (sw->clicked && hovering)
      *sw->value = !*sw->value;
    sw->clicked = false;
  }

  // Animate state toward target
  sw->anim.state += ((float)*sw->value - sw->anim.state) * sw->anim.velocity;

  // Blend colors based on animation state
  sw->base.color = lerp_u32(sw->color, sw->active_color, sw->anim.state);

  // Draw background and knob
  ui_rect(ui, &sw->base);
  draw_switch_knob(ui, sw);
}
