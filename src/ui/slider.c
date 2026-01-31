#include "ui.h"
#include <stdbool.h>
#include <stdint.h>

static bool is_hovering_slider(t_ui_ctx *ui, t_ui_slider *slider) {
  const int32_t mx = ui->get_mouse_x(ui->data);
  const int32_t my = ui->get_mouse_y(ui->data);

  // Extend hit area to include knob
  int32_t ext = slider->knob_radius;
  return (mx >= slider->base.x - ext && mx < slider->base.x + slider->base.w + ext &&
          my >= slider->base.y - ext && my < slider->base.y + slider->base.h + ext);
}

static float get_slider_value_from_mouse(t_ui_ctx *ui, t_ui_slider *slider) {
  const int32_t mx = ui->get_mouse_x(ui->data);
  float t = (float)(mx - slider->base.x) / (float)slider->base.w;
  if (t < 0.0f) t = 0.0f;
  if (t > 1.0f) t = 1.0f;
  return t;
}

static void draw_slider_track(t_ui_ctx *ui, t_ui_slider *slider) {
  // Track background
  t_ui_base track = {
    .x = slider->base.x,
    .y = slider->base.y + (slider->base.h - 4) / 2,
    .w = slider->base.w,
    .h = 4,
    .color = slider->track_color,
    .border = {.radius = 2}
  };
  ui_rect(ui, &track);

  // Filled portion
  int32_t fill_w = (int32_t)(slider->anim.state * slider->base.w);
  if (fill_w > 0) {
    t_ui_base fill = {
      .x = slider->base.x,
      .y = slider->base.y + (slider->base.h - 4) / 2,
      .w = fill_w,
      .h = 4,
      .color = slider->fill_color,
      .border = {.radius = 2}
    };
    ui_rect(ui, &fill);
  }
}

static void draw_slider_knob(t_ui_ctx *ui, t_ui_slider *slider) {
  int32_t knob_x = slider->base.x + (int32_t)(slider->anim.state * slider->base.w);
  int32_t knob_y = slider->base.y + slider->base.h / 2;
  int32_t r = slider->knob_radius;

  t_ui_base knob = {
    .x = knob_x - r,
    .y = knob_y - r,
    .w = r * 2,
    .h = r * 2,
    .color = slider->knob_color,
    .border = {.radius = r}
  };
  ui_rect(ui, &knob);
}

void ui_slider(t_ui_ctx *ui, t_ui_slider *slider) {
  const bool hovering = is_hovering_slider(ui, slider);
  const bool pressed = ui->mouse_pressed(ui->data);
  const bool just_pressed = pressed && !ui->prev_mouse_down;

  // Start dragging on click
  if (just_pressed && hovering) {
    slider->dragging = true;
  }

  // Update value while dragging
  if (slider->dragging) {
    if (pressed) {
      *slider->value = get_slider_value_from_mouse(ui, slider);
    } else {
      slider->dragging = false;
    }
  }

  // Animate toward target value
  slider->anim.state += (*slider->value - slider->anim.state) * slider->anim.velocity;

  // Draw components
  draw_slider_track(ui, slider);
  draw_slider_knob(ui, slider);
}
