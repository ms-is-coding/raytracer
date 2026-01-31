#include "ui.h"
#include <stdint.h>

// HSV color picker with hue strip and saturation/value square

static bool is_in_hue_strip(t_ui_ctx *ui, t_ui_color_picker *cp) {
  const int32_t mx = ui->get_mouse_x(ui->data);
  const int32_t my = ui->get_mouse_y(ui->data);
  const int32_t strip_w = 20;
  const int32_t strip_x = cp->base.x + cp->base.w - strip_w - 4;

  return (mx >= strip_x && mx < strip_x + strip_w &&
          my >= cp->base.y + 4 && my < cp->base.y + cp->base.h - 4);
}

static bool is_in_sv_area(t_ui_ctx *ui, t_ui_color_picker *cp) {
  const int32_t mx = ui->get_mouse_x(ui->data);
  const int32_t my = ui->get_mouse_y(ui->data);
  const int32_t sv_w = cp->base.w - 28;
  const int32_t sv_h = cp->base.h - 8;

  return (mx >= cp->base.x + 4 && mx < cp->base.x + 4 + sv_w &&
          my >= cp->base.y + 4 && my < cp->base.y + 4 + sv_h);
}

static void draw_hue_strip(t_ui_ctx *ui, t_ui_color_picker *cp) {
  const int32_t strip_w = 20;
  const int32_t strip_x = cp->base.x + cp->base.w - strip_w - 4;
  const int32_t strip_y = cp->base.y + 4;
  const int32_t strip_h = cp->base.h - 8;

  for (int32_t y = 0; y < strip_h; y++) {
    float hue = (float)y / (float)strip_h * 360.0f;
    uint32_t color = color_from_hsv(hue, 1.0f, 1.0f);

    for (int32_t x = 0; x < strip_w; x++) {
      ui->put_pixel(ui->data, strip_x + x, strip_y + y, color);
    }
  }

  // Draw hue indicator
  int32_t hue_y = strip_y + (int32_t)(cp->hue / 360.0f * strip_h);
  for (int32_t x = 0; x < strip_w; x++) {
    ui->put_pixel(ui->data, strip_x + x, hue_y, 0xFFFFFF);
    if (hue_y > strip_y)
      ui->put_pixel(ui->data, strip_x + x, hue_y - 1, 0x000000);
    if (hue_y < strip_y + strip_h - 1)
      ui->put_pixel(ui->data, strip_x + x, hue_y + 1, 0x000000);
  }
}

static void draw_sv_area(t_ui_ctx *ui, t_ui_color_picker *cp) {
  const int32_t sv_x = cp->base.x + 4;
  const int32_t sv_y = cp->base.y + 4;
  const int32_t sv_w = cp->base.w - 28;
  const int32_t sv_h = cp->base.h - 8;

  for (int32_t y = 0; y < sv_h; y++) {
    float v = 1.0f - (float)y / (float)sv_h;
    for (int32_t x = 0; x < sv_w; x++) {
      float s = (float)x / (float)sv_w;
      uint32_t color = color_from_hsv(cp->hue, s, v);
      ui->put_pixel(ui->data, sv_x + x, sv_y + y, color);
    }
  }

  // Draw crosshair at current position
  int32_t cx = sv_x + (int32_t)(cp->saturation * sv_w);
  int32_t cy = sv_y + (int32_t)((1.0f - cp->brightness) * sv_h);

  // Horizontal line
  for (int32_t x = cx - 4; x <= cx + 4; x++) {
    if (x >= sv_x && x < sv_x + sv_w && x != cx) {
      ui->put_pixel(ui->data, x, cy, 0xFFFFFF);
    }
  }
  // Vertical line
  for (int32_t y = cy - 4; y <= cy + 4; y++) {
    if (y >= sv_y && y < sv_y + sv_h && y != cy) {
      ui->put_pixel(ui->data, cx, y, 0xFFFFFF);
    }
  }
}

void ui_color_picker(t_ui_ctx *ui, t_ui_color_picker *cp) {
  const bool pressed = ui->mouse_pressed(ui->data);
  const bool just_pressed = pressed && !ui->prev_mouse_down;

  // Start dragging
  if (just_pressed) {
    if (is_in_hue_strip(ui, cp)) {
      cp->dragging_hue = true;
    } else if (is_in_sv_area(ui, cp)) {
      cp->dragging_sv = true;
    }
  }

  // Update while dragging
  if (cp->dragging_hue) {
    if (pressed) {
      const int32_t my = ui->get_mouse_y(ui->data);
      const int32_t strip_y = cp->base.y + 4;
      const int32_t strip_h = cp->base.h - 8;
      float t = (float)(my - strip_y) / (float)strip_h;
      if (t < 0.0f) t = 0.0f;
      if (t > 1.0f) t = 1.0f;
      cp->hue = t * 360.0f;
    } else {
      cp->dragging_hue = false;
    }
  }

  if (cp->dragging_sv) {
    if (pressed) {
      const int32_t mx = ui->get_mouse_x(ui->data);
      const int32_t my = ui->get_mouse_y(ui->data);
      const int32_t sv_x = cp->base.x + 4;
      const int32_t sv_y = cp->base.y + 4;
      const int32_t sv_w = cp->base.w - 28;
      const int32_t sv_h = cp->base.h - 8;

      float s = (float)(mx - sv_x) / (float)sv_w;
      float v = 1.0f - (float)(my - sv_y) / (float)sv_h;

      if (s < 0.0f) s = 0.0f;
      if (s > 1.0f) s = 1.0f;
      if (v < 0.0f) v = 0.0f;
      if (v > 1.0f) v = 1.0f;

      cp->saturation = s;
      cp->brightness = v;
    } else {
      cp->dragging_sv = false;
    }
  }

  // Update output color
  if (cp->value) {
    *cp->value = color_from_hsv(cp->hue, cp->saturation, cp->brightness);
  }

  // Draw background
  t_ui_base bg = cp->base;
  bg.color = 0x303030;
  ui_rect(ui, &bg);

  // Draw components
  draw_sv_area(ui, cp);
  draw_hue_strip(ui, cp);
}
