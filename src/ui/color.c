#include "ui.h"
#include <math.h>
#include <stdint.h>

// ============================================================================
// Color Channel Getters
// ============================================================================

inline uint8_t color_get_a(uint32_t c) { return ((c >> 24) & 0xff); }
inline uint8_t color_get_r(uint32_t c) { return ((c >> 16) & 0xff); }
inline uint8_t color_get_g(uint32_t c) { return ((c >> 8) & 0xff); }
inline uint8_t color_get_b(uint32_t c) { return (c & 0xff); }

// ============================================================================
// Color Constructors
// ============================================================================

uint32_t color_from_rgb(uint8_t r, uint8_t g, uint8_t b) {
  return ((uint32_t)((r << 16) | (g << 8) | b));
}

uint32_t color_from_argb(uint8_t a, uint8_t r, uint8_t g, uint8_t b) {
  return ((uint32_t)((a << 24) | (r << 16) | (g << 8) | b));
}

// ============================================================================
// HSV to RGB Conversion
// ============================================================================

static inline uint8_t add_m(float v, float m) {
  return ((uint8_t)((v + m) * 255));
}

static inline void hsv_convert(float h, float c, float x, float m,
                               uint8_t *r, uint8_t *g, uint8_t *b) {
  if (h < 60) {
    *r = add_m(c, m); *g = add_m(x, m); *b = add_m(0, m);
  } else if (h < 120) {
    *r = add_m(x, m); *g = add_m(c, m); *b = add_m(0, m);
  } else if (h < 180) {
    *r = add_m(0, m); *g = add_m(c, m); *b = add_m(x, m);
  } else if (h < 240) {
    *r = add_m(0, m); *g = add_m(x, m); *b = add_m(c, m);
  } else if (h < 300) {
    *r = add_m(x, m); *g = add_m(0, m); *b = add_m(c, m);
  } else {
    *r = add_m(c, m); *g = add_m(0, m); *b = add_m(x, m);
  }
}

uint32_t color_from_hsv(float h, float s, float v) {
  const float c = v * s;
  const float x = c * (1 - fabsf(fmodf(h / 60.0f, 2) - 1));
  const float m = v - c;
  uint8_t r, g, b;
  hsv_convert(h, c, x, m, &r, &g, &b);
  return ((uint32_t)(r << 16) | (uint32_t)(g << 8) | b);
}

// ============================================================================
// Color Blending (Alpha)
// ============================================================================

inline uint32_t color_blend(uint32_t fg, uint32_t bg) {
  const uint32_t inv_a = color_get_a(fg) + 1;
  const uint32_t a = 256 - inv_a;
  if (inv_a == 1) return fg;
  if (inv_a == 256) return bg;
  uint8_t r = (uint8_t)((color_get_r(fg) * a + color_get_r(bg) * inv_a) >> 8);
  uint8_t g = (uint8_t)((color_get_g(fg) * a + color_get_g(bg) * inv_a) >> 8);
  uint8_t b = (uint8_t)((color_get_b(fg) * a + color_get_b(bg) * inv_a) >> 8);
  return color_from_rgb(r, g, b);
}

// ============================================================================
// Color Linear Interpolation
// ============================================================================

inline uint32_t color_lerp(uint32_t c0, uint32_t c1, float t) {
  uint8_t r = color_get_r(c0);
  uint8_t g = color_get_g(c0);
  uint8_t b = color_get_b(c0);
  r = (uint8_t)(r + (color_get_r(c1) - r) * t);
  g = (uint8_t)(g + (color_get_g(c1) - g) * t);
  b = (uint8_t)(b + (color_get_b(c1) - b) * t);
  return color_from_rgb(r, g, b);
}
