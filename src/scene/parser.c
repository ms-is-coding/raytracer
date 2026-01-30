#include "internal.h"
#include "scene.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
  STATE_ROOT,
  STATE_RESOLUTION,
  STATE_CAMERA,
  STATE_RENDER,
  STATE_OBJECTS,
  STATE_OBJECT,
  STATE_LIGHTS,
  STATE_LIGHT,
  STATE_MATERIALS,  // Skip materials section for now
} parser_state;

typedef struct {
  t_lexer       lex;
  t_scene       *scene;
  parser_state  state;
  int           base_indent;
  int           obj_idx;
  int           light_idx;
  int           obj_cap;
  int           light_cap;
} t_parser;

// Incremental parser state (opaque struct)
struct s_incremental_parser {
  t_parser p;
  t_file file;
  bool done;
  bool in_objects;
  bool in_lights;
};

static void parser_error(t_parser *p, yaml_token tok, const char *msg) {
  scene_error(p->lex.file, tok, msg);
}

static yaml_token expect(t_parser *p, yaml_token_type type, const char *msg) {
  yaml_token tok = lexer_next(&p->lex);
  if (tok.type != type) {
    parser_error(p, tok, msg);
  }
  return tok;
}

static void skip_newlines(t_parser *p) {
  while (lexer_peek(&p->lex).type == TOK_NEWLINE) {
    lexer_next(&p->lex);
  }
}

static double parse_number(t_parser *p) {
  yaml_token tok = lexer_next(&p->lex);
  if (tok.type != TOK_NUMBER) {
    parser_error(p, tok, "expected number");
  }
  return tok.num_val;
}

static cl_float4 parse_vector(t_parser *p) {
  cl_float4 v = {{0, 0, 0, 0}};

  expect(p, TOK_LBRACKET, "expected '['");

  yaml_token tok = lexer_next(&p->lex);
  if (tok.type != TOK_NUMBER) {
    parser_error(p, tok, "expected number");
  }
  v.s[0] = tok.num_val;

  expect(p, TOK_COMMA, "expected ','");

  tok = lexer_next(&p->lex);
  if (tok.type != TOK_NUMBER) {
    parser_error(p, tok, "expected number");
  }
  v.s[1] = tok.num_val;

  expect(p, TOK_COMMA, "expected ','");

  tok = lexer_next(&p->lex);
  if (tok.type != TOK_NUMBER) {
    parser_error(p, tok, "expected number");
  }
  v.s[2] = tok.num_val;

  expect(p, TOK_RBRACKET, "expected ']'");

  return v;
}

static cl_float4 parse_color(t_parser *p) {
  cl_float4 c = parse_vector(p);
  c.s[0] /= 255.0f;
  c.s[1] /= 255.0f;
  c.s[2] /= 255.0f;
  return c;
}

static void grow_objects(t_parser *p) {
  if (p->scene->obj_count >= p->obj_cap) {
    p->obj_cap = p->obj_cap ? p->obj_cap * 2 : 8;
    p->scene->objects = realloc(p->scene->objects,
                                p->obj_cap * sizeof(t_object));
  }
}

static void grow_lights(t_parser *p) {
  if (p->scene->light_count >= p->light_cap) {
    p->light_cap = p->light_cap ? p->light_cap * 2 : 8;
    p->scene->lights = realloc(p->scene->lights,
                               p->light_cap * sizeof(t_light));
  }
}

static int key_matches(yaml_token tok, const char *name) {
  size_t len = strlen(name);
  return tok.len == len && memcmp(tok.start, name, len) == 0;
}

static void parse_resolution(t_parser *p, yaml_token key) {
  if (key_matches(key, "width")) {
    p->scene->width = (int)parse_number(p);
  } else if (key_matches(key, "height")) {
    p->scene->height = (int)parse_number(p);
  } else if (key_matches(key, "ambient")) {
    p->scene->ambient_color = parse_color(p);
  }
}

static void parse_camera(t_parser *p, yaml_token key) {
  if (key_matches(key, "pos")) {
    if (p->scene->camera.fov < 0.01f)
      p->scene->camera.pos = parse_vector(p);
    else parse_vector(p);
  } else if (key_matches(key, "dir")) {
    if (p->scene->camera.fov < 0.01f)
    p->scene->camera.dir = parse_vector(p);
    else parse_vector(p);
  } else if (key_matches(key, "fov")) {
    if (p->scene->camera.fov < 0.01f)
    p->scene->camera.fov = parse_number(p);
    else parse_number(p);
  }
}

static void parse_render(t_parser *p, yaml_token key) {
  if (key_matches(key, "max_bounces")) {
    p->scene->render.max_bounces = (int)parse_number(p);
  } else if (key_matches(key, "gamma")) {
    p->scene->render.gamma = parse_number(p);
  } else if (key_matches(key, "exposure")) {
    p->scene->render.exposure = parse_number(p);
  } else if (key_matches(key, "epsilon")) {
    p->scene->render.epsilon = parse_number(p);
  }
}

static void new_object(t_parser *p) {
  grow_objects(p);
  p->obj_idx = p->scene->obj_count++;
  t_object *obj = &p->scene->objects[p->obj_idx];
  memset(obj, 0, sizeof(*obj));
  obj->mat.ior = 1.0f;
}

static void skip_value(t_parser *p) {
  // Skip tokens until end of line or next key
  yaml_token tok = lexer_peek(&p->lex);
  int line = tok.line;
  while (tok.type != TOK_EOF && tok.type != TOK_NEWLINE && tok.line == line) {
    lexer_next(&p->lex);
    tok = lexer_peek(&p->lex);
  }
}

static void parse_object_property(t_parser *p, yaml_token key) {
  t_object *obj = &p->scene->objects[p->obj_idx];

  if (key_matches(key, "type")) {
    yaml_token val = lexer_next(&p->lex);
    if (val.type != TOK_STRING) {
      parser_error(p, val, "expected object type");
    }
    if (val.len == 6 && memcmp(val.start, "sphere", 6) == 0) {
      obj->type = TYPE_SPHERE;
    } else if (val.len == 5 && memcmp(val.start, "plane", 5) == 0) {
      obj->type = TYPE_PLANE;
    } else if (val.len == 3 && memcmp(val.start, "box", 3) == 0) {
      obj->type = TYPE_BOX;
    } else if (val.len == 7 && memcmp(val.start, "quadric", 7) == 0) {
      obj->type = TYPE_QUADRIC;
    } else if (val.len == 8 && memcmp(val.start, "cylinder", 8) == 0) {
      //obj->type = TYPE_QUADRIC;
      //// Set default cylinder coefficients: x² + z² - r² = 0
      //memset(obj->quadric.coeffs, 0, sizeof(obj->quadric.coeffs));
      //obj->quadric.coeffs[0] = 1.0f;  // x²
      //obj->quadric.coeffs[2] = 1.0f;  // z²
      //obj->quadric.coeffs[9] = -1.0f; // -r² (default r=1)
    } else if (val.len == 4 && memcmp(val.start, "cone", 4) == 0) {
      obj->type = TYPE_QUADRIC;
      // Set default cone coefficients: x² + z² - y² = 0
      memset(obj->quadric.coeffs, 0, sizeof(obj->quadric.coeffs));
      obj->quadric.coeffs[0] = 1.0f;  // x²
      obj->quadric.coeffs[1] = -1.0f; // -y²
      obj->quadric.coeffs[2] = 1.0f;  // z²
    } else if (val.len == 5 && memcmp(val.start, "torus", 5) == 0) {
      obj->type = TYPE_TORUS;
    } else if (val.len == 6 && memcmp(val.start, "mobius", 6) == 0) {
      obj->type = TYPE_MOBIUS;
    } else {
      parser_error(p, val, "unknown object type");
    }
  } else if (key_matches(key, "pos")) {
    obj->pos = parse_vector(p);
  } else if (key_matches(key, "radius")) {
    obj->sphere.radius = parse_number(p);
  } else if (key_matches(key, "normal")) {
    obj->plane.normal = parse_vector(p);
  } else if (key_matches(key, "size")) {
    obj->box.half_size = parse_vector(p);
  } else if (key_matches(key, "major")) {
    obj->torus.major = parse_number(p);
  } else if (key_matches(key, "minor")) {
    obj->torus.minor = parse_number(p);
  } else if (key_matches(key, "width")) {
    obj->mobius.width = parse_number(p);
  } else if (key_matches(key, "color")) {
    obj->mat.color = parse_color(p);
  } else if (key_matches(key, "reflection")) {
    obj->mat.reflection = parse_number(p);
  } else if (key_matches(key, "transparency")) {
    obj->mat.transparency = parse_number(p);
  } else if (key_matches(key, "ior")) {
    obj->mat.ior = parse_number(p);
  } else if (key_matches(key, "roughness")) {
    obj->mat.roughness = parse_number(p);
  } else if (key_matches(key, "metallic")) {
    obj->mat.metallic = parse_number(p);
  } else {
    // Skip unknown properties
    skip_value(p);
  }
}

static void new_light(t_parser *p) {
  grow_lights(p);
  p->light_idx = p->scene->light_count++;
  t_light *light = &p->scene->lights[p->light_idx];
  memset(light, 0, sizeof(*light));
  light->intensity = 1.0f;
  light->color = (cl_float4){{1.0f, 1.0f, 1.0f, 0.0f}};
  light->type = LIGHT_POINT;
  light->angle = 0.5f; // ~30 degrees default for spot
}

static void parse_light_property(t_parser *p, yaml_token key) {
  t_light *light = &p->scene->lights[p->light_idx];

  if (key_matches(key, "pos")) {
    light->pos = parse_vector(p);
  } else if (key_matches(key, "dir")) {
    light->dir = parse_vector(p);
    // Normalize direction
    float len = sqrt(light->dir.s[0]*light->dir.s[0] +
                     light->dir.s[1]*light->dir.s[1] +
                     light->dir.s[2]*light->dir.s[2]);
    if (len > 0.0001f) {
      light->dir.s[0] /= len;
      light->dir.s[1] /= len;
      light->dir.s[2] /= len;
    }
  } else if (key_matches(key, "intensity")) {
    light->intensity = parse_number(p);
  } else if (key_matches(key, "color")) {
    light->color = parse_color(p);
  } else if (key_matches(key, "angle")) {
    light->angle = parse_number(p) * 3.14159265f / 180.0f; // degrees to radians
  } else if (key_matches(key, "type")) {
    yaml_token val = lexer_next(&p->lex);
    if (val.type != TOK_STRING) {
      parser_error(p, val, "expected light type");
    }
    if (val.len == 5 && memcmp(val.start, "point", 5) == 0) {
      light->type = LIGHT_POINT;
    } else if (val.len == 11 && memcmp(val.start, "directional", 11) == 0) {
      light->type = LIGHT_DIRECTIONAL;
    } else if (val.len == 4 && memcmp(val.start, "spot", 4) == 0) {
      light->type = LIGHT_SPOT;
    } else {
      parser_error(p, val, "unknown light type");
    }
  }
}

void parse_scene(const char *filename, t_scene *scene) {
  t_file file = read_file(filename);

  t_parser p = {0};
  lexer_init(&p.lex, file);
  p.scene = scene;
  p.state = STATE_ROOT;
  p.obj_idx = -1;
  p.light_idx = -1;

  // Initialize scene defaults
  scene->objects = NULL;
  scene->lights = NULL;
  scene->obj_count = 0;
  scene->light_count = 0;
  scene->ambient_color = (cl_float4){{0.1f, 0.1f, 0.1f, 0.0f}};
  // scene->background_color = (cl_float4){{0.05f, 0.05f, 0.1f, 0.0f}};
  scene->background_color = (cl_float4){{0, 0, 0, 0}};
  scene->render.max_bounces = 5;
  scene->render.gamma = 2.2f;
  scene->render.exposure = 1.0f;
  scene->render.epsilon = 0.001f;

  while (1) {
    skip_newlines(&p);
    yaml_token tok = lexer_peek(&p.lex);

    if (tok.type == TOK_EOF) break;

    // Check for state transitions based on indent
    if (tok.type == TOK_KEY && tok.indent == 0) {
      // Top-level key
      lexer_next(&p.lex); // consume

      if (key_matches(tok, "resolution")) {
        p.state = STATE_RESOLUTION;
        p.base_indent = 2;
      } else if (key_matches(tok, "camera")) {
        p.state = STATE_CAMERA;
        p.base_indent = 2;
      } else if (key_matches(tok, "render")) {
        p.state = STATE_RENDER;
        p.base_indent = 2;
      } else if (key_matches(tok, "objects")) {
        p.state = STATE_OBJECTS;
        p.base_indent = 2;
      } else if (key_matches(tok, "lights")) {
        p.state = STATE_LIGHTS;
        p.base_indent = 2;
      } else if (key_matches(tok, "materials")) {
        p.state = STATE_MATERIALS;
        p.base_indent = 2;
      } else if (key_matches(tok, "ambient")) {
        scene->ambient_color = parse_color(&p);
      } else if (key_matches(tok, "background")) {
        scene->background_color = parse_color(&p);
      }
      continue;
    }

    // Handle based on current state
    switch (p.state) {
    case STATE_ROOT:
      // Unexpected token at root
      if (tok.type != TOK_EOF) {
        parser_error(&p, tok, "unexpected token at root level");
      }
      break;

    case STATE_RESOLUTION:
      if (tok.type == TOK_KEY && tok.indent >= p.base_indent) {
        lexer_next(&p.lex);
        parse_resolution(&p, tok);
      } else {
        p.state = STATE_ROOT;
      }
      break;

    case STATE_CAMERA:
      if (tok.type == TOK_KEY && tok.indent >= p.base_indent) {
        lexer_next(&p.lex);
        parse_camera(&p, tok);
      } else {
        p.state = STATE_ROOT;
      }
      break;

    case STATE_RENDER:
      if (tok.type == TOK_KEY && tok.indent >= p.base_indent) {
        lexer_next(&p.lex);
        parse_render(&p, tok);
      } else {
        p.state = STATE_ROOT;
      }
      break;

    case STATE_OBJECTS:
      if (tok.type == TOK_DASH && tok.indent >= p.base_indent) {
        lexer_next(&p.lex);
        new_object(&p);
        // Check for inline property (e.g., "- type: sphere")
        yaml_token next = lexer_peek(&p.lex);
        if (next.type == TOK_KEY && next.line == tok.line) {
          lexer_next(&p.lex);
          parse_object_property(&p, next);
        }
        p.state = STATE_OBJECT;
      } else if (tok.indent < p.base_indent) {
        p.state = STATE_ROOT;
      } else {
        parser_error(&p, tok, "expected '-' for new object");
      }
      break;

    case STATE_OBJECT:
      if (tok.type == TOK_DASH && tok.indent == p.base_indent) {
        // New object at same level
        lexer_next(&p.lex);
        new_object(&p);
        // Check for inline property
        yaml_token next = lexer_peek(&p.lex);
        if (next.type == TOK_KEY && next.line == tok.line) {
          lexer_next(&p.lex);
          parse_object_property(&p, next);
        }
      } else if (tok.type == TOK_KEY && tok.indent > p.base_indent) {
        lexer_next(&p.lex);
        parse_object_property(&p, tok);
      } else if (tok.type == TOK_KEY && tok.indent == 0) {
        // Back to root level
        p.state = STATE_ROOT;
      } else if (tok.indent < p.base_indent) {
        p.state = STATE_ROOT;
      } else {
        parser_error(&p, tok, "expected property or new object");
      }
      break;

    case STATE_LIGHTS:
      if (tok.type == TOK_DASH && tok.indent >= p.base_indent) {
        lexer_next(&p.lex);
        new_light(&p);
        // Check for inline property (e.g., "- pos: [...]")
        yaml_token next = lexer_peek(&p.lex);
        if (next.type == TOK_KEY && next.line == tok.line) {
          lexer_next(&p.lex);
          parse_light_property(&p, next);
        }
        p.state = STATE_LIGHT;
      } else if (tok.indent < p.base_indent) {
        p.state = STATE_ROOT;
      } else {
        parser_error(&p, tok, "expected '-' for new light");
      }
      break;

    case STATE_LIGHT:
      if (tok.type == TOK_DASH && tok.indent == p.base_indent) {
        lexer_next(&p.lex);
        new_light(&p);
        // Check for inline property
        yaml_token next = lexer_peek(&p.lex);
        if (next.type == TOK_KEY && next.line == tok.line) {
          lexer_next(&p.lex);
          parse_light_property(&p, next);
        }
      } else if (tok.type == TOK_KEY && tok.indent > p.base_indent) {
        lexer_next(&p.lex);
        parse_light_property(&p, tok);
      } else if (tok.type == TOK_KEY && tok.indent == 0) {
        p.state = STATE_ROOT;
      } else if (tok.indent < p.base_indent) {
        p.state = STATE_ROOT;
      } else {
        parser_error(&p, tok, "expected property or new light");
      }
      break;

    case STATE_MATERIALS:
      // Skip materials section - consume tokens until we hit a root-level key
      if (tok.type == TOK_KEY && tok.indent == 0) {
        p.state = STATE_ROOT;
      } else {
        lexer_next(&p.lex);
      }
      break;
    }
  }
}

// Count scene items (objects and lights) without full parsing
t_scene_counts count_scene_items(const char *filename) {
  t_scene_counts counts = {0, 0};
  t_file file = read_file(filename);
  t_lexer lex;
  lexer_init(&lex, file);

  bool in_objects = false;
  bool in_lights = false;
  int base_indent = 2;

  while (1) {
    yaml_token tok = lexer_next(&lex);
    if (tok.type == TOK_EOF) break;

    // Track section changes
    if (tok.type == TOK_KEY && tok.indent == 0) {
      in_objects = (tok.len == 7 && memcmp(tok.start, "objects", 7) == 0);
      in_lights = (tok.len == 6 && memcmp(tok.start, "lights", 6) == 0);
      continue;
    }

    // Count dashes in objects/lights sections
    if (tok.type == TOK_DASH && tok.indent >= base_indent) {
      if (in_objects) counts.obj_count++;
      else if (in_lights) counts.light_count++;
    }
  }

  return counts;
}

// Parse only scene header (resolution, camera, render settings)
void parse_scene_header(const char *filename, t_scene *scene) {
  t_file file = read_file(filename);

  t_parser p = {0};
  lexer_init(&p.lex, file);
  p.scene = scene;
  p.state = STATE_ROOT;
  p.obj_idx = -1;
  p.light_idx = -1;

  // Initialize scene defaults
  scene->objects = NULL;
  scene->lights = NULL;
  scene->obj_count = 0;
  scene->light_count = 0;
  scene->bvh_nodes = NULL;
  scene->bvh_node_count = 0;
  scene->ambient_color = (cl_float4){{0.1f, 0.1f, 0.1f, 0.0f}};
  scene->background_color = (cl_float4){{0, 0, 0, 0}};
  scene->render.max_bounces = 5;
  scene->render.gamma = 2.2f;
  scene->render.exposure = 1.0f;
  scene->render.epsilon = 0.001f;

  while (1) {
    yaml_token tok = lexer_peek(&p.lex);
    if (tok.type == TOK_EOF) break;

    // Only process root-level keys for header info
    if (tok.type == TOK_KEY && tok.indent == 0) {
      lexer_next(&p.lex);

      if (key_matches(tok, "resolution")) {
        p.state = STATE_RESOLUTION;
        p.base_indent = 2;
      } else if (key_matches(tok, "camera")) {
        p.state = STATE_CAMERA;
        p.base_indent = 2;
      } else if (key_matches(tok, "render")) {
        p.state = STATE_RENDER;
        p.base_indent = 2;
      } else if (key_matches(tok, "ambient")) {
        scene->ambient_color = parse_color(&p);
      } else if (key_matches(tok, "background")) {
        scene->background_color = parse_color(&p);
      } else if (key_matches(tok, "objects") || key_matches(tok, "lights")) {
        // Stop at objects/lights sections
        break;
      }
      continue;
    }

    // Handle section content
    switch (p.state) {
    case STATE_RESOLUTION:
      if (tok.type == TOK_KEY && tok.indent >= p.base_indent) {
        lexer_next(&p.lex);
        parse_resolution(&p, tok);
      } else {
        p.state = STATE_ROOT;
      }
      break;

    case STATE_CAMERA:
      if (tok.type == TOK_KEY && tok.indent >= p.base_indent) {
        lexer_next(&p.lex);
        parse_camera(&p, tok);
      } else {
        p.state = STATE_ROOT;
      }
      break;

    case STATE_RENDER:
      if (tok.type == TOK_KEY && tok.indent >= p.base_indent) {
        lexer_next(&p.lex);
        parse_render(&p, tok);
      } else {
        p.state = STATE_ROOT;
      }
      break;

    default:
      lexer_next(&p.lex);
      break;
    }
  }
}

// Initialize incremental parser
t_incremental_parser *incremental_parser_init(const char *filename, t_scene *scene,
                                               t_scene_counts counts) {
  t_incremental_parser *ip = malloc(sizeof(t_incremental_parser));
  if (!ip) return NULL;

  ip->file = read_file(filename);
  ip->done = false;
  ip->in_objects = false;
  ip->in_lights = false;

  lexer_init(&ip->p.lex, ip->file);
  ip->p.scene = scene;
  ip->p.state = STATE_ROOT;
  ip->p.base_indent = 2;
  ip->p.obj_idx = -1;
  ip->p.light_idx = -1;
  ip->p.obj_cap = counts.obj_count;
  ip->p.light_cap = counts.light_count;

  // Pre-allocate arrays based on counts
  if (counts.obj_count > 0) {
    scene->objects = malloc(sizeof(t_object) * counts.obj_count);
  }
  if (counts.light_count > 0) {
    scene->lights = malloc(sizeof(t_light) * counts.light_count);
  }
  scene->obj_count = 0;
  scene->light_count = 0;

  // Skip to objects section
  while (1) {
    yaml_token tok = lexer_peek(&ip->p.lex);
    if (tok.type == TOK_EOF) {
      ip->done = true;
      break;
    }

    if (tok.type == TOK_KEY && tok.indent == 0) {
      lexer_next(&ip->p.lex);
      if (key_matches(tok, "objects")) {
        ip->in_objects = true;
        ip->p.state = STATE_OBJECTS;
        break;
      } else if (key_matches(tok, "lights")) {
        ip->in_lights = true;
        ip->p.state = STATE_LIGHTS;
        break;
      }
    } else {
      lexer_next(&ip->p.lex);
    }
  }

  return ip;
}

// Parse next object/light, returns false when done
bool incremental_parse_next(t_incremental_parser *ip) {
  if (ip->done) return false;

  t_parser *p = &ip->p;

  while (1) {
    yaml_token tok = lexer_peek(&p->lex);
    if (tok.type == TOK_EOF) {
      ip->done = true;
      return false;
    }

    // Check for section transitions
    if (tok.type == TOK_KEY && tok.indent == 0) {
      lexer_next(&p->lex);
      if (key_matches(tok, "objects")) {
        ip->in_objects = true;
        ip->in_lights = false;
        p->state = STATE_OBJECTS;
      } else if (key_matches(tok, "lights")) {
        ip->in_objects = false;
        ip->in_lights = true;
        p->state = STATE_LIGHTS;
      } else {
        // Some other section, we might be done
        ip->in_objects = false;
        ip->in_lights = false;
        p->state = STATE_ROOT;
      }
      continue;
    }

    // Handle objects section
    if (ip->in_objects) {
      if (tok.type == TOK_DASH && tok.indent >= p->base_indent) {
        lexer_next(&p->lex);
        new_object(p);

        // Check for inline property
        yaml_token next = lexer_peek(&p->lex);
        if (next.type == TOK_KEY && next.line == tok.line) {
          lexer_next(&p->lex);
          parse_object_property(p, next);
        }
        p->state = STATE_OBJECT;

        // Parse remaining object properties
        while (1) {
          yaml_token prop = lexer_peek(&p->lex);
          if (prop.type == TOK_EOF) break;
          if (prop.type == TOK_DASH && prop.indent == p->base_indent) break;
          if (prop.type == TOK_KEY && prop.indent == 0) break;
          if (prop.type == TOK_KEY && prop.indent > p->base_indent) {
            lexer_next(&p->lex);
            parse_object_property(p, prop);
          } else {
            break;
          }
        }
        return true;  // Parsed one object
      } else if (tok.indent < p->base_indent || (tok.type == TOK_KEY && tok.indent == 0)) {
        ip->in_objects = false;
        p->state = STATE_ROOT;
      } else {
        lexer_next(&p->lex);
      }
      continue;
    }

    // Handle lights section
    if (ip->in_lights) {
      if (tok.type == TOK_DASH && tok.indent >= p->base_indent) {
        lexer_next(&p->lex);
        new_light(p);

        // Check for inline property
        yaml_token next = lexer_peek(&p->lex);
        if (next.type == TOK_KEY && next.line == tok.line) {
          lexer_next(&p->lex);
          parse_light_property(p, next);
        }
        p->state = STATE_LIGHT;

        // Parse remaining light properties
        while (1) {
          yaml_token prop = lexer_peek(&p->lex);
          if (prop.type == TOK_EOF) break;
          if (prop.type == TOK_DASH && prop.indent == p->base_indent) break;
          if (prop.type == TOK_KEY && prop.indent == 0) break;
          if (prop.type == TOK_KEY && prop.indent > p->base_indent) {
            lexer_next(&p->lex);
            parse_light_property(p, prop);
          } else {
            break;
          }
        }
        return true;  // Parsed one light
      } else if (tok.indent < p->base_indent || (tok.type == TOK_KEY && tok.indent == 0)) {
        ip->in_lights = false;
        p->state = STATE_ROOT;
      } else {
        lexer_next(&p->lex);
      }
      continue;
    }

    // Not in objects or lights, check if we can enter
    lexer_next(&p->lex);
  }
}

void incremental_parser_cleanup(t_incremental_parser *ip) {
  if (ip) {
    free(ip);
  }
}
