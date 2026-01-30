#pragma once

#include "../types.h"
#include "token.h"
#include <stdbool.h>

typedef struct {
  const char *name;
  const char *data;
  size_t      size;
} t_file;

typedef struct {
  int obj_count;
  int light_count;
} t_scene_counts;

typedef struct {
  int width;
  int height;
  cl_float4 ambient_color;
  cl_float4 background_color;
  t_camera camera;
  t_render render;

  t_object *objects;
  t_light *lights;
  t_material *materials;
  t_bvh_node *bvh_nodes;
  int *obj_order;
  int obj_count;
  int light_count;
  int material_count;
  int bvh_node_count;
} t_scene;

void parse_scene(const char *filename, t_scene *scene);
void parse_scene_header(const char *filename, t_scene *scene);
void scene_error(t_file f, yaml_token t, const char *msg);

// Incremental parsing
t_scene_counts count_scene_items(const char *filename);

typedef struct s_incremental_parser t_incremental_parser;
t_incremental_parser *incremental_parser_init(const char *filename, t_scene *scene,
                                               t_scene_counts counts);
bool incremental_parse_next(t_incremental_parser *ip);
void incremental_parser_cleanup(t_incremental_parser *ip);
