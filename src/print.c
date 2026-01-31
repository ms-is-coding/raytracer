#include "scene/scene.h"
#include <stdio.h>

#define C_RESET   "\033[0m"
#define C_GREEN   "\033[32m"
#define C_YELLOW  "\033[33m"
#define C_CYAN    "\033[36m"
#define C_DIM     "\033[37m"

#define MAX_BVH_DEPTH 128

typedef struct {
  int node_idx;
  int depth;
  int is_last;
} t_stack_entry;

static void print_indent(int depth, int *last_stack) {
  for (int i = 0; i < depth; i++) {
    printf(last_stack[i] ? "   " : "│  ");
  }
}

void debug_print_bvh(t_scene *scene) {
  if (!scene || !scene->bvh_nodes || scene->bvh_node_count == 0) {
    printf("BVH empty.\n");
    return;
  }

  printf("=== BVH Tree (%d nodes) ===\n", scene->bvh_node_count);

  t_stack_entry stack[MAX_BVH_DEPTH];
  int last_stack[MAX_BVH_DEPTH] = {0};
  int sp = 0;

  // push root
  stack[sp++] = (t_stack_entry){ .node_idx = 0, .depth = 0, .is_last = 1 };

  while (sp) {
    t_stack_entry e = stack[--sp];
    t_bvh_node *n = &scene->bvh_nodes[e.node_idx];
    int leaf = (n->left == -1 && n->right == -1);

    last_stack[e.depth] = e.is_last;
    printf(C_YELLOW "%6d " C_RESET, e.node_idx);
    print_indent(e.depth, last_stack);

    printf(e.is_last ? "└─" : "├─");

    if (leaf) {
      printf("Leaf objs=%d start=%d ",
             n->obj_count, n->obj_start);
    } else {
      printf(C_CYAN "Node " C_RESET);
    }

    printf("min(%.2f %.2f %.2f) max(%.2f %.2f %.2f)\n",
           n->bounds.min.s[0], n->bounds.min.s[1], n->bounds.min.s[2],
           n->bounds.max.s[0], n->bounds.max.s[1], n->bounds.max.s[2]);

    // push right first so left prints first (stack LIFO)
    if (!leaf) {
      if (n->right != -1)
        stack[sp++] = (t_stack_entry){ n->right, e.depth + 1, 1 };
      if (n->left != -1)
        stack[sp++] = (t_stack_entry){ n->left,  e.depth + 1, 0 };
    }
  }

  printf("==========================\n");
}

