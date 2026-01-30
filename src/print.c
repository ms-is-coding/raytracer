#include "types.h"
#include "scene/scene.h"

#include <stdio.h>

/**
 * Recursively prints the BVH structure.
 * @param nodes The array of BVH nodes.
 * @param node_idx The current node index being printed.
 * @param depth Current depth (used for indentation).
 */
static void print_bvh_recursive(t_bvh_node *nodes, int node_idx, int depth) {
    if (node_idx == -1) return;

    t_bvh_node *node = &nodes[node_idx];
    
    // Create indentation
    for (int i = 0; i < depth; i++) printf("  ");

    if (node->left == -1 && node->right == -1) {
        // Leaf Node
        printf("L [Idx: %d] Leaf: %d objs (start: %d) | Min:(%.2f, %.2f, %.2f) Max:(%.2f, %.2f, %.2f)\n",
               node_idx,
               node->obj_count,
               node->obj_start,
               node->bounds.min.s[0], node->bounds.min.s[1], node->bounds.min.s[2],
               node->bounds.max.s[0], node->bounds.max.s[1], node->bounds.max.s[2]);
    } else {
        // Internal Node
        printf("I [Idx: %d] Internal | Min:(%.2f, %.2f, %.2f) Max:(%.2f, %.2f, %.2f)\n",
               node_idx,
               node->bounds.min.s[0], node->bounds.min.s[1], node->bounds.min.s[2],
               node->bounds.max.s[0], node->bounds.max.s[1], node->bounds.max.s[2]);
        
        print_bvh_recursive(nodes, node->left, depth + 1);
        print_bvh_recursive(nodes, node->right, depth + 1);
    }
}

/**
 * Public entry point to print the scene's BVH.
 */
void debug_print_bvh(t_scene *scene) {
    if (!scene || !scene->bvh_nodes || scene->bvh_node_count == 0) {
        printf("BVH is empty or not built.\n");
        return;
    }

    printf("--- BVH Tree Structure (%d nodes) ---\n", scene->bvh_node_count);
    // The root is always index 0 based on your build_recursive logic
    print_bvh_recursive(scene->bvh_nodes, 0, 0);
    printf("------------------------------------\n");
}
