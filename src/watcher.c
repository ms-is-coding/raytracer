#include "minirt.h"
#include <sys/inotify.h>
#include <stdio.h>
#include <string.h>

#define EVENT_SIZE  (sizeof(struct inotify_event))
#define BUF_LEN     (1024 * (EVENT_SIZE + 16))

typedef struct s_watcher {
  int fd;
  int wd;
  char *filename;
} t_watcher;

static t_watcher g_watcher = {-1, -1, NULL};

/**
 * Initializes the inotify watcher for the specified scene file.
 */
void init_config_watcher(const char *filename) {
  g_watcher.fd = inotify_init();
  if (g_watcher.fd < 0) {
    perror("inotify_init");
    return;
  }

  // Set non-blocking so we don't hang the render loop
  fcntl(g_watcher.fd, F_SETFL, O_NONBLOCK);

  g_watcher.filename = strdup(filename);

  // Initial watch
  g_watcher.wd = inotify_add_watch(g_watcher.fd, g_watcher.filename, IN_CLOSE_WRITE | IN_MODIFY | IN_MOVE_SELF);
  if (g_watcher.wd < 0) {
    perror("inotify_add_watch");
  }
}

/**
 * Checks if the configuration file has changed.
 * If changed, it re-parses the scene and updates GPU buffers.
 */
void check_for_scene_updates(t_data *data) {
  if (g_watcher.fd < 0) return;

  char buffer[BUF_LEN];
  // Read all available events
  ssize_t length = read(g_watcher.fd, buffer, BUF_LEN);

  // If length > 0, at least one event occurred
  if (length > 0) {
    // Re-add watch immediately as some editors replace the file entirely (atomic saves)
    if (g_watcher.wd >= 0) {
      inotify_rm_watch(g_watcher.fd, g_watcher.wd);
    }

    // Retry loop to wait for the file to reappear if it was deleted/renamed
    int retries = 0;
    while (retries < 10) {
      g_watcher.wd = inotify_add_watch(g_watcher.fd, g_watcher.filename, IN_CLOSE_WRITE | IN_MODIFY | IN_MOVE_SELF);
      if (g_watcher.wd >= 0) break;
      usleep(20000); // Wait 20ms
      retries++;
    }

    if (g_watcher.wd < 0) {
      // If we still can't find it, we stop here and wait for next check
      return;
    }

    // Drain the inotify buffer to prevent an infinite loop of triggers
    // caused by multiple events (like MOVE_SELF followed by CLOSE_WRITE)
    while (read(g_watcher.fd, buffer, BUF_LEN) > 0);

    printf("Change detected in %s. Reloading...\n", g_watcher.filename);

    // 1. Clean up old CPU memory
    if (data->scene.objects) free(data->scene.objects);
    if (data->scene.lights) free(data->scene.lights);

    // 2. Re-parse the file
    parse_scene(g_watcher.filename, &data->scene);

    // 3. Free old BVH and rebuild
    free_bvh(&data->scene);
    build_bvh(&data->scene);
    // debug_print_bvh(&data->scene);

    // 4. Update OpenCL buffers
    cl_int err;

    // Release old buffers
    if (data->cl.obj_buffer) clReleaseMemObject(data->cl.obj_buffer);
    if (data->cl.light_buffer) clReleaseMemObject(data->cl.light_buffer);
    if (data->cl.bvh_buffer) clReleaseMemObject(data->cl.bvh_buffer);

    // Create new buffers with updated data
    data->cl.obj_buffer = clCreateBuffer(data->cl.context,
                                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(t_object) * data->scene.obj_count, data->scene.objects, &err);

    data->cl.light_buffer = clCreateBuffer(data->cl.context,
                                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(t_light) * data->scene.light_count, data->scene.lights, &err);

    // Create new BVH buffer
    if (data->scene.bvh_node_count > 0) {
      data->cl.bvh_buffer = clCreateBuffer(data->cl.context,
                                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(t_bvh_node) * data->scene.bvh_node_count,
                                           data->scene.bvh_nodes, &err);
    } else {
      t_bvh_node dummy = {0};
      data->cl.bvh_buffer = clCreateBuffer(data->cl.context,
                                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(t_bvh_node), &dummy, &err);
    }

    // 5. Force a re-render in the main loop
    data->should_render = 1;
  }
}

/**
 * Cleans up the watcher resources.
 */
void cleanup_config_watcher() {
  if (g_watcher.fd >= 0) {
    if (g_watcher.wd >= 0) inotify_rm_watch(g_watcher.fd, g_watcher.wd);
    close(g_watcher.fd);
  }
  if (g_watcher.filename) free(g_watcher.filename);
}
