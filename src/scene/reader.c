#include "internal.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>

t_file read_file(const char *filename) {
  int         fd;
  char        *data;
  struct stat sb;

  fd = open(filename, O_RDONLY);
  if (fd < 0) {
    perror("Error opening config");
    exit(1);
  }
  fstat(fd, &sb);
  data = mmap(NULL, sb.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  if (data == MAP_FAILED) {
    perror("mmap failed");
    exit(1);
  }
  return (t_file){
    .name = filename,
    .data = data,
    .size = sb.st_size
  };
}
