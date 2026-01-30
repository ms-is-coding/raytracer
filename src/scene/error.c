#include "internal.h"
#include "token.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define COLOR_RESET  "\x1b[0m"
#define COLOR_BOLD   "\x1b[1m"
#define COLOR_GRAY   "\x1b[90m"
#define COLOR_RED    "\x1b[91m"
#define COLOR_YELLOW "\x1b[93m"
#define COLOR_BLUE   "\x1b[94m"

int int_len(int x) {
  int len = 0;
  if (x <= 0)
    len++;
  while (x) {
    len++;
    x /= 10;
  }
  return len;
}

void repeat(char *c, int count) {
  for (int i = 0; i < count; i++)
    fprintf(stderr, "%s", c);
}

void scene_error(t_file f, yaml_token t, const char *msg)
{
  const char *ls, *le;
  int col;

  // same logic as find_line
  const char *p = t.start;
  while (p > f.data && p[-1] != '\n') p--;
  ls = p;
  le = strchr(t.start, '\n');
  if (!le) le = f.data + f.size;
  col = (int)(t.start - ls) + 1;

  fprintf(stderr,
          COLOR_BOLD "--> " COLOR_BLUE "%s" COLOR_RESET ":"
          COLOR_YELLOW "%d" COLOR_RESET ":" COLOR_YELLOW "%d" COLOR_RESET
          COLOR_BOLD COLOR_RED " Parse Error\n" COLOR_RESET,
          f.name, t.line, col);

  fprintf(stderr, " " COLOR_BOLD "%d | " COLOR_RESET, t.line);
  fprintf(stderr, "%.*s", col - 1, ls);
  fprintf(stderr, COLOR_RED "%.*s" COLOR_RESET, (int)t.len, t.start);
  fprintf(stderr, "%.*s\n", (int)(le - (t.start + t.len)), t.start + t.len);

  repeat((char *)" ", int_len(t.line) + 2);
  fprintf(stderr, COLOR_BOLD "|" COLOR_RESET);
  repeat((char *)" ", col);
  fprintf(stderr, COLOR_RED "^");
  repeat((char *)"~", t.len > 1 ? t.len - 1 : 0);
  fprintf(stderr, COLOR_RED " %s\n\n" COLOR_RESET, msg);

  exit(1);
}

