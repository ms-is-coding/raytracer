#pragma once

#include "scene.h"

typedef struct {
  t_file      file;
  const char  *cursor;
  const char  *line_start;
  int         line;
  int         line_indent;
  yaml_token  peeked;
  int         has_peeked;
} t_lexer;

t_file      read_file(const char *filename);
void        lexer_init(t_lexer *lex, t_file file);
yaml_token  lexer_next(t_lexer *lex);
yaml_token  lexer_peek(t_lexer *lex);
