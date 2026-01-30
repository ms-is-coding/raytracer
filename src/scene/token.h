#pragma once

#include <stddef.h>

typedef enum {
  TOK_EOF,
  TOK_NEWLINE,
  TOK_KEY,          // identifier followed by ':'
  TOK_DASH,         // list item marker '-'
  TOK_COLON,        // ':'
  TOK_NUMBER,       // integer or float
  TOK_STRING,       // unquoted word
  TOK_LBRACKET,     // '['
  TOK_RBRACKET,     // ']'
  TOK_COMMA,        // ','
  TOK_ERROR,
} yaml_token_type;

typedef struct {
  yaml_token_type type;
  int             line;
  int             col;
  int             indent;
  const char      *start;
  size_t          len;
  double          num_val;
} yaml_token;
