#include "internal.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

void lexer_init(t_lexer *lex, t_file file) {
  lex->file = file;
  lex->cursor = file.data;
  lex->line_start = file.data;
  lex->line = 1;
  lex->has_peeked = 0;
  lex->line_indent = 0;
}

static int at_end(t_lexer *lex) {
  return lex->cursor >= lex->file.data + lex->file.size;
}

static char peek_char(t_lexer *lex) {
  if (at_end(lex)) return '\0';
  return *lex->cursor;
}

static char advance(t_lexer *lex) {
  if (at_end(lex)) return '\0';
  return *lex->cursor++;
}

static yaml_token make_token(t_lexer *lex, yaml_token_type type,
                             const char *start, size_t len) {
  return (yaml_token){
    .type = type,
    .line = lex->line,
    .col = (int)(start - lex->line_start) + 1,
    .indent = 0,
    .start = start,
    .len = len,
    .num_val = 0,
  };
}

static yaml_token error_token(t_lexer *lex, const char *start) {
  return make_token(lex, TOK_ERROR, start, 1);
}

static void skip_comment(t_lexer *lex) {
  while (!at_end(lex) && peek_char(lex) != '\n') {
    advance(lex);
  }
}

static int is_ident_start(char c) {
  return isalpha(c) || c == '_';
}

static int is_ident_char(char c) {
  return isalnum(c) || c == '_' || c == '.';
}

static yaml_token scan_number(t_lexer *lex, const char *start) {
  while (isdigit(peek_char(lex))) advance(lex);

  if (peek_char(lex) == '.' && isdigit(lex->cursor[1])) {
    advance(lex); // consume '.'
    while (isdigit(peek_char(lex))) advance(lex);
  }

  yaml_token tok = make_token(lex, TOK_NUMBER, start, lex->cursor - start);
  tok.num_val = strtod(start, NULL);
  return tok;
}

static yaml_token scan_identifier(t_lexer *lex, const char *start) {
  while (is_ident_char(peek_char(lex))) advance(lex);

  size_t len = lex->cursor - start;

  // Skip whitespace to check for colon
  const char *save = lex->cursor;
  while (peek_char(lex) == ' ' || peek_char(lex) == '\t') advance(lex);

  if (peek_char(lex) == ':') {
    advance(lex); // consume ':'
    return make_token(lex, TOK_KEY, start, len);
  }

  // Not a key, restore position and return string
  lex->cursor = save;
  return make_token(lex, TOK_STRING, start, len);
}

static yaml_token scan_token(t_lexer *lex, int indent) {
  // Skip inline whitespace
  while (peek_char(lex) == ' ' || peek_char(lex) == '\t') {
    advance(lex);
  }

  // Skip comments
  if (peek_char(lex) == '#') {
    skip_comment(lex);
  }

  if (at_end(lex)) {
    return make_token(lex, TOK_EOF, lex->cursor, 0);
  }

  const char *start = lex->cursor;
  char c = advance(lex);

  // Newline
  if (c == '\n') {
    yaml_token tok = make_token(lex, TOK_NEWLINE, start, 1);
    lex->line++;
    lex->line_start = lex->cursor;
    return tok;
  }

  // Single character tokens
  if (c == '[') return make_token(lex, TOK_LBRACKET, start, 1);
  if (c == ']') return make_token(lex, TOK_RBRACKET, start, 1);
  if (c == ',') return make_token(lex, TOK_COMMA, start, 1);
  if (c == ':') return make_token(lex, TOK_COLON, start, 1);

  // Dash: could be list marker or negative number
  if (c == '-') {
    if (peek_char(lex) == ' ' || peek_char(lex) == '\n' || at_end(lex)) {
      yaml_token tok = make_token(lex, TOK_DASH, start, 1);
      tok.indent = indent;
      return tok;
    }
    if (isdigit(peek_char(lex))) {
      return scan_number(lex, start);
    }
    return make_token(lex, TOK_DASH, start, 1);
  }

  // Numbers
  if (isdigit(c)) {
    lex->cursor--; // back up
    return scan_number(lex, start);
  }

  // Identifiers and keys
  if (is_ident_start(c)) {
    lex->cursor--; // back up
    yaml_token tok = scan_identifier(lex, start);
    tok.indent = indent;
    return tok;
  }

  return error_token(lex, start);
}

yaml_token lexer_next(t_lexer *lex) {
  if (lex->has_peeked) {
    lex->has_peeked = 0;
    return lex->peeked;
  }

  // At start of line, count indent
  if (lex->cursor == lex->line_start) {
    lex->line_indent = 0;
    while (peek_char(lex) == ' ') {
      advance(lex);
      lex->line_indent++;
    }
    // Handle tab as error or convert to spaces
    if (peek_char(lex) == '\t') {
      return error_token(lex, lex->cursor);
    }
  }

  // Skip empty lines and comment-only lines
  while (peek_char(lex) == '#' || peek_char(lex) == '\n') {
    if (peek_char(lex) == '#') {
      skip_comment(lex);
    }
    if (peek_char(lex) == '\n') {
      advance(lex);
      lex->line++;
      lex->line_start = lex->cursor;
      lex->line_indent = 0;
      while (peek_char(lex) == ' ') {
        advance(lex);
        lex->line_indent++;
      }
    }
  }

  if (at_end(lex)) {
    return make_token(lex, TOK_EOF, lex->cursor, 0);
  }

  yaml_token tok = scan_token(lex, lex->line_indent);
  tok.indent = lex->line_indent;
  return tok;
}

yaml_token lexer_peek(t_lexer *lex) {
  if (!lex->has_peeked) {
    lex->peeked = lexer_next(lex);
    lex->has_peeked = 1;
  }
  return lex->peeked;
}
