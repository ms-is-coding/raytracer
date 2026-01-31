NAME			:= rt

SRC_DIR			:= src
LIB_DIR			:= lib
BUILD_DIR		:= build

CC				:= cc
CFLAGS			:= -Wall -Wextra -MMD -MP -std=gnu2x -mavx2
CFLAGS_DEBUG	:= -Og -g3 -Wshadow -Wpadded -Wconversion -Wstrict-prototypes \
								 -Wmissing-declarations -Wstrict-prototypes -Wundef \
								 -Wmissing-prototypes -Wold-style-definition -Winline \
								 -Wsign-conversion -Wcast-align -Wcast-qual -Wwrite-strings \
								 -Wuninitialized -Wdouble-promotion -Wfloat-equal -Wvla \
								 -Wnull-dereference -Wformat=2 -fstack-protector-strong
CFLAGS_RELEASE	:= -O2 -DNDEBUG -march=native -mavx2 -D__is_42sh
CFLAGS_SANITIZE	:= -fsanitize=address,undefined,leak

ifeq ($(MODE), release)
	CFLAGS += $(CFLAGS_RELEASE)
else ifeq ($(MODE), debug)
	CFLAGS += $(CFLAGS_DEBUG)
else ifeq ($(MODE), sanitize)
	CFLAGS += $(CFLAGS_DEBUG) $(CFLAGS_SANITIZE)
else
	MODE = default
	CFLAGS += -Werror
endif

ROOT_DIR	:= $(BUILD_DIR)/$(MODE)
OBJ_DIR		:= $(ROOT_DIR)/obj

SRC_SCENE := $(addprefix scene/, reader.c lexer.c parser.c error.c)
SRC_UI := $(addprefix ui/, ui.c color.c rect.c switch.c slider.c button.c label.c panel.c color_picker.c)
SRC_FILES		:= $(SRC_SCENE) $(SRC_UI) main.c opencl.c watcher.c bvh.c cpu_render.c print.c

SRCS			:= $(addprefix $(SRC_DIR)/, $(SRC_FILES))
OBJS			:= $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
DEPS			:= $(OBJS:.o=.d)

LIBMLX_DIR		:= $(LIB_DIR)/mlx
LIBMLX			:= $(LIBMLX_DIR)/libmlx.a
LDLIBS			:= -lX11 -lXext -lmlx -lOpenCL -lm -lpthread
LDFLAGS			:= -L$(LIBMLX_DIR)

INCLUDES		:= -Iinclude -I$(LIBMLX_DIR)


.PHONY: all
all: $(NAME)
	@$(MAKE) postbuild --no-print-directory


.PHONY: default
default: all


.PHONY: release
release:
	@$(MAKE) MODE=release --no-print-directory


.PHONY: debug
debug:
	@$(MAKE) MODE=debug --no-print-directory


.PHONY: sanitize
sanitize:
	@$(MAKE) MODE=sanitize --no-print-directory


.PHONY: bonus
bonus: release


.PHONY: postbuild
postbuild:
	cp -f $(ROOT_DIR)/$(NAME) $(NAME)


$(NAME): libmlx
	@make $(ROOT_DIR)/$(NAME) --no-print-directory


$(ROOT_DIR)/$(NAME): $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(OBJS) $(LDFLAGS) $(LDLIBS) -o $(ROOT_DIR)/$(NAME)


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@


.PHONY: libmlx
libmlx:
	$(MAKE) CC=clang -C $(LIBMLX_DIR) --no-print-directory


.PHONY: clean
clean:
	@$(MAKE) -C $(LIBMLX_DIR) clean --no-print-directory
	rm -rf $(BUILD_DIR)


.PHONY: fclean
fclean:
	rm -rf $(BUILD_DIR)
	rm -f $(NAME)


.PHONY: re
re: fclean
	@$(MAKE) all --no-print-directory


.PHONY: norm
norm:
	@echo $(SRCS) $(shell find . -name "*.h") | xargs -n1 -P$(shell nproc) norminette | grep -v OK!


.PHONY: tidy
tidy:
	echo $(SRCS) | xargs -n1 -P$(shell nproc) clang-tidy -p .


-include $(DEPS)
