CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -pedantic -g -lm
IN = autodiff.c nn.c main.c
OUT = autodiff

make:
	$(CC) $(IN) -o $(OUT) $(CFLAGS)