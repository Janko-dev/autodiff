CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -pedantic -g -lm
SRC = src/
IN = $(SRC)autodiff.c $(SRC)nn.c $(SRC)main.c
OUT = autodiff

make:
	$(CC) $(IN) -o $(OUT) $(CFLAGS)