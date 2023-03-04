CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -pedantic -g -std=c11 -lm 
SRC = src/
IN = $(SRC)autodiff.c $(SRC)mlp.c $(SRC)main.c
OUT = autodiff

make: $(IN)
	$(CC) $(IN) -o $(OUT) $(CFLAGS)