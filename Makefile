CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -pedantic -g -std=c11 -lm 
SRC = src/
EXAMPLES = examples/
IN = $(SRC)autodiff.c $(SRC)main.c
OUT = autodiff

make: $(IN)
	$(CC) $(IN) -o $(OUT) $(CFLAGS)

mlp_example: $(SRC)autodiff.c $(EXAMPLES)mlp.c
	$(CC) $(SRC)autodiff.c $(EXAMPLES)mlp.c -o mlp_demo $(CFLAGS)