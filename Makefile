CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -pedantic -g -std=c11 -lm
SRC = src/
IN = $(SRC)autodiff.c $(SRC)mlp.c $(SRC)main.c
OUT = autodiff
make: $(IN)
	$(CC) $(IN) -o $(OUT) $(CFLAGS)

run_demo: $(DEMO_C) $(DEMO_H)
	$(CC) demo/demo_old.c demo/autodiff_old.c
	./a.out
	$(CC) demo/demo_new.c src/autodiff.c
	./a.out
