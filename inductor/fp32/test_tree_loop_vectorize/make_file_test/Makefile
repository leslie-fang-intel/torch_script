# False result: 36
# flag = -v -O2 -DNDEBUG -ftree-loop-vectorize -ffast-math -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -march=native -fopenmp

# Correct result: 1188864
flag = -v -O2 -DNDEBUG -ffast-math -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -march=native -fopenmp
main: main.o
	g++ $(flag) -o main main.o
main.o: main.cpp
	g++ $(flag) -c main.cpp
clean:
	rm main *.o