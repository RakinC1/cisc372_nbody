FLAGS= -DDEBUG
LIBS= -lm
CUDA_FLAGS= -arch=sm_70 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o
	nvcc $(FLAGS) $(CUDA_FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) $(CUDA_FLAGS) -c $< 
compute.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) $(CUDA_FLAGS) -c $< 
clean:
	rm -f *.o nbody
