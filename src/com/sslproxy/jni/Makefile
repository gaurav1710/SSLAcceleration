MPI_INCLUDE=-I/usr/lib/openmpi/include/
GNUPLOT_INCLUDE=-I/home/gaurav/MCS/VISem/MajorProject2/Work/lib/gnuplot_i/src/
LINKER_DIRS=-L/usr/local/cuda/lib64 -L/usr/lib/openmpi/lib
LINKER_FLAGS=-lcudart -lcudadevrt -lpthread 
all:
	nvcc -w -c -arch=sm_35 -Xcompiler '-fPIC -fopenmp' $(GNUPLOT_INCLUDE) $(MPI_INCLUDE) -I/usr/include/python2.7 bkernels.cu
	g++ -c -pg cpu_algos.cpp -fopenmp -I/usr/include/python2.7 
	nvcc -w -c -arch=sm_35  gpu_algos.cu
	mpic++ -fPIC -pg -fopenmp bkernels.o cpu_algos.o gpu_algos.o /home/gaurav/MCS/VISem/MajorProject2/Work/lib/gnuplot_i/gnuplot_i.o -o ktest $(LINKER_DIRS) $(LINKER_FLAGS)	
clean:
	rm ktest bkernels.o cpu_algos.o	gpu_algos.o