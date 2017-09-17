CC=g++
LINKER_DIRS=-L/usr/local/cuda/lib64
LINKER_FLAGS=-lcudart -lcudadevrt -fopenmp
MPI_INCLUDE=-I/usr/lib/openmpi/include/ -I/usr/lib/gcc/x86_64-linux-gnu/4.8/include/
GNUPLOT_INCLUDE=-I/home/gaurav/MCS/VISem/MajorProject2/Work/lib/gnuplot_i/src/
GNUPLOT_O=/home/gaurav/MCS/VISem/MajorProject2/Work/lib/gnuplot_i/src/gnuplot_i.o
NVCC=nvcc
CUDA_ARCHITECTURE=20
OCELOT=`OcelotConfig -l`
COMMAND=$(CC) -w -fPIC  bkernell.o bkernels.o com_sslproxy_jni_CudaKernels.o -o libCudaKernels.so -shared $(LINKER_DIRS) $(LINKER_FLAGS)


all: createlib

com_sslproxy_jni_CudaKernels.o: com_sslproxy_jni_CudaKernels.c
	$(CC) -w -fPIC -I/usr/lib/jvm/java-8-openjdk-amd64/include/ -I/usr/lib/jvm/java-8-openjdk-amd64/include/linux/ -c com_sslproxy_jni_CudaKernels.c

buildkernels:
	nvcc -w -c -arch=sm_35 -Xcompiler '-fPIC -fopenmp' $(GNUPLOT_INCLUDE) $(MPI_INCLUDE) -rdc=true bkernels.cu
	nvcc -w -arch=sm_35 -Xcompiler '-fPIC' -dlink -o bkernell.o bkernels.o

clean:
	rm -f CudaKernels.class com_sslproxy_jni_CudaKernels.h com_sslproxy_jni_CudaKernels.o bkernell.o bkernels.o libCudaKernels.so

jni:    	
	javac *.java
	javah -cp "../../../" -jni com.sslproxy.jni.CudaKernels

createlib:	jni com_sslproxy_jni_CudaKernels.o buildkernels
	$(COMMAND)
