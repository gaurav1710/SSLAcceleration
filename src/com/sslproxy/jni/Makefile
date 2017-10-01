CC=g++
LINKER_DIRS=-L/usr/local/cuda/lib64
LINKER_FLAGS=-lcudart -lcudadevrt 
NVCC=nvcc
CUDA_ARCHITECTURE=20
OCELOT=`OcelotConfig -l`
COMMAND=$(CC) -w -fPIC  bkernell.o bkernels.o gpu_algos.o com_sslproxy_jni_CudaKernels.o -o libCudaKernels.so -shared $(LINKER_DIRS) $(LINKER_FLAGS)
JNI_INCLUDE=-I/usr/lib/jvm/java-8-openjdk-amd64/include -I/usr/lib/jvm/java-8-openjdk-amd64/include/linux/

all: createlib

com_sslproxy_jni_CudaKernels.o: com_sslproxy_jni_CudaKernels.c
	$(CC) -w -fPIC $(JNI_INCLUDE) -c com_sslproxy_jni_CudaKernels.c

buildkernels:
	nvcc -w -c -arch=sm_35 -Xcompiler '-fPIC'  $(JNI_INCLUDE) -rdc=true bkernels.cu
	nvcc -w -arch=sm_35 -Xcompiler '-fPIC' -dlink -o bkernell.o bkernels.o
	nvcc -w -c -arch=sm_35 -Xcompiler '-fPIC' gpu_algos.cu

clean:
	rm -f CudaKernels.class com_sslproxy_jni_CudaKernels.h com_sslproxy_jni_CudaKernels.o bkernell.o bkernels.o libCudaKernels.so gpu_algos.o

jni:    	
	javac *.java
	javah -cp "../../../" -jni com.sslproxy.jni.CudaKernels

createlib:	jni com_sslproxy_jni_CudaKernels.o buildkernels
	$(COMMAND)