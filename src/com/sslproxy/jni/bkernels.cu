z#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
//#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include "config.h"
#include "cpu_algos.h"
#include "gpu_algos.h"

//Execution flag on HPC cluster(K40M) or GT730
#define HPC 1

//#if HPC
//#include "/usr/lib/jvm/java-7-openjdk-amd64/include/jni.h"
////#include "/usr/lib/gcc/x86_64-redhat-linux/4.4.4/include/jni.h"
//#else
//#include "/usr/lib/jvm/java-7-openjdk-amd64/include/jni.h"
//extern "C" {
//#include "gnuplot_i.h"
//}
//;
//#endif
//#include "kernels.h"


//Number of CUDA enabled cards
int num_of_devices = 1;

//compute threads per block = multiple of number of threads in a warp(calculated for maximum occupancy per SM)
int THREADSPERBLOCK[NUMKERNELS] = { 512, 512, 512, 512, 512, 512, 512 };

char *kernel_names[] = { "pmul", "padd", "psub", "right_shift", "pcopy",
		"pcopy_wcondition", "convert_to_base" };

cudaStream_t *streams[MAXNUMCPUTHREADS];

request_batch *cpu_reqs[MAXNUMCPUTHREADS];

int offset_k[MAXNUMCPUTHREADS] = { 0 };
int offset_2k[MAXNUMCPUTHREADS] = { 0 };
int offset_req[MAXNUMCPUTHREADS] = { 0 };
int offset_base[MAXNUMCPUTHREADS] = { 0 };

int hoffset_k[MAXNUMCPUTHREADS] = { 0 };
int hoffset_2k[MAXNUMCPUTHREADS] = { 0 };
int hoffset_req[MAXNUMCPUTHREADS] = { 0 };
int hoffset_base[MAXNUMCPUTHREADS] = { 0 };

void clear_tops() {
	for (int i = 0; i < MAXNUMCPUTHREADS; i++) {
		offset_k[i] = 0;
		offset_2k[i] = 0;
		offset_req[i] = 0;
		offset_base[i] = 0;

		hoffset_k[i] = 0;
		hoffset_2k[i] = 0;
		hoffset_req[i] = 0;
		hoffset_base[i] = 0;
	}
}

//x, y coordinates for gnuplot
double sx[NUMSAMPLES];
double sy[NUMSAMPLES];

const int NUMSTREAMS = 1;//2
		//* (MAXREQUESTSPERCPUTHREAD / MAXREQUESTSPERSTREAM
		//		+ (MAXREQUESTSPERCPUTHREAD % MAXREQUESTSPERSTREAM != 0));
const int MEMK = 0;
const int MEM2K = 1;
const int MEMREQ = 2;
const int MEMBASE = 3;

double GPUCPUMULRATIO=1.0;
double GPUCPUADDRATIO=1.0;
double GPUCPUSUBRATIO=1.0;
double GPUCPUREQRATIO=1.0;

int cpu_v = 1;
int gpu_v = 1;

//memory pool for device
radix_type *memory_pool_k[MAXNUMCPUTHREADS];
radix_type *memory_pool_2k[MAXNUMCPUTHREADS];
radix_type *memory_pool_req[MAXNUMCPUTHREADS];
radix_type *memory_pool_base[MAXNUMCPUTHREADS];

//host memory pool
radix_type *hmemory_pool_k[MAXNUMCPUTHREADS];
radix_type *hmemory_pool_2k[MAXNUMCPUTHREADS];
radix_type *hmemory_pool_req[MAXNUMCPUTHREADS];
radix_type *hmemory_pool_base[MAXNUMCPUTHREADS];
radix_type *hzeros[MAXNUMCPUTHREADS];

void create_memory_pool(int k) {
	for (int i = 0; i < MAXNUMCPUTHREADS; i++){
	//int i;
	//omp_set_num_threads(MAXNUMCPUTHREADS);
	//#pragma omp parallel
	//{
		//int i = omp_get_thread_num();
#if !SERIAL
		cudaSetDevice(i % num_of_devices);
		//Device Global Memory
		CUDAMEMDEVICE((void **) &memory_pool_k[i],
		MAXMEMSEGMENTS * k * MAXREQUESTSPERCPUTHREAD * sizeof(radix_type));
		CUDAMEMDEVICE((void **) &memory_pool_2k[i],
		MAXMEMSEGMENTS * 2 * k * MAXREQUESTSPERCPUTHREAD * sizeof(radix_type));
		CUDAMEMDEVICE((void **) &memory_pool_req[i],
		MAXMEMSEGMENTS * MAXREQUESTSPERCPUTHREAD * sizeof(radix_type));
		CUDAMEMDEVICE((void **) &memory_pool_base[i],
				MAXMEMSEGMENTS * k * BASE * MAXREQUESTSPERCPUTHREAD
						* sizeof(radix_type));

		//Pinned Host Memory
		CUDAMEMHOST((void **) &hmemory_pool_k[i],
		MAXMEMSEGMENTS * k * MAXREQUESTSPERCPUTHREAD * sizeof(radix_type));
		CUDAMEMHOST((void **) &hmemory_pool_2k[i],
		MAXMEMSEGMENTS * 2 * k * MAXREQUESTSPERCPUTHREAD * sizeof(radix_type));
		CUDAMEMHOST((void **) &hmemory_pool_req[i],
		MAXMEMSEGMENTS * MAXREQUESTSPERCPUTHREAD * sizeof(radix_type));
		CUDAMEMHOST((void **) &hmemory_pool_base[i],
				MAXMEMSEGMENTS * k * BASE * MAXREQUESTSPERCPUTHREAD
						* sizeof(radix_type));
		CUDAMEMHOST((void **) &hzeros[i],
				4 * k * MAXMEMSEGMENTS * sizeof(radix_type));

		memset(hzeros[i], 0, 4 * k * MAXMEMSEGMENTS * sizeof(radix_type));
#endif
	}

}
void print_mem_offset_status() {
	printf("offset_k=%d\n", offset_k[0]);
	printf("offset_2k=%d\n", offset_2k[0]);
	printf("offset_req=%d\n", offset_req[0]);
	printf("offset_base=%d\n", offset_base[0]);
}



void copy_zeros_to_device(radix_type *da, int words, int streamid) {
	cudaMemcpyAsync(da, hzeros[streamid], words * sizeof(radix_type),
			cudaMemcpyHostToDevice, streams[streamid][0]);

}
void copy_data_to_device(radix_type *ha, radix_type *da, int words,
		int streamid) {
	//Host pinned to device global memory
	cudaMemcpyAsync(da, ha, words * sizeof(radix_type), cudaMemcpyHostToDevice,
			streams[streamid][0]);
}
void copy_data_to_host(radix_type *ha, radix_type *da, int words,
		int streamid) {
	cudaMemcpyAsync(ha, da, words * sizeof(radix_type), cudaMemcpyDeviceToHost,
			streams[streamid][0]);
}

void copy_data_to_device_ds(radix_type *ha, radix_type *da, int words,
		int streamid) {
	//Host pinned to device global memory
	cudaMemcpyAsync(da, ha, words * sizeof(radix_type), cudaMemcpyHostToDevice,
			streams[streamid][1]);
}
void copy_data_to_host_ds(radix_type *ha, radix_type *da, int words,
		int streamid) {
	cudaMemcpyAsync(ha, da, words * sizeof(radix_type), cudaMemcpyDeviceToHost,
			streams[streamid][1]);
}

//Return memory = type*MAXREQUESTSPERCPUTHREAD
radix_type *allocate(int type, int k, int num_req, int tid) {
	radix_type *mem_ponter = NULL;
	switch (type) {
	case MEMK:
		if (offset_k[tid]
				< MAXMEMSEGMENTS * MAXREQUESTSPERCPUTHREAD * k
						* MAXNUMCPUTHREADS) {
			mem_ponter = &memory_pool_k[tid][offset_k[tid] + k * num_req];
			offset_k[tid] += k * num_req;
		}
		break;
	case MEM2K:
		if (offset_2k[tid]
				< MAXMEMSEGMENTS * 2 * k * MAXREQUESTSPERCPUTHREAD
						* MAXNUMCPUTHREADS) {
			mem_ponter = &memory_pool_2k[tid][offset_2k[tid] + 2 * k * num_req];
			offset_2k[tid] += 2 * k * num_req;
		}
		break;
	case MEMREQ:
		if (offset_req[tid]
				< MAXMEMSEGMENTS * MAXREQUESTSPERCPUTHREAD * MAXNUMCPUTHREADS) {
			mem_ponter = &memory_pool_req[tid][offset_req[tid] + num_req];
			offset_req[tid] += num_req;
		}
		break;
	case MEMBASE:
		if (offset_base[tid]
				< MAXMEMSEGMENTS * k * BASE * MAXREQUESTSPERCPUTHREAD
						* MAXNUMCPUTHREADS) {
			mem_ponter = &memory_pool_base[tid][offset_base[tid]
					+ k * BASE * num_req];
			offset_base[tid] += k * BASE * num_req;
		}
		break;
	}
	return mem_ponter;
}

radix_type *hallocate(int type, int k, int num_req, int tid) {
	radix_type *mem_ponter = NULL;
	if(tid == -1){
		return sallocate(type , k , num_req , tid );
	}
#if !SERIAL
	switch (type) {
	case MEMK:
		if (hoffset_k[tid]
				< MAXMEMSEGMENTS * MAXREQUESTSPERCPUTHREAD * k
						* MAXNUMCPUTHREADS) {
			mem_ponter = &hmemory_pool_k[tid][hoffset_k[tid] + k * num_req];
			hoffset_k[tid] += k * num_req;
		}
		break;
	case MEM2K:
		if (hoffset_2k[tid]
				< MAXMEMSEGMENTS * 2 * k * MAXREQUESTSPERCPUTHREAD
						* MAXNUMCPUTHREADS) {
			mem_ponter =
					&hmemory_pool_2k[tid][hoffset_2k[tid] + 2 * k * num_req];
			hoffset_2k[tid] += 2 * k * num_req;
		}
		break;
	case MEMREQ:
		if (hoffset_req[tid]
				< MAXMEMSEGMENTS * MAXREQUESTSPERCPUTHREAD * MAXNUMCPUTHREADS) {
			mem_ponter = &hmemory_pool_req[tid][hoffset_req[tid] + num_req];
			hoffset_req[tid] += num_req;
		}
		break;
	case MEMBASE:
		if (hoffset_base[tid]
				< MAXMEMSEGMENTS * k * BASE * MAXREQUESTSPERCPUTHREAD
						* MAXNUMCPUTHREADS) {
			mem_ponter = &hmemory_pool_base[tid][hoffset_base[tid]
					+ k * BASE * num_req];
			hoffset_base[tid] += k * BASE * num_req;
		}
		break;
	}
#endif
#if SERIAL
	mem_ponter = sallocate(type , k , num_req , tid );
#endif
	return mem_ponter;
}

void cleanup() {
	for (int i = 0; i < MAXNUMCPUTHREADS; i++) {
		cudaSetDevice(i % num_of_devices);
		cudaFree(memory_pool_k[i]);
		cudaFree(memory_pool_2k[i]);
		cudaFree(memory_pool_req[i]);
		cudaFree(memory_pool_base[i]);
	}
	clear_tops();
}

void deallocate(int type, int k, int num_req, int tid) {
	switch (type) {
	case MEMK:
		offset_k[tid] -= k * num_req;
		break;
	case MEM2K:
		offset_2k[tid] -= 2 * k * num_req;
		break;
	case MEMREQ:
		offset_req[tid] -= num_req;
		break;
	case MEMBASE:
		offset_base[tid] -= k * BASE * num_req;
		break;
	}
}

void hdeallocate(int type, int k, int num_req, int tid) {
	switch (type) {
	case MEMK:
		hoffset_k[tid] -= k * num_req;
		break;
	case MEM2K:
		hoffset_2k[tid] -= 2 * k * num_req;
		break;
	case MEMREQ:
		hoffset_req[tid] -= num_req;
		break;
	case MEMBASE:
		hoffset_base[tid] -= k * BASE * num_req;
		break;
	}
}


void print_array(radix_type a[], int l) {
	for (int i = l - 1; i >= 0; i--) {
		printf("%d ", a[i]);
	}
	printf("\n");

}


inline void extract_req_bits(short_radix_type *base2y,
		short_radix_type *base2_bits, int ind, int num_req, int k,
		int streamid) {
	extract_bits<<<1, num_req, 0, streams[streamid][0]>>>(base2y, base2_bits,
			ind, k);
	cudaError_t err = cudaStreamSynchronize(streams[streamid][0]);
	if (err != cudaSuccess)
		printf("Error Encountered in Extract_bits---%s\n",
				cudaGetErrorString(err));
}

inline void parallel_mult(radix_type *m, radix_type *n, int k, radix_type *res,
		int stridem, int striden, int tnum_req, int streamid, double share_ratio) {
	//THREADSPERBLOCK[0]
	int num_req = (int)(((double)tnum_req)*share_ratio);
	int streams_req = num_req / MAXREQUESTSPERSTREAM
			+ (num_req % MAXREQUESTSPERSTREAM != 0);
	streams_req = streams_req > NUMSTREAMS ? NUMSTREAMS : streams_req;
	streams_req = 1;
	int num_blocks = (num_req * k) / THREADSPERBLOCK[0]
			+ ((num_req * k) % THREADSPERBLOCK[0] != 0);
	int num_threads = THREADSPERBLOCK[0];
	if ((num_req * k) < THREADSPERBLOCK[0])
		num_threads = num_req * k;
	int num_req_per_block = num_threads / k;

	radix_type *hmp = hallocate(MEM2K, stridem, tnum_req - num_req, streamid);
	radix_type *hnp = hallocate(MEM2K, striden, tnum_req - num_req, streamid);
	radix_type *hrp = hallocate(MEM2K, 2*k, tnum_req - num_req, streamid);
	if((tnum_req - num_req) != 0){
		copy_data_to_host(hmp, &m[num_req*stridem], (tnum_req - num_req)*stridem,streamid);
		copy_data_to_host(hnp, &n[num_req*striden], (tnum_req - num_req)*striden,streamid);
		copy_data_to_host(hrp, &res[num_req*2*k], (tnum_req - num_req)*2*k,streamid);
		cudaError_t err = cudaStreamSynchronize(streams[streamid][0]);
	}


	for (int i = 0; i < streams_req; i++) {
		pmul<<<num_blocks, num_threads,
				5 * k * num_req_per_block * sizeof(radix_type),
				streams[streamid][i]>>>(m, n, k, res, stridem, striden,
				num_threads);
	}

	//execute some requests on CPU while waiting for GPU to complete..
	if((tnum_req - num_req) != 0){
		smul(hmp, hnp, k, hrp, stridem, striden,tnum_req - num_req, EXTRATHREADS);
		copy_data_to_device(hmp, &m[num_req*stridem], (tnum_req - num_req)*stridem,streamid);
		copy_data_to_device(hnp, &n[num_req*striden], (tnum_req - num_req)*striden,streamid);
		copy_data_to_device(hrp,  &res[num_req*2*k], (tnum_req - num_req)*2*k,streamid);

	}
	hdeallocate(MEM2K, stridem, tnum_req - num_req, streamid);
	hdeallocate(MEM2K, striden, tnum_req - num_req, streamid);
	hdeallocate(MEM2K, 2*k, tnum_req - num_req, streamid);
	for (int i = 0; i < streams_req; i++) {
		cudaError_t err = cudaStreamSynchronize(streams[streamid][i]);
		if (err != cudaSuccess)
			printf("Error Encountered in Parallel_Mult---%s\n",
					cudaGetErrorString(err));
	}
}
inline void parallel_add(radix_type *a, radix_type *b, int k, int tnum_req,
		int stridea, int strideb, radix_type *residue_carry, int streamid, double share_ratio) {
	//THREADSPERBLOCK[1]
	int num_req = (int)(((double)tnum_req)*share_ratio);
	int num_blocks = (num_req * k) / THREADSPERBLOCK[1]
			+ ((num_req * k) % THREADSPERBLOCK[1] != 0);
	int num_threads = THREADSPERBLOCK[1];
	if ((num_req * k) < THREADSPERBLOCK[1])
		num_threads = num_req * k;
	int num_req_per_block = num_threads / k;


	//CPU task sharing

	radix_type *hap = hallocate(MEM2K, stridea, tnum_req - num_req, streamid);
	radix_type *hbp = hallocate(MEM2K, strideb, tnum_req - num_req, streamid);
	radix_type *hrcp = hallocate(MEMREQ, k, tnum_req - num_req, streamid);
	if((tnum_req - num_req) != 0){
		copy_data_to_host_ds(hap, &a[num_req*stridea], (tnum_req - num_req)*stridea,streamid);
		copy_data_to_host_ds(hbp, &b[num_req*strideb], (tnum_req - num_req)*strideb,streamid);
		copy_data_to_host_ds(hrcp, &residue_carry[num_req], (tnum_req - num_req),streamid);
	}
	if(num_req!=0){
		padd<<<num_blocks, num_threads,
					(2 * k + 1) * num_req_per_block * sizeof(radix_type),
					streams[streamid][0]>>>(a, b, k, num_req, stridea, strideb,
					residue_carry, num_threads);
	}
	if((tnum_req - num_req) != 0){
		cudaError_t err = cudaStreamSynchronize(streams[streamid][1]);
		sadd(hap, hbp, k, tnum_req - num_req, stridea, strideb,hrcp, EXTRATHREADS);
		copy_data_to_device_ds(hap, &a[num_req*stridea], (tnum_req - num_req)*stridea,streamid);
		copy_data_to_device_ds(hrcp, &residue_carry[num_req], (tnum_req - num_req),streamid);
	}
	hdeallocate(MEM2K, stridea, tnum_req - num_req, streamid);
	hdeallocate(MEM2K, strideb, tnum_req - num_req, streamid);
	hdeallocate(MEMREQ, k, tnum_req - num_req, streamid);
	cudaError_t err = cudaStreamSynchronize(streams[streamid][0]);
	if (err != cudaSuccess)
		printf("Error Encountered in Parallel_Add---%s\n",
				cudaGetErrorString(err));

}

inline void parallel_sub(radix_type *a, radix_type *b, int k, radix_type *res,
		radix_type *add_carry, int tnum_req, int stridea, int strideb,
		int compare, int streamid, double share_ratio) {
	//THREADSPERBLOCK[3]
	int num_req = (int)(((double)tnum_req)*share_ratio);
	int num_blocks = (num_req * k) / THREADSPERBLOCK[3]
			+ ((num_req * k) % THREADSPERBLOCK[3] != 0);
	int num_threads = THREADSPERBLOCK[3];
	if ((num_req * k) < THREADSPERBLOCK[3])
		num_threads = num_req * k;
	int num_req_per_block = num_threads / k;

	radix_type *hap = NULL;
	radix_type *hbp = NULL;
	radix_type *hrp = NULL;
	radix_type *hacp = NULL;
	if((tnum_req - num_req) != 0){
		hap = hallocate(MEM2K, stridea, tnum_req - num_req, streamid);
		hbp = hallocate(MEM2K, strideb, tnum_req - num_req, streamid);
		hrp = hallocate(MEM2K, k, tnum_req - num_req, streamid);
		hacp = hallocate(MEMREQ, k, tnum_req - num_req, streamid);
		copy_data_to_host(hap, &a[num_req*stridea], (tnum_req - num_req)*stridea,streamid);
		copy_data_to_host(hbp, &b[num_req*strideb], (tnum_req - num_req)*strideb,streamid);
		copy_data_to_host(hrp, &res[num_req*k], (tnum_req - num_req)*k,streamid);
		copy_data_to_host(hacp, &add_carry[num_req], (tnum_req - num_req),streamid);
		cudaError_t err = cudaStreamSynchronize(streams[streamid][0]);
	}
	if(num_req!=0){
		psub<<<num_blocks, num_threads,
				2 * k * num_req_per_block * sizeof(radix_type), streams[streamid][0]>>>(
				a, b, k, res, add_carry, stridea, strideb, compare, num_threads);
	}
	if((tnum_req - num_req) != 0){
		ssub(hap, hbp, k, hrp, hacp, stridea, strideb, compare, tnum_req - num_req, EXTRATHREADS);
		copy_data_to_device(hrp, &res[num_req*k], (tnum_req - num_req)*k,streamid);
	}

	cudaError_t err = cudaStreamSynchronize(streams[streamid][0]);
	if (err != cudaSuccess)
		printf("Error Encountered in Parallel_sub---%s\n",
				cudaGetErrorString(err));
}

inline void parallel_rshift(radix_type *a, int stridea, int k, int num_req,
		radix_type *res, int streamid) {
	//THREADSPERBLOCK[2]
	int num_blocks = (num_req * k) / THREADSPERBLOCK[2]
			+ ((num_req * k) % THREADSPERBLOCK[2] != 0);
	int num_threads = THREADSPERBLOCK[2];
	if ((num_req * k) < THREADSPERBLOCK[2])
		num_threads = num_req * k;
	right_shift<<<num_blocks, num_threads, 0, streams[streamid][0]>>>(a,
			stridea, k, res, num_threads);
	cudaError_t err = cudaStreamSynchronize(streams[streamid][0]);
	if (err != cudaSuccess)
		printf("Error Encountered in Parallel_rshift---%s\n",
				cudaGetErrorString(err));
}

inline void parallel_copy(radix_type *a, radix_type *b, int stridea,
		int strideb, int k, int num_req, int streamid) {
	//THREADSPERBLOCK[4]
	int num_blocks = (num_req * k) / THREADSPERBLOCK[4]
			+ ((num_req * k) % THREADSPERBLOCK[4] != 0);
	int num_threads = THREADSPERBLOCK[4];
	if ((num_req * k) < THREADSPERBLOCK[4])
		num_threads = num_req * k;
	pcopy<<<num_blocks, num_threads, 0, streams[streamid][0]>>>(a, b, stridea,
			strideb, k, num_threads);
	cudaError_t err = cudaStreamSynchronize(streams[streamid][0]);
	if (err != cudaSuccess)
		printf("Error Encountered in Parallel_copy---%s\n",
				cudaGetErrorString(err));
}

inline void parallel_copy_wcondition(radix_type *a, radix_type *b,
		short_radix_type *base2, int k, int stridea, int strideb, int num_req,
		int streamid) {
	//THREADSPERBLOCK[5]
	int num_blocks = (num_req * k) / THREADSPERBLOCK[5]
			+ ((num_req * k) % THREADSPERBLOCK[5] != 0);
	int num_threads = THREADSPERBLOCK[5];
	if ((num_req * k) < THREADSPERBLOCK[5])
		num_threads = num_req * k;
	pcopy_wcondition<<<num_blocks, num_threads, 0, streams[streamid][0]>>>(a, b,
			base2, k, stridea, strideb, num_threads);
	cudaError_t err = cudaStreamSynchronize(streams[streamid][0]);
	if (err != cudaSuccess)
		printf("Error Encountered in Parallel_copy_wcondition---%s\n",
				cudaGetErrorString(err));
}

inline void parallel_base_convert(radix_type *x, short_radix_type *y, int k,
		int base, int num_req, int streamid) {
	//THREADSPERBLOCK[6]
	int num_blocks = num_req / THREADSPERBLOCK[6];
	if (num_req % THREADSPERBLOCK[6] != 0)
		num_blocks++;
	convert_to_base<<<num_blocks, THREADSPERBLOCK[6], 0, streams[streamid][0]>>>(
			x, y, k, 2, num_req, THREADSPERBLOCK[6]);
	cudaError_t err = cudaStreamSynchronize(streams[streamid][0]);
	if (err != cudaSuccess)
		printf("Error Encountered in Parallel_base_convert---%s\n",
				cudaGetErrorString(err));
}

unsigned long convert_2base10(radix_type *a, int k) {
	unsigned long res = 0;
	unsigned long pow = 1;

	for (int i = 0; i < k; i++) {
		res += pow * a[i];
		pow *= (BASEMINUS1 + 1);
	}
	return res;
}

inline void montgomery(radix_type *a, radix_type *b, int k, radix_type *m,
		radix_type *rinv, radix_type *mbar, radix_type *res, int num_req,
		int tid) {

	radix_type *T, *M, *U, *carry, *addition_carry;

	T = allocate(MEM2K, k, num_req, tid);
	M = allocate(MEM2K, k, num_req, tid);
	U = allocate(MEM2K, k, num_req, tid);
	carry = allocate(MEM2K, k, num_req, tid);
	addition_carry = allocate(MEMREQ, k, num_req, tid);

	copy_zeros_to_device(T, 2 * k * num_req, tid);

	//T=a*b
	parallel_mult(a, b, k, T, k, k, num_req, tid, GPUCPUMULRATIO);

	if (DEBUG) {
		printf("montgomery-MulT:");
		print_array(T, 2 * k);
	}

	copy_zeros_to_device(M, 2 * k * num_req, tid);

	//M=T*m'
	parallel_mult(T, mbar, k, M, 2 * k, k, num_req, tid, GPUCPUMULRATIO);
	if (DEBUG) {
		printf("montgomery-MulM:");
		print_array(M, 2 * k);
	}
	copy_zeros_to_device(U, 2 * k * num_req, tid);
	//U=M*m
	parallel_mult(M, m, k, U, 2 * k, k, num_req, tid, GPUCPUMULRATIO);
	if (DEBUG) {
		printf("montgomery-MulU:");
		print_array(U, 2 * k);
	}
	copy_zeros_to_device(addition_carry, num_req, tid);

	//U=T+M*m
	parallel_add(U, T, 2 * k, num_req, 2 * k, 2 * k, addition_carry, tid, GPUCPUADDRATIO);
	if (DEBUG) {
		printf("montgomery-AddU2:");
		print_array(U, 2 * k);
	}
	copy_zeros_to_device(carry, 2 * k * num_req, tid);

	parallel_rshift(U, 2 * k, k, num_req, carry, tid);	//carry is result here..
	if (DEBUG) {
		printf("montgomery-RShiftU:");
		print_array(carry, 2 * k);
	}
	parallel_copy(res, carry, k, 2 * k, k, num_req, tid);

	parallel_sub(carry, m, k, res, addition_carry, num_req, 2 * k, k, 1, tid, GPUCPUSUBRATIO);
	if (DEBUG) {
		printf("montgomery-SubRes:");
		print_array(res, k);
	}
	deallocate(MEM2K, k, num_req, tid);
	deallocate(MEM2K, k, num_req, tid);
	deallocate(MEM2K, k, num_req, tid);
	deallocate(MEM2K, k, num_req, tid);
	deallocate(MEMREQ, k, num_req, tid);
}

inline void hmod_exp_sqmul(radix_type *x, radix_type *y, int k, radix_type *m,
		radix_type *res, radix_type *rinv, radix_type *mbar, radix_type *r2modm,
		int num_req, int tid) {
	radix_type *temp = allocate(MEMK, k, num_req, tid);
	short_radix_type *base2y = allocate(MEMBASE, k, num_req, tid);
	radix_type *ones = allocate(MEMK, k, num_req, tid);
	short_radix_type *base2_bits = allocate(MEMREQ, k, num_req, tid);
	radix_type *htemp = hallocate(MEMK, k, num_req, tid);
	radix_type *hones = hallocate(MEMK, k, num_req, tid);

	memset(htemp, 0, num_req * k * sizeof(radix_type));
	memset(hones, 0, num_req * k * sizeof(radix_type));

	for (int i = 0; i < num_req; i++) {
		htemp[i * k] = 1;	//initialize with 1
		hones[i * k] = 1;
	}
	copy_data_to_device(htemp, temp, k * num_req, tid);
	copy_data_to_device(hones, ones, k * num_req, tid);

	copy_zeros_to_device(res, k * num_req, tid);

	parallel_base_convert(y, base2y, k, 2, num_req, tid);

	montgomery(temp, r2modm, k, m, rinv, mbar, res, num_req, tid);

	if (DEBUG) {
		printf("R2*1 mod m:");
		print_array(res, k);
	}
	parallel_copy(temp, res, k, k, k, num_req, tid);

	copy_zeros_to_device(res, k * num_req, tid);
	montgomery(x, r2modm, k, m, rinv, mbar, res, num_req, tid);

	if (DEBUG) {
		printf("R2*x mod m:");
		print_array(res, k);
	}
	parallel_copy(x, res, k, k, k, num_req, tid);

	for (int i = BASE * k - 1; i >= 0; i--) {
		copy_zeros_to_device(res, k * num_req, tid);
		montgomery(temp, temp, k, m, rinv, mbar, res, num_req, tid);

		parallel_copy(temp, res, k, k, k, num_req, tid);
		copy_zeros_to_device(res, k * num_req, tid);
		montgomery(temp, x, k, m, rinv, mbar, res, num_req, tid);

		extract_req_bits(base2y, base2_bits, i, num_req, k, tid);
		parallel_copy_wcondition(temp, res, base2_bits, k, k, k, num_req, tid);
	}

	copy_zeros_to_device(res, k * num_req, tid);
	montgomery(temp, ones, k, m, rinv, mbar, res, num_req, tid);
	deallocate(MEMK, k, num_req, tid);
	deallocate(MEMBASE, k, num_req, tid);
	deallocate(MEMK, k, num_req, tid);
	deallocate(MEMREQ, k, num_req, tid);
	hdeallocate(MEMK, k, num_req, tid);
	hdeallocate(MEMK, k, num_req, tid);
}

inline void montgomery_with_conv(radix_type *a, radix_type *b, radix_type *m,
		radix_type *rinv, radix_type *mbar, radix_type *r2modm, radix_type *res,
		int k, int num_req, int tid) {
	radix_type *ar, *br, *abr, *ones;
	ar = allocate(MEMK, k, num_req, tid);
	br = allocate(MEMK, k, num_req, tid);
	abr = allocate(MEMK, k, num_req, tid);
	ones = allocate(MEMK, k, num_req, tid);
	radix_type *hones = hallocate(MEMK, k, num_req, tid);
	memset(hones, 0, num_req * k * sizeof(radix_type));
	copy_zeros_to_device(abr, k * num_req, tid);
	copy_zeros_to_device(ar, k * num_req, tid);
	copy_zeros_to_device(br, k * num_req, tid);

	for (int i = 0; i < num_req; i++) {
		hones[i * k] = 1;
	}
	copy_data_to_device(hones, ones, k * num_req, tid);

	//copy_data_to_device(hones,ones,k*num_req,tid);
	//a*r mod m
	montgomery(a, r2modm, k, m, rinv, mbar, ar, num_req, tid);
	//b*r mod m
	montgomery(b, r2modm, k, m, rinv, mbar, br, num_req, tid);
	//a*b*r mod m
	montgomery(ar, br, k, m, rinv, mbar, abr, num_req, tid);
	//a*b mod m
	montgomery(abr, ones, k, m, rinv, mbar, res, num_req, tid);
	deallocate(MEMK, k, num_req, tid);
	deallocate(MEMK, k, num_req, tid);
	deallocate(MEMK, k, num_req, tid);
	deallocate(MEMK, k, num_req, tid);
	hdeallocate(MEMK, k, num_req, tid);
}

void calmerge_m1_m2(radix_type *c, radix_type *e, radix_type *m, int k,
		radix_type *rinv, radix_type *mbar, radix_type *r2modm, radix_type *q,
		radix_type *qinv, radix_type *res, int num_req, int tid) {
	radix_type *t1, *t2, *t3, *mont;
	radix_type *M, *addition_carry;

	//Calculate M1 and M2
	M = allocate(MEM2K, k, num_req, tid);

	copy_zeros_to_device(M, k * 2 * num_req, tid);

	hmod_exp_sqmul(c, e, k, m, M, rinv, mbar, r2modm, 2 * num_req, tid);

	t1 = allocate(MEMK, k, num_req, tid);
	t2 = allocate(MEM2K, k, num_req, tid);
	t3 = allocate(MEM2K, k, num_req, tid);
	addition_carry = allocate(MEMREQ, k, num_req, tid);
	mont = allocate(MEMK, k, num_req, tid);

	copy_zeros_to_device(t1, k * num_req, tid);
	copy_zeros_to_device(addition_carry, num_req, tid);
	if (DEBUG) {
		printf("M1=%lu\n", convert_2base10(M, k));
		printf("M2=%lu\n", convert_2base10(&M[num_req * k], k));
	}
	//M1-M2
	parallel_sub(M, &M[num_req * k], k, t1, addition_carry, num_req, k, k, 0,
			tid, GPUCPUSUBRATIO);
	if (DEBUG) {
		printf("M1-M2=%lu\n", convert_2base10(t1, k));
	}
	copy_zeros_to_device(t2, 2 * k * num_req, tid);
	copy_zeros_to_device(t3, 2 * k * num_req, tid);
	copy_zeros_to_device(mont, k * num_req, tid);

	//(M1-M2)*(qinv mod p) mod p
	montgomery_with_conv(t1, qinv, m, rinv, mbar, r2modm, mont, k, num_req,
			tid);
	if (DEBUG) {
		printf("(M1-M2)*q^inv mod p=%lu\n", convert_2base10(mont, k));
	}
	copy_zeros_to_device(t2, 2 * k * num_req, tid);
	copy_zeros_to_device(t3, 2 * k * num_req, tid);

	//(M1-M2)*(qinv mod p)*q
	parallel_mult(mont, q, k, t2, k, k, num_req, tid, GPUCPUMULRATIO);
	if (DEBUG) {
		printf("((M1-M2)*q^inv mod p)*q=%lu\n", convert_2base10(t2, 2 * k));
	}
	parallel_copy(res, t2, 2 * k, 2 * k, 2 * k, num_req, tid);

	copy_zeros_to_device(addition_carry, num_req, tid);
	//M2 + (M1-M2)*(qinv mod p)*q
	parallel_add(res, &M[num_req * k], k, num_req, 2 * k, k, addition_carry,
			tid, GPUCPUADDRATIO);
	if (DEBUG) {
		printf("M2+((M1-M2)*q^inv mod p)*q=%lu\n", convert_2base10(res, 2 * k));
	}
	deallocate(MEMK, k, num_req, tid);
	deallocate(MEM2K, k, num_req, tid);
	deallocate(MEM2K, k, num_req, tid);
	deallocate(MEMREQ, k, num_req, tid);
	deallocate(MEMK, k, num_req, tid);
}

void h_sub(radix_type *a, radix_type *b, int k, radix_type *res) {
	radix_type carry = 0;
	radix_type dig = 0;
	for (int i = 0; i < k; i++) {
		dig = b[i] + carry;
		carry = dig > a[i];
		res[i] = ((carry) * (BASEMINUS1 + 1) + a[i] - dig);
	}
}

inline void setup_streams() {

	for (int tid = 0; tid < MAXNUMCPUTHREADS; tid++) {
		cudaSetDevice(tid % num_of_devices);
		streams[tid] = new cudaStream_t[NUMSTREAMS+COPYSTREAMS];
		cudaError_t error;
		for (int i = 0; i < NUMSTREAMS+COPYSTREAMS; i++) {
			error = cudaStreamCreate(&streams[tid][i]);
			if (error != cudaSuccess)
				printf("Error Encountered stream_setup---%s\n",
						cudaGetErrorString(error));
		}
	}
}

inline void destroy_streams() {
	cudaError_t error;
	for (int j = 0; j < MAXNUMCPUTHREADS; j++) {
		cudaSetDevice(j % num_of_devices);
		for (int i = 0; i < NUMSTREAMS+COPYSTREAMS; i++) {
			error = cudaStreamDestroy(streams[j][i]);
			if (error != cudaSuccess)
				printf("Error Encountered stream_destroy---%s\n",
						cudaGetErrorString(error));
		}
	}
}

static struct timeval tm1;

static inline void start() {
	gettimeofday(&tm1, NULL);
}
static inline double stop() {
	struct timeval tm2;
	gettimeofday(&tm2, NULL);

	double t = 1000 * (tm2.tv_sec - tm1.tv_sec)
			+ (tm2.tv_usec - tm1.tv_usec) / 1000;
	return t;
}

int sample_no = 0;

void *execute(void *vid) {
	request_batch *reqs = (request_batch *) vid;
	int *tid = &reqs->request_id;

	cudaSetDevice((reqs->request_id) % num_of_devices);
	radix_type *dx, *dy, *dm, *dres, *drinv, *dmbar, *dr2modm, *dm1m2, *dq,
			*dqinv;
	int num_req = reqs->batch_size;
	int k = BITLENGTH;

	dx = allocate(MEM2K, k, num_req, *tid);
	dy = allocate(MEM2K, k, num_req, *tid);
	dm = allocate(MEM2K, k, num_req, *tid);
	dres = allocate(MEM2K, k, num_req, *tid);
	dmbar = allocate(MEM2K, k, num_req, *tid);
	drinv = allocate(MEM2K, k, num_req, *tid);
	dr2modm = allocate(MEM2K, k, num_req, *tid);
	dm1m2 = allocate(MEM2K, k, num_req, *tid);
	dq = allocate(MEMK, k, num_req, *tid);
	dqinv = allocate(MEMK, k, num_req, *tid);

	memset(reqs->res, 0, 2 * num_req * k * sizeof(radix_type));

	copy_data_to_device(reqs->x, dx, 2 * k * num_req, *tid);
	copy_data_to_device(reqs->y, dy, 2 * k * num_req, *tid);
	copy_data_to_device(reqs->m, dm, 2 * k * num_req, *tid);
	copy_data_to_device(reqs->res, dres, 2 * k * num_req, *tid);
	copy_data_to_device(reqs->mbar, dmbar, 2 * k * num_req, *tid);
	copy_data_to_device(reqs->rinv, drinv, 2 * k * num_req, *tid);
	copy_data_to_device(reqs->r2modm, dr2modm, 2 * k * num_req, *tid);
	copy_data_to_device(reqs->m1m2, dm1m2, 2 * k * num_req, *tid);
	copy_data_to_device(reqs->q, dq, k * num_req, *tid);
	copy_data_to_device(reqs->qinv, dqinv, k * num_req, *tid);

	long c = reqs->x[0] + (BASEMINUS1 + 1) * reqs->x[1];
	long e = reqs->y[0] + (BASEMINUS1 + 1) * reqs->y[1];
	long mod = reqs->m[0] + (BASEMINUS1 + 1) * reqs->m[1];

	//printf("Calculating %ld^%ld mod %ld..\n", c, e, mod);
	start();
	calmerge_m1_m2(dx, dy, dm, k, drinv, dmbar, dr2modm, dq, dqinv, dm1m2,
			num_req, *tid);
	double t = stop();
	sy[sample_no] = t;
	printf("(GPU)M = f(M1,M2):%lf\n",t);

	copy_data_to_host(reqs->m1m2, dm1m2, 2 * k * num_req, *tid);
	cudaError_t err = cudaStreamSynchronize(streams[*tid][0]);
	for (int i = 0; i < num_req; i++) {
		if (BITLENGTH == 2) {
			if ((reqs->m1m2[3 + i * k * 2] == 12001)
					&& (reqs->m1m2[2 + i * k * 2] == 33560)
					&& (reqs->m1m2[1 + i * k * 2] == 27692)
					&& (reqs->m1m2[0 + i * k * 2] == 64771))
				printf("GPU %d PASS\n", i);
			else
				printf("GPU %d FAIL\n", i);
			//Ans:12001 33560 27692 64771
			assert(reqs->m1m2[3 + i * k * 2] == 12001);
			assert(reqs->m1m2[2 + i * k * 2] == 33560);
			assert(reqs->m1m2[1 + i * k * 2] == 27692);
			assert(reqs->m1m2[0 + i * k * 2] == 64771);
		}
	}
	deallocate(MEM2K, k, num_req, *tid);
	deallocate(MEM2K, k, num_req, *tid);
	deallocate(MEM2K, k, num_req, *tid);
	deallocate(MEM2K, k, num_req, *tid);
	deallocate(MEM2K, k, num_req, *tid);
	deallocate(MEM2K, k, num_req, *tid);
	deallocate(MEM2K, k, num_req, *tid);
	deallocate(MEM2K, k, num_req, *tid);
	deallocate(MEMK, k, num_req, *tid);
	deallocate(MEMK, k, num_req, *tid);

	return vid;
}

void *execute_serial(void *vid) {


	request_batch *reqs = (request_batch *) vid;
	int *tid = &reqs->request_id;
	int num_req = reqs->batch_size;
	int k = BITLENGTH;

	memset(reqs->res, 0, 2 * num_req * k * sizeof(radix_type));

	start();
	if(cpu_v == 1){
		scalmerge_m1_m2(reqs->x, reqs->y, reqs->m, k, reqs->rinv, reqs->mbar, reqs->r2modm, reqs->q, reqs->qinv, reqs->m1m2,
			num_req, *tid);
	}else{
		modex(num_req);
	}
	double t = stop();
	sy[sample_no] = t;
	printf("(CPU)M = f(M1,M2):%lf\n",t);
//	for (int i = 0; i < num_req; i++) {
//		if (BITLENGTH == 2) {
//			if ((reqs->m1m2[3 + i * k * 2] == 12001)
//					&& (reqs->m1m2[2 + i * k * 2] == 33560)
//					&& (reqs->m1m2[1 + i * k * 2] == 27692)
//					&& (reqs->m1m2[0 + i * k * 2] == 64771))
//				printf("CPU %d PASS\n", i);
//			else
//				printf("CPU %d FAIL\n", i);
//			//Ans:12001 33560 27692 64771
//			assert(reqs->m1m2[3 + i * k * 2] == 12001);
//			assert(reqs->m1m2[2 + i * k * 2] == 33560);
//			assert(reqs->m1m2[1 + i * k * 2] == 27692);
//			assert(reqs->m1m2[0 + i * k * 2] == 64771);
//		}
//	}
	return vid;
}

#if !HPC
void plot_graph(char *file_name, int size) {
	char file_command[100];
	strcpy(file_command, "set output \"");
	strcat(file_command, file_name);
	strcat(file_command, "\"");
	gnuplot_ctrl * g = gnuplot_init();
	gnuplot_cmd(g, "set terminal png");
	gnuplot_cmd(g, file_command);
	gnuplot_setstyle(g, "lines");
	gnuplot_set_xlabel(g, "Number of threads per block");
	gnuplot_set_ylabel(g, "Latency(ms)");
	gnuplot_plot_xy(g, sx, sy, size,
			"Number of Threads Per Block Vs Latency(ms)");
	gnuplot_close(g);
}

void plot_graph_param(char *file_name, double *x, double *y, int size, char *xl,char *yl, char *head) {
	char file_command[100];
	strcpy(file_command, "set output \"");
	strcat(file_command, file_name);
	strcat(file_command, "\"");
	gnuplot_ctrl * g = gnuplot_init();
	gnuplot_cmd(g, "set terminal png");
	gnuplot_cmd(g, file_command);
	gnuplot_setstyle(g, "lines");
	gnuplot_set_xlabel(g, xl);
	gnuplot_set_ylabel(g, yl);
	gnuplot_plot_xy(g, x, y, size,
			head);
	gnuplot_close(g);
}

int find_miny() {
	double min = 24 * 60 * 60 * 1000;
	int min_tpb = BITLENGTH;
	for (int i = 0; i < NUMSAMPLES; i++) {
		if (min >= sy[i] && sy[i] != 0) {
			min = sy[i];
			min_tpb = sx[i];
		}
	}
	return min_tpb;
}

void testm1m2_vary_tpb_plot() {
	int z = 0;
	char file_name[100];
	setup_streams();
	create_memory_pool(BITLENGTH);
	int offset = BITLENGTH;
	for (int j = 0; j < NUMKERNELS; j++) {
		THREADSPERBLOCK[j] = 2 * BITLENGTH;
		clear_tops();
		int size = 0;
		for (int i = 0; i < NUMSAMPLES && THREADSPERBLOCK[j] <= 1024; i++) {
			sample_no = i;
			execute(&z);
			sx[i] = THREADSPERBLOCK[j];
			THREADSPERBLOCK[j] *= 2;
			size++;
			clear_tops();
		}
		strcpy(file_name, "NumberOfThreadsPerBlockVsLatency_");
		strcat(file_name, kernel_names[j]);
		strcat(file_name, ".png");
		plot_graph(file_name, size);
		THREADSPERBLOCK[j] = find_miny();
	}
	clear_tops();
	printf("Kernel Name\t:\tThreads Per Block\n");
	for (int j = 0; j < NUMKERNELS; j++) {
		printf("%s\t:\t%d\n", kernel_names[j], THREADSPERBLOCK[j]);
	}

}
#endif

void testm1m2_multi_stream() {
	pthread_t *threads = new pthread_t[MAXNUMCPUTHREADS];
	int *tids = new int[MAXNUMCPUTHREADS];
	setup_streams();
	create_memory_pool(BITLENGTH);

#if ENABLEPARAMDETERM
	//determine optimal runtime grid dimensions by running few tests..
	testm1m2_vary_tpb_plot();
#else
	//Use the last optimal parameters..
	THREADSPERBLOCK[PMUL] = 128;
	THREADSPERBLOCK[PADD] = 128;
	THREADSPERBLOCK[PSUB] = 512;
	THREADSPERBLOCK[RSHIFT] = 1024;
	THREADSPERBLOCK[PCOPY] = 256;
	THREADSPERBLOCK[PCOPYCOND] = 1024;
	THREADSPERBLOCK[CONVERTTOBASE] = 512;
#endif
	for (int i = 0; i < MAXNUMCPUTHREADS; i++) {
		tids[i] = i;
		int resp = pthread_create(&threads[i], NULL, execute, (void*) &tids[i]);
		if (resp) {
			printf("Error spawning CPU thread %d\n", resp);
		}
	}

	for (int i = 0; i < MAXNUMCPUTHREADS; i++) {
		pthread_join(threads[i], NULL);
	}
	destroy_streams();
	cleanup();
}

int num_req = 0;
request_batch *reqs[MAXREQUESTSPERCPUTHREAD*MAXNUMCPUTHREADS + MAXREQUESTSPERPY * MAXNUMCPUSERIALTHREADS];

void *schedule(void *reqs) {
	execute(reqs);
	//execute_serial(reqs);
}

request_batch *create_batch(int tid, int num_req_per){
	int k = BITLENGTH;
	request_batch *req_batch = (request_batch *) malloc(sizeof(request_batch));
	req_batch->batch_size = num_req_per;
	req_batch->request_id = tid;
	req_batch->x = hallocate(MEM2K, k, num_req_per, tid);
	req_batch->y = hallocate(MEM2K, k, num_req_per, tid);
	req_batch->m = hallocate(MEM2K, k, num_req_per, tid);
	req_batch->res = hallocate(MEM2K, k, num_req_per, tid);
	req_batch->mbar = hallocate(MEM2K, k, num_req_per, tid);
	req_batch->rinv = hallocate(MEM2K, k, num_req_per, tid);
	req_batch->r2modm = hallocate(MEM2K, k, num_req_per, tid);
	req_batch->m1m2 = hallocate(MEM2K, k, num_req_per, tid);
	req_batch->q = hallocate(MEMK, k, num_req_per, tid);
	req_batch->qinv = hallocate(MEMK, k, num_req_per, tid);
	return req_batch;
}

void collect(request_batch *req, int force_start) {
	if (num_req == 0) {
		start();	//start timer for batch formation period..
	}
	int k = BITLENGTH;
	reqs[num_req++] = req;
	long elapsed = stop();
	int tid = 0;
	//max requests or timeout
	if (num_req == MAXNUMCPUTHREADS * MAXREQUESTSPERCPUTHREAD + MAXREQUESTSPERPY * MAXNUMCPUSERIALTHREADS
			/*|| elapsed >= BATCHTIMEOUT*/ || force_start) {
		printf("Allocating Requests To CPUs and GPUs..\n");
		pthread_t *threads = new pthread_t[MAXNUMCPUTHREADS + MAXNUMCPUSERIALTHREADS];

		int num_req_gpu = MAXNUMCPUTHREADS * MAXREQUESTSPERCPUTHREAD;//num_req*GPUCPUREQRATIO;
		int num_req_cpu = MAXREQUESTSPERPY * MAXNUMCPUSERIALTHREADS;

		int num_req_perg = num_req_gpu / MAXNUMCPUTHREADS;
		int num_req_perc = MAXREQUESTSPERPY;//MAXNUMCPUSERIALTHREADS!=0?num_req_cpu / MAXNUMCPUSERIALTHREADS:0;
		start();
		int thrs = 0;
		if(num_req_perg > 0 && SERIAL != 1){
			for (int tid = 0; tid < MAXNUMCPUTHREADS; tid++) {
				request_batch *req_batch = create_batch(tid,num_req_perg);
				printf(
						"Creating a batch of %d requests for processing on GPUTHREAD %d.\n",
						num_req_perg, tid);

				int offseti = tid*num_req_perg;
				//GPU requests
				for (int i = 0; i < num_req_perg;
						i++) {
					//M1
					for (int j = 0; j < BITLENGTH; j++) {
						req_batch->x[i * k + j] = reqs[offseti+i]->x[j];
						req_batch->y[i * k + j] = reqs[offseti+i]->y[j];
						req_batch->m[i * k + j + 0] = reqs[offseti+i]->m[j];
						req_batch->rinv[i * k + j + 0] = reqs[offseti+i]->rinv[j];
						req_batch->mbar[i * k + j + 0] = reqs[offseti+i]->mbar[j];
						req_batch->r2modm[i * k + j + 0] = reqs[offseti+i]->r2modm[j];
						req_batch->q[i * k + j + 0] = reqs[offseti+i]->q[j];
						req_batch->qinv[i * k + j + 0] = reqs[offseti+i]->qinv[j];
					}
					//M2
					for (int j = 0; j < k; j++) {
						req_batch->x[(num_req_perg + i) * k + j + 0] = reqs[offseti+i]->x[k
								+ j];
						req_batch->y[(num_req_perg + i) * k + j + 0] = reqs[offseti+i]->y[k
								+ j];
						req_batch->m[(num_req_perg + i) * k + j + 0] = reqs[offseti+i]->m[k
								+ j];
						req_batch->rinv[(num_req_perg + i) * k + j + 0] =
								reqs[offseti+i]->rinv[k + j];
						req_batch->mbar[(num_req_perg + i) * k + j + 0] =
								reqs[offseti+i]->mbar[k + j];
						req_batch->r2modm[(num_req_perg + i) * k + j + 0] =
								reqs[offseti+i]->r2modm[k + j];
					}
				}
				//load-balance,schedule and execute..
				int resp = pthread_create(&threads[thrs++], NULL, schedule,
						(void*) req_batch);
			}
		}

#if MAXNUMCPUSERIALTHREADS
		load_modex();
		for(int tid=0; tid<MAXNUMCPUSERIALTHREADS && num_req_perc >0;tid++){
			request_batch *req_batch = create_batch(-1,num_req_perc);
			printf(
					"Creating a batch of %d requests for processing on CPUTHREAD %d.\n",
					num_req_perc, tid);

			int offseti = num_req_gpu + tid*num_req_perc;
			//GPU requests
			for (int i = 0; i < num_req_perc;
					i++) {
				//M1
				for (int j = 0; j < BITLENGTH; j++) {
					req_batch->x[i * k + j] = reqs[offseti+i]->x[j];
					req_batch->y[i * k + j] = reqs[offseti+i]->y[j];
					req_batch->m[i * k + j + 0] = reqs[offseti+i]->m[j];
					req_batch->rinv[i * k + j + 0] = reqs[offseti+i]->rinv[j];
					req_batch->mbar[i * k + j + 0] = reqs[offseti+i]->mbar[j];
					req_batch->r2modm[i * k + j + 0] = reqs[offseti+i]->r2modm[j];
					req_batch->q[i * k + j + 0] = reqs[offseti+i]->q[j];
					req_batch->qinv[i * k + j + 0] = reqs[offseti+i]->qinv[j];
				}
				//M2
				for (int j = 0; j < k; j++) {
					req_batch->x[(num_req_perc + i) * k + j + 0] = reqs[offseti+i]->x[k
							+ j];
					req_batch->y[(num_req_perc + i) * k + j + 0] = reqs[offseti+i]->y[k
							+ j];
					req_batch->m[(num_req_perc + i) * k + j + 0] = reqs[offseti+i]->m[k
							+ j];
					req_batch->rinv[(num_req_perc + i) * k + j + 0] =
							reqs[offseti+i]->rinv[k + j];
					req_batch->mbar[(num_req_perc + i) * k + j + 0] =
							reqs[offseti+i]->mbar[k + j];
					req_batch->r2modm[(num_req_perc + i) * k + j + 0] =
							reqs[offseti+i]->r2modm[k + j];
				}
			}
			//load-balance,schedule and execute..
			int resp = pthread_create(&threads[thrs++], NULL, execute_serial,
					(void*) req_batch);
		}
#endif
		for (int i = 0; i < thrs; i++) {
			pthread_join(threads[i], NULL);
		}
		double lat = stop();
		printf("Total Execution Time:%lf ms\n",lat);
		for (int tid = 0; tid < MAXNUMCPUTHREADS; tid++) {
			hdeallocate(MEM2K, k, num_req_perg, tid);
			hdeallocate(MEM2K, k, num_req_perg, tid);
			hdeallocate(MEM2K, k, num_req_perg, tid);
			hdeallocate(MEM2K, k, num_req_perg, tid);
			hdeallocate(MEM2K, k, num_req_perg, tid);
			hdeallocate(MEM2K, k, num_req_perg, tid);
			hdeallocate(MEM2K, k, num_req_perg, tid);
			hdeallocate(MEM2K, k, num_req_perg, tid);
			hdeallocate(MEMK, k, num_req_perg, tid);
			hdeallocate(MEMK, k, num_req_perg, tid);

		}
		for(int i=0;i<num_req;i++)
			free(reqs[i]);
		num_req = 0;
	}
}

void simulate() {
#if !SERIAL
	cudaGetDeviceCount(&num_of_devices);
	printf("Found %d cuda enabled devices..\n", num_of_devices);
	//num_of_devices = 1;
	printf("Setting up streams..\n");
	setup_streams();
#endif
	printf("Creating memory pool..\n");
	create_memory_pool(BITLENGTH);
	printf("Ready to receive requests..\n");

#if ENABLEPARAMDETERM
	//determine optimal runtime grid dimensions by running few tests..
	printf("Determining optimal runtime parameters..\n");
	testm1m2_vary_tpb_plot();
#else
	//Use the last optimal parameters..
	THREADSPERBLOCK[PMUL] = 128;
	THREADSPERBLOCK[PADD] = 128;
	THREADSPERBLOCK[PSUB] = 512;
	THREADSPERBLOCK[RSHIFT] = 1024;
	THREADSPERBLOCK[PCOPY] = 256;
	THREADSPERBLOCK[PCOPYCOND] = 1024;
	THREADSPERBLOCK[CONVERTTOBASE] = 512;
#endif
	int num_reqs = MAXREQUESTSPERCPUTHREAD*MAXNUMCPUTHREADS + MAXREQUESTSPERPY * MAXNUMCPUSERIALTHREADS;
	for (int i = 0; i < num_reqs; i++) {
		request_batch *req = (request_batch *) malloc(sizeof(request_batch));
		int k = BITLENGTH;
		req->x = (radix_type *) malloc(2 * k * sizeof(radix_type));
		req->y = (radix_type *) malloc(2 * k * sizeof(radix_type));
		req->m = (radix_type *) malloc(2 * k * sizeof(radix_type));
		req->res = (radix_type *) malloc(2 * k * sizeof(radix_type));
		req->mbar = (radix_type *) malloc(2 * k * sizeof(radix_type));
		req->rinv = (radix_type *) malloc(2 * k * sizeof(radix_type));
		req->r2modm = (radix_type *) malloc(2 * k * sizeof(radix_type));
		req->m1m2 = (radix_type *) malloc(2 * k * sizeof(radix_type));
		req->q = (radix_type *) malloc(k * sizeof(radix_type));
		req->qinv = (radix_type *) malloc(k * sizeof(radix_type));
		for (int j = 0; j < k; j += 2) {
			req->x[j] = 8185;
			req->x[j + 1] = 9350;

			req->y[j] = 38117;
			req->y[j + 1] = 18971;

			req->m[j + 0] = 18883;
			req->m[j + 1] = 47293;

			req->rinv[j + 0] = 16735;
			req->rinv[j + 1] = 21234;

			req->mbar[j + 0] = 4373;
			req->mbar[j + 1] = 29425;

			req->r2modm[j + 0] = 9630;
			req->r2modm[j + 1] = 6723;

			req->q[j + 0] = 19929;
			req->q[j + 1] = 32870;

			req->qinv[j + 0] = 38310;
			req->qinv[j + 1] = 16001;
		}

		for (int j = 0; j < k; j += 2) {
			req->x[k + j + 0] = 8185;
			req->x[k + j + 1] = 9350;

			req->y[k + j + 0] = 54089;
			req->y[k + j + 1] = 29809;

			req->m[k + j + 0] = 19929;
			req->m[k + j + 1] = 32870;

			req->rinv[k + j + 0] = 55775;
			req->rinv[k + j + 1] = 16560;

			req->mbar[k + j + 0] = 40343;
			req->mbar[k + j + 1] = 33018;

			req->r2modm[k + j + 0] = 2968;
			req->r2modm[k + j + 1] = 13939;
		}

		collect(req,i==num_reqs-1?1:0);
	}
	printf("Simulation finished..\n");
#if !SERIAL
	printf("Destroying streams..\n");
	destroy_streams();
	printf("Cleaning up..\n");
	cleanup();
#endif
}

extern void decrypt_batch(JNIEnv *env, jobject jobj, jint jk, jobjectArray n, jobjectArray pd, jobjectArray qd, jobjectArray e, jobjectArray p, jobjectArray q, jobjectArray emsg, jobjectArray dmsg, jobjectArray qinv, jint jnum_req){
	radix_type *demsg, *dpd, *dqd, *dp, *dq, *dqinv, *dres, *dm1, *dm2;
	int num_req = (int)jnum_req;
	int k = (int)jk;	
	cudaMallocManaged((void **) &demsg, num_req*k*sizeof(radix_type));
	cudaMallocManaged((void **) &dpd, num_req*k*sizeof(radix_type));
	cudaMallocManaged((void **) &dqd, num_req*k*sizeof(radix_type));
	cudaMallocManaged((void **) &dp, num_req*k*sizeof(radix_type));
	cudaMallocManaged((void **) &dq, num_req*k*sizeof(radix_type));
	cudaMallocManaged((void **) &dqinv, num_req*k*sizeof(radix_type));
	cudaMallocManaged((void **) &dres, num_req*2*k*sizeof(radix_type));
	cudaMallocManaged((void **) &dm1, num_req*k*sizeof(radix_type));
	cudaMallocManaged((void **) &dm2, num_req*k*sizeof(radix_type));
	
	convert_arr(env, emsg, demsg, num_req, k);
	convert_arr(env, pd, dpd, num_req, k);
	convert_arr(env, qd, dqd, num_req, k);
	convert_arr(env, p, dp, num_req, k);
	convert_arr(env, q, dq, num_req, k);
	convert_arr(env, qinv, dqinv, num_req, k);
	convert_arr(env, dmsg, dres, num_req, 2*k);
	for(int i=0;i<num_req;i++){
		std_print(&dp[i*k],k);
		std_print(&demsg[i*k],k);
	}
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 6);

	memset(dm1,0,num_req*k*sizeof(radix_type));
	memset(dm2,0,num_req*k*sizeof(radix_type));
	//internal batch and allocate blocks to requests
	int num_blocks = num_req/THREADSPERBLOCK;
	if(num_req%THREADSPERBLOCK!=0)
		num_blocks++;
	crt_m<<<num_blocks,THREADSPERBLOCK>>>(demsg,dpd,k,dp,dq,dm1,dm2,num_req);

	calmerge_m1_m2<<<num_blocks,THREADSPERBLOCK>>>(demsg, dpd, dqd, k, dp, dq, dqinv, dm1, dm2, dres,num_req);
	cudaError_t err = cudaDeviceSynchronize();
  	if (err != cudaSuccess) fprintf(stderr,"Error Encountered---%s\n", cudaGetErrorString(err));
	/*for(int l=0;l<num_req;l++){
		fprintf(stderr,"%d:",l);		
		for(int j=0;j<2*k;j++){
			fprintf(stderr,"%d", dres[l*k+j]);
		}
		fprintf(stderr,"\n");
	}*/
  	for(int i=0;i<num_req;i++){
		jchar *message = (jchar *)calloc(sizeof(jchar), 2*k);
		for (int j = 0; j<2*k; j++) {
				message[j] = (jchar)(dres[i*2*k+j]+'0');
		}
		jcharArray  arrMsg = env->NewCharArray(2*k);
		env->SetCharArrayRegion(arrMsg, 0, 2*k, message);
		env->SetObjectArrayElement(dmsg,i,arrMsg);
		//dmsg[i] = arrMsg;
  	}
	cudaFree(demsg);
	cudaFree(dpd);
	cudaFree(dqd);
	cudaFree(dp);
	cudaFree(dq);
	cudaFree(dqinv);
	cudaFree(dres);
}

extern void execute_test() {
	//test_pmul();
	//test_padd();
	//testm1m2_multi_stream();
	//testm1m2_vary_tpb_plot();
	//testm1m2_vary_requests_plot();
}
void pretty_print(double *x, double *y,int size, char *xh, char *yh){
	printf("%s\t|\t%s\n",xh,yh);
	printf("------------------------------------\n");
	for(int i=0;i<size;i++){
		printf("%.2lf\t\t|\t%.2lf\n",x[i],y[i]);
	}

}

void test_sharing_ratios(int op){
	double x[10];
	double y[10];
	GPUCPUADDRATIO= 1.0;
	GPUCPUMULRATIO= 1.0;
	GPUCPUSUBRATIO= 1.0;
	double min = 24*60*60*1000;
	double minv = 1.0;
	switch(op){
	case 1:
		for(int i=0;i<10;i++){
				x[i] = GPUCPUMULRATIO;
				start();
				simulate();
				y[i] = stop();
				if(y[i]<min){
					min = y[i];
					minv = GPUCPUMULRATIO;
				}
				GPUCPUMULRATIO -= 0.1;
			}
		#if !HPC
		plot_graph_param("MULSharingRatio.png", x, y, 10, "Sharing Ratio","Latency(ms)", "GPU-CPU Mul Task Sharing Vs Latency");
		#endif
		#if HPC
		pretty_print(x,y,10,"MUL Sharing Ratio", "Latency(ms)");
		#endif
			break;
	case 2:
		for(int i=0;i<10;i++){
			x[i] = GPUCPUADDRATIO;
			start();
			simulate();
			y[i] = stop();
			if(y[i]<min){
				min = y[i];
				minv = GPUCPUADDRATIO;
			}
			GPUCPUADDRATIO -= 0.1;
		}
	#if !HPC
	plot_graph_param("ADDSharingRatio.png", x, y, 10, "Sharing Ratio","Latency(ms)", "GPU-CPU Add Task Sharing Vs Latency");
	#endif
	#if HPC
	pretty_print(x,y,10,"ADD Sharing Ratio", "Latency(ms)");
	#endif
			break;
	case 3:
		for(int i=0;i<10;i++){
			x[i] = GPUCPUSUBRATIO;
			start();
			simulate();
			y[i] = stop();
			if(y[i]<min){
				min = y[i];
				minv = GPUCPUSUBRATIO;
			}
			GPUCPUSUBRATIO -= 0.1;
		}
	#if !HPC
	plot_graph_param("SUBSharingRatio.png", x, y, 10, "Sharing Ratio","Latency(ms)", "GPU-CPU Sub Task Sharing Vs Latency");
	#endif
	#if HPC
	pretty_print(x,y,10,"SUB Sharing Ratio", "Latency(ms)");
	#endif
			break;
	}
}

void test_hetro_ratio(){
	double x[10];
	double y[10];
	GPUCPUREQRATIO= 1.0;
	double min = 24*60*60*1000;
	double minv = 1.0;
	for(int i=0;i<10;i++){
		x[i] = GPUCPUREQRATIO;
		start();
		simulate();
		y[i] = stop();
		if(y[i]<min){
			min = y[i];
			minv = GPUCPUREQRATIO;
		}
		GPUCPUREQRATIO -= 0.1;
	}
	#if !HPC
	plot_graph_param("REQRatio.png", x, y, 10, "Sharing Ratio","Latency(ms)", "GPU-CPU Request Sharing Vs Latency");
	#endif
	#if HPC
	pretty_print(x,y,10,"REQ Sharing Ratio", "Latency(ms)");
	#endif
}

void cpu_c(){
	printf("Running CPU C Version.");
	GPUCPUMULRATIO = 0.0;
	GPUCPUADDRATIO = 0.0;
	GPUCPUSUBRATIO = 0.0;
	cpu_v = 1;
	simulate();
}

void cpu_py(){
	printf("Running CPU C-Py Version.");
	GPUCPUMULRATIO = 0.0;
	GPUCPUADDRATIO = 0.0;
	GPUCPUSUBRATIO = 0.0;
	cpu_v = 2;
	simulate();
}

void gpu_only(){
	printf("Running GPU Only Version.");
	GPUCPUMULRATIO = 1.0;
	GPUCPUADDRATIO = 1.0;
	GPUCPUSUBRATIO = 1.0;
	simulate();
}

void gpu_cpu(){
	printf("Running GPU-CPU Version.");
	GPUCPUMULRATIO = 1.0;
	GPUCPUADDRATIO = 0.0;
	GPUCPUSUBRATIO = 1.0;
	simulate();
}

void run(){
#if SERIAL
	if(cpu_v == 1)
		cpu_c();
	else
		cpu_py();
#endif

#if !SERIAL
	if(gpu_v == 1)
		gpu_only();
	else
		gpu_cpu();
#endif
}

int main(int argc, char *argv[]) {

	int op_type = 1;
	if(argc>=3){
		GPUCPUMULRATIO = atof(argv[1]);
		GPUCPUADDRATIO = atof(argv[2]);
		GPUCPUSUBRATIO = atof(argv[3]);
	}

	if(argc == 2){
		op_type = atoi(argv[1]);
	}
	printf("Task Sharing Ratios: MUL(%lf) ADD(%lf) SUB(%lf)\n",GPUCPUMULRATIO, GPUCPUADDRATIO, GPUCPUSUBRATIO);
	run();
}
