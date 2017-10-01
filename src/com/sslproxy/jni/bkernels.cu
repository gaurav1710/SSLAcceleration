#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include "config.h"
#include "gpu_algos.h"
#include "/usr/lib/jvm/java-8-openjdk-amd64/include/jni.h"


//Number of CUDA enabled cards
int num_of_devices = 1;

//compute threads per block = multiple of number of threads in a warp(calculated for maximum occupancy per SM)
int THREADSPERBLOCK[] = { 512, 512, 512, 512, 512, 512, 512 };

cudaStream_t *streams[1];

inline void print_array(radix_type *a, int n){
}

inline void setup_streams() {

	streams[0] = new cudaStream_t[1];
	cudaError_t error;
	error = cudaStreamCreate(streams[0]);
	if (error != cudaSuccess){
		printf("Error Encountered stream_setup---%s\n",
					cudaGetErrorString(error));
	}

}

inline void destroy_streams() {
	cudaError_t error;
	error = cudaStreamDestroy(*streams[0]);
	if (error != cudaSuccess){
		printf("Error Encountered stream_destroy---%s\n",
						cudaGetErrorString(error));
	}
}

void copy_zeros_to_device(radix_type *da, int words, int streamid) {
	radix_type *zeros = (radix_type *)malloc(words*sizeof(radix_type));
	memset(zeros, 0, words * sizeof(radix_type));
	cudaMemcpyAsync(da, zeros, words * sizeof(radix_type),
			cudaMemcpyHostToDevice, *streams[streamid]);
	//free(zeros);
}
void copy_data_to_device(radix_type *ha, radix_type *da, int words,
		int streamid) {
	cudaMemcpyAsync(da, ha, words * sizeof(radix_type), cudaMemcpyHostToDevice,
			*streams[streamid]);
}
void copy_data_to_host(radix_type *ha, radix_type *da, int words,
		int streamid) {
	cudaMemcpyAsync(ha, da, words * sizeof(radix_type), cudaMemcpyDeviceToHost,
			*streams[streamid]);
}

void copy_data_to_device_ds(radix_type *ha, radix_type *da, int words,
		int streamid) {
	cudaMemcpyAsync(da, ha, words * sizeof(radix_type), cudaMemcpyHostToDevice,
			*streams[streamid]);
}
void copy_data_to_host_ds(radix_type *ha, radix_type *da, int words,
		int streamid) {
	cudaMemcpyAsync(ha, da, words * sizeof(radix_type), cudaMemcpyDeviceToHost,
			*streams[streamid]);
}

radix_type *allocate(int size) {
	radix_type *mem_pointer = NULL;
	CUDAMEMHOST((void **) &mem_pointer, size * sizeof(radix_type));
	return mem_pointer;
}

radix_type *hallocate(int size) {
	radix_type *mem_ponter = (radix_type *)malloc(size * sizeof(radix_type));	
	return mem_ponter;
}


void deallocate(radix_type *mem) {
	cudaFree(mem);
}

void hdeallocate(radix_type *mem) {
	free(mem);
}

inline void extract_req_bits(short_radix_type *base2y,
		short_radix_type *base2_bits, int ind, int num_req, int k,
		int streamid, int base) {
	extract_bits<<<1, num_req, 0, *streams[streamid]>>>(base2y, base2_bits,
			ind, k, base);
	cudaError_t err = cudaStreamSynchronize(*streams[streamid]);
	if (err != cudaSuccess)
		printf("Error Encountered in Extract_bits---%s\n",
				cudaGetErrorString(err));
}

inline void parallel_mult(radix_type *m, radix_type *n, int k, radix_type *res,
		int stridem, int striden, int num_req, int streamid, int base, int basem1) {
	//THREADSPERBLOCK[0]
	int streams_req =  1;
	int num_blocks = (num_req * k) / THREADSPERBLOCK[0]
			+ ((num_req * k) % THREADSPERBLOCK[0] != 0);
	int num_threads = THREADSPERBLOCK[0];
	if ((num_req * k) < THREADSPERBLOCK[0])
		num_threads = num_req * k;
	int num_req_per_block = num_threads / k;

	for (int i = 0; i < streams_req; i++) {
		pmul<<<num_blocks, num_threads,
				5 * k * num_req_per_block * sizeof(radix_type),
				*streams[streamid]>>>(m, n, k, res, stridem, striden,
				num_threads, base, basem1);
	}

	for (int i = 0; i < streams_req; i++) {
		cudaError_t err = cudaStreamSynchronize(streams[streamid][i]);
		if (err != cudaSuccess)
			printf("Error Encountered in Parallel_Mult---%s\n",
					cudaGetErrorString(err));
	}
}
inline void parallel_add(radix_type *a, radix_type *b, int k, int num_req,
		int stridea, int strideb, radix_type *residue_carry, int streamid, int base, int basem1) {
	//THREADSPERBLOCK[1]
	int num_blocks = (num_req * k) / THREADSPERBLOCK[1]
			+ ((num_req * k) % THREADSPERBLOCK[1] != 0);
	int num_threads = THREADSPERBLOCK[1];
	if ((num_req * k) < THREADSPERBLOCK[1])
		num_threads = num_req * k;
	int num_req_per_block = num_threads / k;

	if(num_req!=0){
		padd<<<num_blocks, num_threads,
					(2 * k + 1) * num_req_per_block * sizeof(radix_type),
					*streams[streamid]>>>(a, b, k, num_req, stridea, strideb,
					residue_carry, num_threads, base, basem1);
	}
	cudaError_t err = cudaStreamSynchronize(*streams[streamid]);
	if (err != cudaSuccess)
		printf("Error Encountered in Parallel_Add---%s\n",
				cudaGetErrorString(err));

}

inline void parallel_sub(radix_type *a, radix_type *b, int k, radix_type *res,
		radix_type *add_carry, int num_req, int stridea, int strideb,
		int compare, int streamid, int base, int basem1) {
	//THREADSPERBLOCK[3]
	int num_blocks = (num_req * k) / THREADSPERBLOCK[3]
			+ ((num_req * k) % THREADSPERBLOCK[3] != 0);
	int num_threads = THREADSPERBLOCK[3];
	if ((num_req * k) < THREADSPERBLOCK[3])
		num_threads = num_req * k;
	int num_req_per_block = num_threads / k;

	if(num_req!=0){
		psub<<<num_blocks, num_threads,
				2 * k * num_req_per_block * sizeof(radix_type), *streams[streamid]>>>(
				a, b, k, res, add_carry, stridea, strideb, compare, num_threads, base, basem1);
	}

	cudaError_t err = cudaStreamSynchronize(*streams[streamid]);
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
	right_shift<<<num_blocks, num_threads, 0, *streams[streamid]>>>(a,
			stridea, k, res, num_threads);
	cudaError_t err = cudaStreamSynchronize(*streams[streamid]);
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
	pcopy<<<num_blocks, num_threads, 0, *streams[streamid]>>>(a, b, stridea,
			strideb, k, num_threads);
	cudaError_t err = cudaStreamSynchronize(*streams[streamid]);
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
	pcopy_wcondition<<<num_blocks, num_threads, 0, *streams[streamid]>>>(a, b,
			base2, k, stridea, strideb, num_threads);
	cudaError_t err = cudaStreamSynchronize(*streams[streamid]);
	if (err != cudaSuccess)
		printf("Error Encountered in Parallel_copy_wcondition---%s\n",
				cudaGetErrorString(err));
}

inline void parallel_base_convert(radix_type *x, short_radix_type *y, int k,
		int base, int num_req, int streamid, int BASE) {
	//THREADSPERBLOCK[6]
	int num_blocks = num_req / THREADSPERBLOCK[6];
	if (num_req % THREADSPERBLOCK[6] != 0)
		num_blocks++;
	convert_to_base<<<num_blocks, THREADSPERBLOCK[6], 0, *streams[streamid]>>>(
			x, y, k, 2, num_req, THREADSPERBLOCK[6], BASE);
	cudaError_t err = cudaStreamSynchronize(*streams[streamid]);
	if (err != cudaSuccess)
		printf("Error Encountered in Parallel_base_convert---%s\n",
				cudaGetErrorString(err));
}

unsigned long convert_2base10(radix_type *a, int k, int basem1) {
	unsigned long res = 0;
	unsigned long pow = 1;

	for (int i = 0; i < k; i++) {
		res += pow * a[i];
		pow *= (basem1 + 1);
	}
	return res;
}

inline void montgomery(radix_type *a, radix_type *b, int k, radix_type *m,
		radix_type *rinv, radix_type *mbar, radix_type *res, int num_req,
		int tid, int base, int basem1) {

	radix_type *T, *M, *U, *carry, *addition_carry;

	T = allocate(2*k*num_req);
	M = allocate(2*k*num_req);
	U = allocate(2*k*num_req);
	carry = allocate(2*k*num_req);
	addition_carry = allocate(num_req);

	copy_zeros_to_device(T, 2 * k * num_req, tid);

	//T=a*b
	parallel_mult(a, b, k, T, k, k, num_req, tid, base, basem1);

	if (DEBUG) {
		printf("montgomery-MulT:");
		print_array(T, 2 * k);
	}

	copy_zeros_to_device(M, 2 * k * num_req, tid);

	//M=T*m'
	parallel_mult(T, mbar, k, M, 2 * k, k, num_req, tid, base, basem1);
	if (DEBUG) {
		printf("montgomery-MulM:");
		print_array(M, 2 * k);
	}
	copy_zeros_to_device(U, 2 * k * num_req, tid);
	//U=M*m
	parallel_mult(M, m, k, U, 2 * k, k, num_req, tid, base, basem1);
	if (DEBUG) {
		printf("montgomery-MulU:");
		print_array(U, 2 * k);
	}
	copy_zeros_to_device(addition_carry, num_req, tid);

	//U=T+M*m
	parallel_add(U, T, 2 * k, num_req, 2 * k, 2 * k, addition_carry, tid , base, basem1);
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

	parallel_sub(carry, m, k, res, addition_carry, num_req, 2 * k, k, 1, tid, base, basem1);
	if (DEBUG) {
		printf("montgomery-SubRes:");
		print_array(res, k);
	}
	deallocate(T);
	deallocate(M);
	deallocate(U);
	deallocate(carry);
	deallocate(addition_carry);
}

inline void hmod_exp_sqmul(radix_type *x, radix_type *y, int k, radix_type *m,
		radix_type *res, radix_type *rinv, radix_type *mbar, radix_type *r2modm,
		int num_req, int tid, int base, int basem1) {
	radix_type *temp = allocate(k*num_req);
	short_radix_type *base2y = allocate(base*k*num_req);
	radix_type *ones = allocate(k*num_req);
	short_radix_type *base2_bits = allocate(num_req);
	radix_type *htemp = hallocate(k*num_req);
	radix_type *hones = hallocate(k*num_req);

	memset(htemp, 0, num_req * k * sizeof(radix_type));
	memset(hones, 0, num_req * k * sizeof(radix_type));

	for (int i = 0; i < num_req; i++) {
		htemp[i * k] = 1;	//initialize with 1
		hones[i * k] = 1;
	}
	copy_data_to_device(htemp, temp, k * num_req, tid);
	copy_data_to_device(hones, ones, k * num_req, tid);

	copy_zeros_to_device(res, k * num_req, tid);

	parallel_base_convert(y, base2y, k, 2, num_req, tid, base);

	montgomery(temp, r2modm, k, m, rinv, mbar, res, num_req, tid,base,basem1);

	if (DEBUG) {
		printf("R2*1 mod m:");
		print_array(res, k);
	}
	parallel_copy(temp, res, k, k, k, num_req, tid);

	copy_zeros_to_device(res, k * num_req, tid);
	montgomery(x, r2modm, k, m, rinv, mbar, res, num_req, tid,base,basem1);

	if (DEBUG) {
		printf("R2*x mod m:");
		print_array(res, k);
	}
	parallel_copy(x, res, k, k, k, num_req, tid);

	for (int i = base * k - 1; i >= 0; i--) {
		copy_zeros_to_device(res, k * num_req, tid);
		montgomery(temp, temp, k, m, rinv, mbar, res, num_req, tid,base,basem1);

		parallel_copy(temp, res, k, k, k, num_req, tid);
		copy_zeros_to_device(res, k * num_req, tid);
		montgomery(temp, x, k, m, rinv, mbar, res, num_req, tid,base,basem1);

		extract_req_bits(base2y, base2_bits, i, num_req, k, tid, base);
		parallel_copy_wcondition(temp, res, base2_bits, k, k, k, num_req, tid);
	}

	copy_zeros_to_device(res, k * num_req, tid);
	montgomery(temp, ones, k, m, rinv, mbar, res, num_req, tid,base,basem1);
	deallocate(temp);
	deallocate(base2y);
	deallocate(ones);
	deallocate(base2_bits);
	hdeallocate(htemp);
	hdeallocate(hones);
}

inline void montgomery_with_conv(radix_type *a, radix_type *b, radix_type *m,
		radix_type *rinv, radix_type *mbar, radix_type *r2modm, radix_type *res,
		int k, int num_req, int tid, int base, int basem1) {
	radix_type *ar, *br, *abr, *ones;
	ar = allocate(k*num_req);
	br = allocate(k*num_req);
	abr = allocate(k*num_req);
	ones = allocate(k*num_req);
	radix_type *hones = hallocate(k*num_req);
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
	montgomery(a, r2modm, k, m, rinv, mbar, ar, num_req, tid,base,basem1);
	//b*r mod m
	montgomery(b, r2modm, k, m, rinv, mbar, br, num_req, tid,base,basem1);
	//a*b*r mod m
	montgomery(ar, br, k, m, rinv, mbar, abr, num_req, tid,base,basem1);
	//a*b mod m
	montgomery(abr, ones, k, m, rinv, mbar, res, num_req, tid,base,basem1);
	deallocate(ar);
	deallocate(br);
	deallocate(abr);
	deallocate(ones);
	hdeallocate(hones);
}

void calmerge_m1_m2(radix_type *c, radix_type *e, radix_type *m, int k,
		radix_type *rinv, radix_type *mbar, radix_type *r2modm, radix_type *q,
		radix_type *qinv, radix_type *res, int num_req, int tid, int base, int basem1) {
	radix_type *t1, *t2, *t3, *mont;
	radix_type *M, *addition_carry;

	//Calculate M1 and M2
	M = allocate(2*k*num_req);

	copy_zeros_to_device(M, k * 2 * num_req, tid);

	hmod_exp_sqmul(c, e, k, m, M, rinv, mbar, r2modm, 2 * num_req, tid, base, basem1);

	t1 = allocate(k*num_req);
	t2 = allocate(2*k*num_req);
	t3 = allocate(2*k*num_req);
	addition_carry = allocate(num_req);
	mont = allocate(k*num_req);

	copy_zeros_to_device(t1, k * num_req, tid);
	copy_zeros_to_device(addition_carry, num_req, tid);
	if (DEBUG) {
		printf("M1=%lu\n", convert_2base10(M, k, basem1));
		printf("M2=%lu\n", convert_2base10(&M[num_req * k], k, basem1));
	}
	//M1-M2
	parallel_sub(M, &M[num_req * k], k, t1, addition_carry, num_req, k, k, 0,
			tid,base,basem1);
	if (DEBUG) {
		printf("M1-M2=%lu\n", convert_2base10(t1, k,basem1));
	}
	copy_zeros_to_device(t2, 2 * k * num_req, tid);
	copy_zeros_to_device(t3, 2 * k * num_req, tid);
	copy_zeros_to_device(mont, k * num_req, tid);

	//(M1-M2)*(qinv mod p) mod p
	montgomery_with_conv(t1, qinv, m, rinv, mbar, r2modm, mont, k, num_req,
			tid,base,basem1);
	if (DEBUG) {
		printf("(M1-M2)*q^inv mod p=%lu\n", convert_2base10(mont, k, basem1));
	}
	copy_zeros_to_device(t2, 2 * k * num_req, tid);
	copy_zeros_to_device(t3, 2 * k * num_req, tid);

	//(M1-M2)*(qinv mod p)*q
	parallel_mult(mont, q, k, t2, k, k, num_req, tid,base,basem1);
	if (DEBUG) {
		printf("((M1-M2)*q^inv mod p)*q=%lu\n", convert_2base10(t2, 2 * k, basem1));
	}
	parallel_copy(res, t2, 2 * k, 2 * k, 2 * k, num_req, tid);

	copy_zeros_to_device(addition_carry, num_req, tid);
	//M2 + (M1-M2)*(qinv mod p)*q
	parallel_add(res, &M[num_req * k], k, num_req, 2 * k, k, addition_carry,
			tid,base,basem1);
	if (DEBUG) {
		printf("M2+((M1-M2)*q^inv mod p)*q=%lu\n", convert_2base10(res, 2 * k, basem1));
	}
	deallocate(t1);
	deallocate(t2);
	deallocate(t3);
	deallocate(addition_carry);
	deallocate(mont);
}

void *execute(request_batch *reqs) {
	radix_type *dx, *dy, *dm, *dres, *drinv, *dmbar, *dr2modm, *dm1m2, *dq,
			*dqinv;
	int num_req = reqs->batch_size;
	int k = reqs->bit_len;
	int base=reqs->base;
	int basem1=reqs->basem1;
	

	dx = allocate(2*k*num_req);
	dy = allocate(2*k*num_req);
	dm = allocate(2*k*num_req);
	dres = allocate(2*k*num_req);
	dmbar = allocate(2*k*num_req);
	drinv = allocate(2*k*num_req);
	dr2modm = allocate(2*k*num_req);
	dm1m2 = allocate(2*k*num_req);
	dq = allocate(k*num_req);
	dqinv = allocate(k*num_req);

	memset(reqs->res, 0, 2 * num_req * k * sizeof(radix_type));

	copy_data_to_device(reqs->x, dx, 2 * k * num_req,0);
	copy_data_to_device(reqs->y, dy, 2 * k * num_req,0);
	copy_data_to_device(reqs->m, dm, 2 * k * num_req,0);
	copy_data_to_device(reqs->res, dres, 2 * k * num_req,0);
	copy_data_to_device(reqs->mbar, dmbar, 2 * k * num_req,0);
	copy_data_to_device(reqs->rinv, drinv, 2 * k * num_req,0);
	copy_data_to_device(reqs->r2modm, dr2modm, 2 * k * num_req,0);
	copy_data_to_device(reqs->m1m2, dm1m2, 2 * k * num_req,0);
	copy_data_to_device(reqs->q, dq, k * num_req,0);
	copy_data_to_device(reqs->qinv, dqinv, k * num_req,0);

	long c = reqs->x[0] + (basem1 + 1) * reqs->x[1];
	long e = reqs->y[0] + (basem1 + 1) * reqs->y[1];
	long mod = reqs->m[0] + (basem1 + 1) * reqs->m[1];

	calmerge_m1_m2(dx, dy, dm, k, drinv, dmbar, dr2modm, dq, dqinv, dm1m2,
			num_req,0, base, basem1);

	copy_data_to_host(reqs->m1m2, dm1m2, 2 * k * num_req,0);
	cudaError_t err = cudaStreamSynchronize(*streams[0]);
	for (int i = 0; i < num_req; i++) {
		if (k == 2) {
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
	
	deallocate(dx);
	deallocate(dy);
	deallocate(dm);
	deallocate(dres);
	deallocate(dmbar);
	deallocate(drinv);
	deallocate(dr2modm);
	deallocate(dm1m2);
	deallocate(dq);
	deallocate(dqinv);

	return reqs;
}

void setup(){
	printf("Initializing kernels..");
	setup_streams();
	THREADSPERBLOCK[PMUL] = 128;
	THREADSPERBLOCK[PADD] = 128;
	THREADSPERBLOCK[PSUB] = 512;
	THREADSPERBLOCK[RSHIFT] = 1024;
	THREADSPERBLOCK[PCOPY] = 256;
	THREADSPERBLOCK[PCOPYCOND] = 1024;
	THREADSPERBLOCK[CONVERTTOBASE] = 512;
}

void tear_down(){
	printf("Destroying streams..\n");
	destroy_streams();
}

void convert_arr(JNIEnv *env, jobjectArray a, radix_type *b, int num_req, int len){
	for(int i=0; i<num_req; i++){
		jintArray reqi= (jintArray)env->GetObjectArrayElement(a, i);
		jint *chars  = env->GetIntArrayElements(reqi, 0);
		for(int j=0; j<len; j++) {
	        	b[i*len+j]= chars[j];
		}
	}
}

void print_arr(radix_type *a, int n, char label[]){
	fprintf( stdout, label);
	fprintf( stdout, ": ");
	for(int i=0;i<n;i++){
		fprintf( stdout, "%d ", a[i]);
	}
	fprintf( stdout, "\n");
}


extern void decrypt_batch(JNIEnv *env, jobject jobj, jint jk, jobjectArray n, jobjectArray pd, jobjectArray qd, jobjectArray e, jobjectArray p, jobjectArray q
		, jobjectArray emsg, jobjectArray dmsg, jobjectArray qinv, jobjectArray rpinv, jobjectArray rqinv, jobjectArray mbarp, jobjectArray mbarq,
		jobjectArray r2p, jobjectArray r2q,jint jnum_req, jint jbase){
	fprintf( stdout, "Starting kernel processing\n" );
	radix_type *demsg, *dpqd, *dpq, *dqinv, *dres, *dm, *drinvpq, *dmbarpq, *dr2pq;
	int num_req = (int)jnum_req;
	int base = (int)jbase;
	int k = (int)jk; // k here is bitlength in binary representation (BASE:2)
	k = k/base; //length of array containing the number represented in BASE
	k/=2;

	demsg = (radix_type *)malloc(2*num_req*k*sizeof(radix_type));
	dpqd = (radix_type *)malloc(2*num_req*k*sizeof(radix_type));
	dpq = (radix_type *)malloc(2*num_req*k*sizeof(radix_type));/*combined p&q store toprevent divergence*/
	dqinv = (radix_type *)malloc(num_req*k*sizeof(radix_type));
	dres = (radix_type *)malloc(num_req*2*k*sizeof(radix_type));
	dm = (radix_type *)malloc(2*num_req*k*sizeof(radix_type));
	drinvpq = (radix_type *)malloc(2*num_req*k*sizeof(radix_type));
	dmbarpq = (radix_type *)malloc(2*num_req*k*sizeof(radix_type));
	dr2pq = (radix_type *)malloc(2*num_req*k*sizeof(radix_type));

	convert_arr(env, emsg, demsg, 2*num_req, k);
	convert_arr(env, pd, dpqd, num_req, k);
	convert_arr(env, qd, &dpqd[num_req*k], num_req, k);
	convert_arr(env, p, dpq, num_req, k);
	convert_arr(env, q, &dpq[num_req*k], num_req, k);
	convert_arr(env, qinv, dqinv, num_req, k);
	convert_arr(env, dmsg, dres, num_req, 2*k);

	//Precalculated values and parameters
	//R^-1 and m'
	convert_arr(env, rpinv, drinvpq, num_req, k);
	convert_arr(env, rqinv, &drinvpq[num_req*k], num_req, k);
	convert_arr(env, mbarp, dmbarpq, num_req, k);
	convert_arr(env, mbarq, &dmbarpq[num_req*k], num_req, k);

	//R^2 mod m
	convert_arr(env, r2p, dr2pq, num_req, k);
	convert_arr(env, r2q, &dr2pq[num_req*k], num_req, k);
print_arr(demsg,num_req*2*k ,"Msg");
	memset(dm,0,2*num_req*k*sizeof(radix_type));
	setup();
	request_batch *req = (request_batch *) malloc(sizeof(request_batch));
	req->x = (radix_type *) malloc(2 * k * num_req * sizeof(radix_type));
	req->y = (radix_type *) malloc(2 * k  * num_req * sizeof(radix_type));
	req->m = (radix_type *) malloc(2 * k  * num_req * sizeof(radix_type));
	req->res = (radix_type *) malloc(2 * k  * num_req * sizeof(radix_type));
	req->mbar = (radix_type *) malloc(2 * k  * num_req * sizeof(radix_type));
	req->rinv = (radix_type *) malloc(2 * k  * num_req * sizeof(radix_type));
	req->r2modm = (radix_type *) malloc(2 * k  * num_req * sizeof(radix_type));
	req->m1m2 = (radix_type *) malloc(2 * k  * num_req * sizeof(radix_type));
	req->q = (radix_type *) malloc(k  * num_req * sizeof(radix_type));
	req->qinv = (radix_type *) malloc(k  * num_req * sizeof(radix_type));
	req->bit_len = k;
	req->batch_size = num_req;
	req->base = base;
	req->basem1 = (int)pow(2,base)-1;
	if(DEBUG){
		fprintf(stdout, "POW=%d", req->basem1);
	}
for (int i = 0; i < num_req; i++) {
		
		for (int j = 0; j < k; j ++) {
			req->x[j] = demsg[i*k+j];
			req->y[j] = dpqd[i*k+j];
			req->m[j + 0] = dpq[i*k+j];
			req->rinv[j + 0] = drinvpq[i*k+j];
			req->mbar[j + 0] = dmbarpq[i*k+j];
			req->r2modm[j + 0] = dr2pq[i*k+j];
			req->q[j + 0] = dpq[i*k+k+j];
			req->qinv[j + 0] = dqinv[i*k+j];
			req->m1m2[j] = 0;
			req->res[j] = 0;
		}
		for (int j = 0; j < k; j ++) {
			req->x[j + k] = demsg[i*k+ k+j];
			req->y[j+ k] = dpqd[i*k+ k+j];
			req->m[j + 0+ k] = dpq[i*k+ k+j];
			req->rinv[j + 0+ k] = drinvpq[i*k+ k+j];
			req->mbar[j + 0+ k] = dmbarpq[i*k+ k+j];
			req->r2modm[j + 0+ k] = dr2pq[i*k+ k+j];
			req->m1m2[j+k] = 0;
			req->res[j+k] = 0;
		}

	}
	execute(req);
	if(DEBUG){
		print_arr(req->m1m2,num_req*2*k ,"Res");
	}
	for(int i=0;i<num_req;i++){
		jint *message = (jint *)calloc(sizeof(jint), 2*k);
		for (int j = 0; j<2*k; j++) {
				message[j] = (jint)(req->m1m2[i*2*k+j]);
		}
		jintArray  arrMsg = env->NewIntArray(2*k);
		env->SetIntArrayRegion(arrMsg, 0, 2*k, message);
		env->SetObjectArrayElement(dmsg,i,arrMsg);
		//dmsg[i] = arrMsg;
	}
	tear_down();
	free(demsg);
	free(dpqd);
	free(dpq);
	free(dqinv);
	free(dm);
	free(dres);
}