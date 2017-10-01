#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "config.h"
#include "gpu_algos.h"


__device__ void acc(radix_type *a, radix_type *b, int k, int base, int basem1) {
	radix_type carry = 0;
	for (int i = 0; i < k; i++) {
		radix_type sum = a[i] + b[i] + carry;
		a[i] = sum & basem1;
		carry = sum >> base;
	}
}

__device__ void dprint_array(radix_type a[], int l) {
	for (int i = l - 1; i >= 0; i--) {
		printf("%d ", a[i]);
	}
	printf("\n");

}

/*
 * MP parallel multiplication implimentation
 * Calculates m*n
 */
__global__ void pmul(radix_type *mg, radix_type *n, int k, radix_type *res,
		int stridem, int striden, int num_threads, int base, int basem1) {
	extern __shared__ radix_type shared_mem[]; //size:(2*k+2*k+k)*(num of req per block)
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	if (tx < num_threads) {
		int reqs = num_threads / k;
		int reqno = tx / k;
		radix_type *carry_buf = &shared_mem[0 + reqno * 2 * k]; //size:k*2*(num of req per block)
		radix_type *inter_buf = &shared_mem[(reqs) * 2 * k + reqno * 2 * k]; //size:k*2*(num of req per block)
		radix_type *m = &shared_mem[(reqs) * 4 * k + reqno * k]; //size:k*(num of req per block)

		int i;
		int x = tx % k;
		if (ENABLEGLOBALLDST) {
			m[x] = mg[bx * stridem * reqs + stridem * reqno + x];
		}
		carry_buf[x] = 0;
		carry_buf[x + k] = 0;
		inter_buf[x] = 0;
		inter_buf[x + k] = 0;
		//digit of one operand that will be taken care of by this thread..
		long_radix_type ndig = 1;
		
		long_radix_type product = 1;
		//int mi = reqno*k;
		for (i = 0; i < k; i++) {
			product = m[i] * ndig;
			//low word
			radix_type lword = product & basem1;
			//high word
			radix_type hword = product >> base;
			//store and add to partial results with carry handling
			int ind = x + i;
			//lword
			radix_type cil = inter_buf[ind] + lword;
			radix_type carry = (cil) >> base;
			inter_buf[ind] = (cil) & basem1;

			__syncthreads();
			//hword
			if (ind + 1 < 2 * k) {
				carry_buf[(ind + 1)] += carry;
				radix_type hcil = inter_buf[ind + 1] + hword;
				carry = ((hcil) >> base);
				inter_buf[ind + 1] = (hcil) & basem1;
			}
			if (ind + 2 < 2 * k) {
				carry_buf[(ind + 2)] += carry;
			}

		}
		if (x == 0) {
			acc(&inter_buf[0], &carry_buf[0], 2 * k, base, basem1);
		}

		//	res[2*k*bx+2*x] = inter_buf[2*x];
		//	res[2*k*bx+2*x+1] = inter_buf[2*x+1];
		int offset = num_threads / k;
		if (ENABLEGLOBALLDST) {
			int offs = 2 * k * offset * bx + reqno * 2 * k + x;
			res[offs] = inter_buf[x];
			res[offs + k] = inter_buf[x + k];
		}
	}
}

__global__ void padd(radix_type *a, radix_type *b, int k, int num_req,
		int stridea, int strideb, radix_type *residue_carry, int num_threads, int base, int basem1) {
	extern __shared__ radix_type shared_mem[];	//size:k+k+1

	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int reqs = num_threads / k;
	int reqno = tx / k;

	radix_type *res = &shared_mem[0 + reqno * k];	//size:k
	radix_type *carry_buf = &shared_mem[reqs * k + reqno * (k + 1)];	//size:k

	int x = tx % k;
	carry_buf[x] = 0;
	res[x] = 0;
	if (x == 0) {
		carry_buf[k] = 0;
	}

	long_radix_type sum = 0;
	if (ENABLEGLOBALLDST) {
		sum = a[bx * reqs * stridea + reqno * stridea + x]
				+ b[bx * reqs * strideb + reqno * strideb + x];
	}

	for (int i = 0; i < k; i++) {
		res[x] = sum & basem1;
		carry_buf[x + 1] = carry_buf[x + 1] + (sum >> base);
		sum = res[x] + carry_buf[x];
		carry_buf[x] = 0;
	}

	if (x == k - 1) {
		if (ENABLEGLOBALLDST) {
			residue_carry[bx * reqs + reqno] = carry_buf[k];
		}
	}
	if (ENABLEGLOBALLDST) {
		a[bx * reqs * stridea + reqno * stridea + x] = res[x];
	}
}
__device__ int d_compare(radix_type *a, radix_type *b, int size) {
	for (int i = size - 1; i >= 0; i--) {
		if (a[i] != b[i])
			return a[i] - b[i];
	}
	return 0;
}

__global__ void psub(radix_type *a, radix_type *b, int k, radix_type *resg,
		radix_type *add_carry, int stridea, int strideb, int compare,
		int num_threads, int base, int basem1) {
	extern __shared__ radix_type shared_mem[];	//size:k+k

	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int reqs = num_threads / k;
	int reqno = tx / k;
	radix_type *res = &shared_mem[0 + reqno * k];	//size:k
	radix_type *carry_buf = &shared_mem[(reqs + reqno) * k];	//size:k+1
	int x = tx % k;
	//bx*reqs*stridea+reqno*stridea
	int pos = bx * reqs + reqno;
	if ((compare == 1
			&& (d_compare(&a[stridea * pos], &b[strideb * pos], k) >= 0))
			|| !compare || (add_carry[pos] == 1)) {
		carry_buf[x] = 0;
		res[x] = 0;
		radix_type carry = 0;
		int dig = 0;
		if (ENABLEGLOBALLDST) {
			dig = a[stridea * pos + x] - b[strideb * pos + x];
		}

		for (int i = 0; i < k; i++) {
			carry = dig < 0;
			res[x] = ((carry) * (basem1 + 1) + dig);
			if ((x + 1) != k)
				carry_buf[x + 1] = carry;
			dig = res[x] - carry_buf[x];
		}
		resg[k * pos + x] = res[x];
	}
}

__global__ void right_shift(radix_type *m, int stride, int k, radix_type *res,
		int num_threads) {
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int reqs = num_threads / k;
	int reqno = tx / k;
	int x = tx % k;
	int pos = bx * reqs + reqno;
	res[pos * stride + x] = m[pos * stride + x + k];
}

__global__ void pcopy(radix_type *a, radix_type *b, int stridea, int strideb,
		int k, int num_threads) {

	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int reqs = num_threads / k;
	int reqno = tx / k;
	int x = tx % k;
	int pos = bx * reqs + reqno;
	a[pos * stridea + x] = b[pos * strideb + x];
}

__global__ void pcopy_wcondition(radix_type *a, radix_type *b,
		short_radix_type *base2, int k, int stridea, int strideb,
		int num_threads) {

	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int reqs = num_threads / k;
	int reqno = tx / k;
	int pos = bx * reqs + reqno;
	if ((base2[pos] == 1)) {
		int x = tx % k;
		a[pos * stridea + x] = b[pos * strideb + x];
	}
}

__global__ void convert_to_base(radix_type *x, short_radix_type *y, int k,
		int base, int num_req, int num_threads_per_block, int BASE) {
	int tid = blockIdx.x * num_threads_per_block + threadIdx.x;
	int ind = 0;
	if (tid < num_req) {
		for (int j = 0; j < k; j++) {
			radix_type xi = x[tid * k + j];
#pragma unroll
			for (int i = 0; i < BASE; i++) {
				y[tid * k * BASE + ind] = xi % base;
				ind++;
				xi = xi / base;
			}
		}
	}
}
__global__ void extract_bits(short_radix_type *base2y,
		short_radix_type *base2_bits, int ind, int k, int base) {
	int x = threadIdx.x;
	base2_bits[x] = base2y[x * base * k + ind];
}
