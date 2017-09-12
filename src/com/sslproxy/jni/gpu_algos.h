__global__ void pmul(radix_type *, radix_type *, int, radix_type *, int, int,
		int);
__global__ void padd(radix_type *, radix_type *, int, int, int, int,
		radix_type *, int);

__device__ int d_compare(radix_type *, radix_type *, int);

__global__ void psub(radix_type *, radix_type *, int, radix_type *,
		radix_type *, int, int, int, int);
__global__ void right_shift(radix_type *, int, int, radix_type *, int);

__global__ void pcopy(radix_type *, radix_type *, int, int, int, int);
__global__ void pcopy_wcondition(radix_type *, radix_type *, short_radix_type *,
		int, int, int, int);
__global__ void convert_to_base(radix_type *, short_radix_type *, int, int, int,
		int);
__global__ void extract_bits(short_radix_type *, short_radix_type *, int, int);
