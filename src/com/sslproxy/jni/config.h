#define K 6
//dynamic parallelism enabled
#define DYPARALLEL  0
//RSA key bit-length(length/BASE of p,q)
#define BITLENGTH 16
//Number of kernels
#define NUMKERNELS 7
//Base 2^k
#define BASE 32
//2^BASE-1
#define BASEMINUS1 4294967295
//Debug logs
#define DEBUG 0
//Number of requests per stream/CPU thread
#define MAXREQUESTSPERCPUTHREAD 512
//CPU pthreads = Maximum number of CPU cores
#define MAXNUMCPUTHREADS 1
//Maximum memory segments to be formed during memory pool stack allocation
#define MAXMEMSEGMENTS 128
//Maximum requests per internal stream(per cpu thread)
#define MAXREQUESTSPERSTREAM 128
//Enable global load/stores - this is used to find power consumed by global load/stores
#define ENABLEGLOBALLDST 1
//Enabled determination of optimal grid dimensions using some tests
#define ENABLEPARAMDETERM 0

#define COPYSTREAMS 1

#define NUMSAMPLES 16

#define BATCHTIMEOUT 1000000

//#define GPUCPUMULRATIO 1.0
//#define GPUCPUADDRATIO 1.0

#define EXTRATHREADS 1

//CPU pthreads = Maximum number of CPU cores
#define MAXNUMCPUSERIALTHREADS 0

#define MAXREQUESTSPERPY 25

#define CUDAMEMDEVICE cudaMalloc
#define CUDAMEMHOST cudaMallocHost

#define PMUL 0
#define PADD 1
#define PSUB 2
#define RSHIFT 3
#define PCOPY 4
#define PCOPYCOND 5
#define CONVERTTOBASE 6

#define SERIAL 0

//typedef unsigned int short_radix_type;
//typedef unsigned int radix_type;
//typedef unsigned int long_radix_type;

typedef long short_radix_type;
typedef long radix_type;
typedef long long_radix_type;

struct request_batch_struct {
	radix_type *x, *y, *m, *res, *rinv, *mbar, *r2modm, *m1m2, *q, *qinv;
	int bit_len;
	int batch_size;
	int base;
};

typedef struct request_batch_struct request_batch;
