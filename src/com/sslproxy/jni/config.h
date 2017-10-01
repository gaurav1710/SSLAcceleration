//Debug logs
#define DEBUG 0
#define CUDAMEMDEVICE cudaMalloc
#define CUDAMEMHOST cudaMallocHost
#define ENABLEGLOBALLDST 1
#define PMUL 0
#define PADD 1
#define PSUB 2
#define RSHIFT 3
#define PCOPY 4
#define PCOPYCOND 5
#define CONVERTTOBASE 6

typedef long short_radix_type;
typedef long radix_type;
typedef long long_radix_type;

struct request_batch_struct {
	radix_type *x, *y, *m, *res, *rinv, *mbar, *r2modm, *m1m2, *q, *qinv;
	int bit_len;
	int batch_size;
	int base;
	int basem1;
};

typedef struct request_batch_struct request_batch;