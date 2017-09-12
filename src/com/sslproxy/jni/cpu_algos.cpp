#include<stdio.h>
#include <omp.h>
#include <stdlib.h>
//#include<python2.7/Python.h>
#include "config.h"
#include "cpu_algos.h"

const int MEMK = 0;
const int MEM2K = 1;
const int MEMREQ = 2;
const int MEMBASE = 3;

void sprint_array(radix_type a[], int l) {
	for (int i = l - 1; i >= 0; i--) {
		printf("%d ", a[i]);
	}
	printf("\n");

}

void smul(radix_type *m, radix_type *n, int k, radix_type *res, int stridem,
		int striden, int num_req, int num_threads) {
	//return;
	omp_set_num_threads(num_threads);
	int num_of_requests_per_thread = num_req / num_threads;
#pragma omp parallel
	{
		int id = omp_get_thread_num();

		int offset = id * num_of_requests_per_thread;
		for (int i = 0; i < num_of_requests_per_thread; i++) {
#pragma unroll
			for (int mi = 0; mi < k; mi++) {
#pragma unroll
				for (int ni = 0; ni < k; ni++) {
					long_radix_type prd = m[stridem * (i + offset) + mi]
							* n[striden * (i + offset) + ni]
							+ res[mi + ni + (i + offset) * 2 * k];
					res[mi + ni + (i + offset) * 2 * k] = prd & BASEMINUS1;
					if (mi + ni + 1 < 2 * k) {
						res[mi + ni + (i + offset) * (2 * k) + 1] +=
								prd >> BASE;
					}
				}
			}
		}
	}
}

void sadd(radix_type *a, radix_type *b, int k, int num_req, int stridea,
		int strideb, radix_type *residue_carry, int num_threads) {
	omp_set_num_threads(num_threads);
	int num_of_requests_per_thread = num_req / num_threads;
#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int offset = id * num_of_requests_per_thread;
		radix_type carry = 0;
		for (int i = 0; i < num_of_requests_per_thread; i++) {
#pragma unroll
			for (int j = 0; j < k; j++) {
				long_radix_type sum = a[stridea * (i + offset) + j]
						+ b[strideb * (i + offset) + j] + carry;
				a[stridea * (i + offset) + j] = sum & BASEMINUS1;
				carry = sum >> BASE;
			}
			residue_carry[(i + offset)] = carry;
		}
	}
}

void ssub(radix_type *a, radix_type *b, int k, radix_type *res,
		radix_type *add_carry, int stridea, int strideb, int compare,
		int num_req, int num_threads) {
	omp_set_num_threads(num_threads);
	int num_of_requests_per_thread = num_req / num_threads;
#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int offset = id * num_of_requests_per_thread;
		for (int i = 0; i < num_of_requests_per_thread; i++) {
			int pos = offset + i;
			if ((compare == 1
					&& (sd_compare(&a[stridea * pos], &b[strideb * pos], k) >= 0))
					|| !compare || (add_carry[pos] == 1)) {
				int carry = 0;
				int dig = 0;
#pragma unroll
				for (int j = 0; j < k; j++) {
					dig  = a[stridea * pos + j] - b[strideb * pos + j] - carry;
					if(dig<0){
						carry = 1;
					}else{
						carry = 0;
					}
					res[pos*k + j] = ((carry) * (BASEMINUS1 + 1) + dig);
				}
			}
		}
	}
}

radix_type * sadd_karatsuba(radix_type *m, radix_type *n, int size){
	radix_type *res = (radix_type *)malloc(size*sizeof(radix_type));
	radix_type carry = 0;
	for (int j = 0; j < size; j++) {
		long_radix_type sum = m[j]+ n[j] + carry;
		res[j] = sum & BASEMINUS1;
		carry = sum >> BASE;
	}
	return res;
}

radix_type * ssub_karatsuba(radix_type *m, radix_type *n, int size){
	radix_type *res = (radix_type *)malloc(size*sizeof(radix_type));
	radix_type dig = 0;
	radix_type carry = 0;

	for (int j = 0; j < size; j++) {
		dig  = m[j] - n[j] - carry;
		if(dig<0){
			carry = 1;
		}else{
			carry = 0;
		}
		res[j] = ((carry) * (BASEMINUS1 + 1) + dig);
	}
	return res;
}


void karatsuba(radix_type *xx, radix_type *yy, radix_type *rr, int tLen)
{
	if(tLen == 1 ){
		rr[0] = xx[0]*yy[0];
		return;
	}
	radix_type *a = &xx[tLen/2];
	radix_type *b = &xx[0];
	radix_type *c = &yy[tLen/2];
	radix_type *d = &yy[0];

    //since only 2d space is required for result
    //hence we 'll use remaining space
	radix_type *wx = &rr[tLen*5]; //sum of xx's halves
	radix_type *wy = &rr[tLen*5 + tLen/2]; //sum of yy's halves

	radix_type *V = &rr[tLen*0];  //location of b*d
	radix_type *U = &rr[tLen*1];  //location of a*c
	radix_type *W = &rr[tLen*2];  //location of (a+b)*(c+d)

    int i;
    //compute wx and wy
    for(i=0; i<tLen/2; i++){
        wx[i] = a[i] + b[i];
        wy[i] = c[i] + d[i];
    }

    //divide
    karatsuba(b, d, V, tLen/2);
    karatsuba(a, c, U, tLen/2);
    karatsuba(wx, wy, W, tLen/2);

    //conquer and combine
    for(i=0; i<tLen; i++)  W[i]=W[i]-U[i]-V[i];
    for(i=0; i<tLen; i++)  rr[i+tLen/2] += W[i];

}

radix_type * smul_karatsuba(radix_type *m, radix_type *n, int size){
	if(size == 1){

//		radix_type *res = (radix_type *)malloc(sizeof(radix_type));
//		res[0] = m[0]*n[0];
		return m;//res;
	}
	//z0
	radix_type *z0 = smul_karatsuba(m,n,size/2);
	//z1
	radix_type *z1 = smul_karatsuba(&m[size/2],&n[size/2],size/2);
	//z2
	radix_type *z2 = smul_karatsuba(sadd_karatsuba(m,&m[size/2],size/2),sadd_karatsuba(n,&n[size/2],size/2),size/2);

	//radix_type *fres = (radix_type *)malloc(2*size*sizeof(radix_type));
	//radix_type *sub1 = ssub_karatsuba(ssub_karatsuba(z1,z2,size),z0,size);
	return NULL;//sadd_karatsuba(z2, sadd_karatsuba(sub1,z0,size),size);
}


void smul_s(radix_type *m, radix_type *n, int k, radix_type *res, int stridem,
		int striden, int num_req, int num_threads) {
	//return;
	int num_of_requests_per_thread = num_req / num_threads;
	{
		int id = 0;

		int offset = id * num_of_requests_per_thread;
		for (int i = 0; i < num_of_requests_per_thread; i++) {
//#pragma unroll
			//smul_karatsuba(&m[stridem * (i + offset)], &n[striden * (i + offset)], k);
			radix_type *rr = (radix_type *)malloc(100*2*k*sizeof(radix_type));
			karatsuba(&m[stridem * (i + offset)], &n[striden * (i + offset)], rr, k);
//			for (int mi = 0; mi < k; mi++) {
//#pragma unroll
//				for (int ni = 0; ni < k; ni++) {
//					int ind = mi + ni + (i + offset) * 2 * k;
//					long_radix_type prd = m[stridem * (i + offset) + mi]
//							* n[striden * (i + offset) + ni]
//							+ res[ind];
//					res[ind] = prd & BASEMINUS1;
//					if (mi + ni + 1 < 2 * k) {
//						res[ind + 1] += prd >> BASE;
//					}
//				}
//			}
		}
	}
}

void sadd_s(radix_type *a, radix_type *b, int k, int num_req, int stridea,
		int strideb, radix_type *residue_carry, int num_threads) {
	int num_of_requests_per_thread = num_req / num_threads;
	{
		int id = 0;
		int offset = id * num_of_requests_per_thread;
		radix_type carry = 0;
		for (int i = 0; i < num_of_requests_per_thread; i++) {
//#pragma unroll
			for (int j = 0; j < k; j++) {
				long_radix_type sum = a[stridea * (i + offset) + j]
						+ b[strideb * (i + offset) + j] + carry;
				a[stridea * (i + offset) + j] = sum & BASEMINUS1;
				carry = sum >> BASE;
			}
			residue_carry[(i + offset)] = carry;
		}
	}
}

void ssub_s(radix_type *a, radix_type *b, int k, radix_type *res,
		radix_type *add_carry, int stridea, int strideb, int compare,
		int num_req, int num_threads) {
	int num_of_requests_per_thread = num_req / num_threads;
	{
		int id = 0;
		int offset = id * num_of_requests_per_thread;
		for (int i = 0; i < num_of_requests_per_thread; i++) {
			int pos = offset + i;
			if ((compare == 1
					&& (sd_compare(&a[stridea * pos], &b[strideb * pos], k) >= 0))
					|| !compare || (add_carry[pos] == 1)) {
				int carry = 0;
				int dig = 0;
//#pragma unroll
				for (int j = 0; j < k; j++) {
					dig  = a[stridea * pos + j] - b[strideb * pos + j] - carry;
					if(dig<0){
						carry = 1;
					}else{
						carry = 0;
					}
					res[pos*k + j] = ((carry) * (BASEMINUS1 + 1) + dig);
				}
			}
		}
	}
}

int sd_compare(radix_type *a, radix_type *b, int size) {
	for (int i = size - 1; i >= 0; i--) {
		if (a[i] != b[i])
			return a[i] - b[i];
	}
	return 0;
}

void sright_shift(radix_type *m, int stride, int k, radix_type *res,
		int num_req) {
	for(int i=0;i<num_req;i++){
		int pos = i;
		for(int j=0;j<k;j++){
			res[pos * stride + j] = m[pos * stride + j + k];
		}
	}
}

void scopy(radix_type *a, radix_type *b, int stridea, int strideb, int k,
		int num_req) {
	for(int i=0;i<num_req;i++){
		int pos = i;
		for(int j=0;j<k;j++){
			a[pos * stridea + j] = b[pos * strideb + j];
		}
	}
}

void scopy_wcondition(radix_type *a, radix_type *b, short_radix_type *base2,
		int k, int stridea, int strideb, int num_req) {
	for(int i=0;i<num_req;i++){
			int pos = i;
		if ((base2[pos] == 1)) {
			for(int j=0;j<k;j++){
				a[pos * stridea + j] = b[pos * strideb + j];
			}
		}
	}
}

void sconvert_to_base(radix_type *x, short_radix_type *y, int k, int base,
		int num_req) {
	for(int tid=0;tid<num_req;tid++){
		int ind = 0;
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

void s_extract_bits(short_radix_type *base2y, short_radix_type *base2_bits,
		int ind, int k, int num_req) {
	for(int i=0;i<num_req;i++){
		base2_bits[i] = base2y[i * BASE * k + ind];
	}
}

radix_type *sallocate(int type, int k, int num_req, int tid) {
	radix_type *ptr = NULL;
	switch(type){
	case MEMK:
		ptr = (radix_type *)malloc(sizeof(radix_type)*num_req*k);
		break;
	case MEM2K:
		ptr = (radix_type *)malloc(sizeof(radix_type)*num_req*2*k);
		break;
	case MEMREQ:
		ptr = (radix_type *)malloc(sizeof(radix_type)*num_req);
		break;
	case MEMBASE:
		ptr = (radix_type *)malloc(sizeof(radix_type)*num_req*k*BASE);
		break;
	}
}

void memset_s(radix_type *a, radix_type val, int size){
	for(int i=0;i<size;i++){
		a[i] = val;
	}
}

inline void smontgomery(radix_type *a, radix_type *b, int k, radix_type *m,
		radix_type *rinv, radix_type *mbar, radix_type *res, int num_req,
		int tid) {

	radix_type *T, *M, *U, *carry, *addition_carry;

	T = sallocate(MEM2K, k, num_req, tid);
	M = sallocate(MEM2K, k, num_req, tid);
	U = sallocate(MEM2K, k, num_req, tid);
	carry = sallocate(MEM2K, k, num_req, tid);
	addition_carry = sallocate(MEMREQ, k, num_req, tid);

	memset_s(T,0,2 * k * num_req);
	//T=a*b
	smul_s(a,b,k,T,k,k,num_req,1);

	//pmul<<<num_req,k>>>(a,b,k,T,carry,k,k);
	if (DEBUG) {
		printf("montgomery-MulT:");
		sprint_array(T, 2 * k);
	}

	memset_s(M,0,2 * k * num_req);

	//M=T*m'
	smul_s(T, mbar, k, M, 2 * k, k, num_req, 1);
	if (DEBUG) {
		printf("montgomery-MulM:");
		sprint_array(M, 2 * k);
	}

	memset_s(U,0,2 * k * num_req);

	//U=M*m
	smul_s(M, m, k, U, 2 * k, k, num_req, 1);
	if (DEBUG) {
		printf("montgomery-MulU:");
		sprint_array(U, 2 * k);
	}
	memset_s(addition_carry,0,num_req);
	//U=T+M*m
	sadd_s(U, T, 2 * k, num_req, 2 * k, 2 * k, addition_carry, 1);

	if (DEBUG) {
		printf("montgomery-AddU2:");
		sprint_array(U, 2 * k);
	}
	memset_s(carry,0,2 * k * num_req);
	sright_shift(U, 2 * k, k, carry, num_req);	//carry is result here..
	if (DEBUG) {
		printf("montgomery-RShiftU:");
		sprint_array(carry, 2 * k);
	}

	scopy(res, carry, k, 2 * k, k, num_req);

	ssub_s(carry, m, k, res, addition_carry, 2 * k, k, 1, num_req,1);
	if (DEBUG) {
		printf("montgomery-SubRes:");
		sprint_array(res, k);
	}
	free(T);
	free(M);
	free(U);
	free(carry);
	free(addition_carry);
}

inline void shmod_exp_sqmul(radix_type *x, radix_type *y, int k, radix_type *m,
		radix_type *res, radix_type *rinv, radix_type *mbar, radix_type *r2modm,
		int num_req, int tid) {
	radix_type *temp = sallocate(MEMK, k, num_req, tid);
	short_radix_type *base2y = sallocate(MEMBASE, k, num_req, tid);
	radix_type *ones = sallocate(MEMK, k, num_req, tid);
	short_radix_type *base2_bits = sallocate(MEMREQ, k, num_req, tid);

	memset_s(temp, 0, num_req * k );
	memset_s(ones, 0, num_req * k );

	for (int i = 0; i < num_req; i++) {
		temp[i * k] = 1;	//initialize with 1
		ones[i * k] = 1;
	}

	memset_s(res,0,k * num_req);

	sconvert_to_base(y, base2y, k, 2, num_req);

	smontgomery(temp, r2modm, k, m, rinv, mbar, res, num_req, tid);

	if (DEBUG) {
		printf("R2*1 mod m:");
		sprint_array(res, k);
	}
	scopy(temp, res, k, k, k, num_req);

	memset_s(res,0,k * num_req);

	smontgomery(x, r2modm, k, m, rinv, mbar, res, num_req, tid);

	if (DEBUG) {
		printf("R2*x mod m:");
		sprint_array(res, k);
	}
	scopy(x, res, k, k, k, num_req);

	for (int i = BASE * k - 1; i >= 0; i--) {
		memset_s(res,0,k * num_req);
		smontgomery(temp, temp, k, m, rinv, mbar, res, num_req, tid);

		scopy(temp, res, k, k, k, num_req);
		memset_s(res,0,k * num_req);
		smontgomery(temp, x, k, m, rinv, mbar, res, num_req, tid);
		s_extract_bits(base2y, base2_bits, i, k, num_req);
		scopy_wcondition(temp, res, base2_bits, k, k, k, num_req);
	}

	memset_s(res,0,k * num_req);
	smontgomery(temp, ones, k, m, rinv, mbar, res, num_req, tid);

	free(temp);
	free(base2y);
	free(ones);
	free(base2_bits);
}

inline void smontgomery_with_conv(radix_type *a, radix_type *b, radix_type *m,
		radix_type *rinv, radix_type *mbar, radix_type *r2modm, radix_type *res,
		int k, int num_req, int tid) {
	radix_type *ar, *br, *abr, *ones;
	ar = sallocate(MEMK, k, num_req, tid);
	br = sallocate(MEMK, k, num_req, tid);
	abr = sallocate(MEMK, k, num_req, tid);
	ones = sallocate(MEMK, k, num_req, tid);

	memset_s(ones, 0, num_req * k );

	memset_s(abr, 0, num_req * k );
	memset_s(ar, 0, num_req * k );
	memset_s(br, 0, num_req * k );

	for (int i = 0; i < num_req; i++) {
		ones[i * k] = 1;
	}

	//copy_data_to_device(hones,ones,k*num_req,tid);
	//a*r mod m
	smontgomery(a, r2modm, k, m, rinv, mbar, ar, num_req, tid);
	//b*r mod m
	smontgomery(b, r2modm, k, m, rinv, mbar, br, num_req, tid);
	//a*b*r mod m
	smontgomery(ar, br, k, m, rinv, mbar, abr, num_req, tid);
	//a*b mod m
	smontgomery(abr, ones, k, m, rinv, mbar, res, num_req, tid);
	free(ar);
	free(br);
	free(abr);
	free(ones);
}

void scalmerge_m1_m2(radix_type *c, radix_type *e, radix_type *m, int k,
		radix_type *rinv, radix_type *mbar, radix_type *r2modm, radix_type *q,
		radix_type *qinv, radix_type *res, int num_req, int tid) {
	radix_type *t1, *t2, *t3, *mont;
	radix_type *M, *addition_carry;

	//Calculate M1 and M2
	M = sallocate(MEM2K, k, num_req, tid);

	memset_s(M, 0, k * 2 * num_req );

	shmod_exp_sqmul(c, e, k, m, M, rinv, mbar, r2modm, 2 * num_req, tid);

	t1 = sallocate(MEMK, k, num_req, tid);
	t2 = sallocate(MEM2K, k, num_req, tid);
	t3 = sallocate(MEM2K, k, num_req, tid);
	addition_carry = sallocate(MEMREQ, k, num_req, tid);
	mont = sallocate(MEMK, k, num_req, tid);

	memset_s(t1, 0, k *  num_req );
	memset_s(addition_carry, 0, num_req );

	//M1-M2
	ssub_s(M, &M[num_req * k], k, t1, addition_carry,  k, k, 0,
			num_req, 1);

	memset_s(t2, 0, 2 * k * num_req );
	memset_s(t3, 0, 2 * k * num_req );
	memset_s(mont, 0, k * num_req );

	//(M1-M2)*(qinv mod p) mod p
	smontgomery_with_conv(t1, qinv, m, rinv, mbar, r2modm, mont, k, num_req,
			tid);

	memset_s(t2, 0, 2 * k * num_req );
	memset_s(t3, 0, 2 * k * num_req );

	//(M1-M2)*(qinv mod p)*q
	smul_s(mont, q, k, t2, k, k, num_req, 1);


	scopy(res, t2, 2 * k, 2 * k, 2 * k, num_req);
	memset_s(addition_carry, 0, num_req );
	//M2 + (M1-M2)*(qinv mod p)*q
	sadd_s(res, &M[num_req * k], k,  num_req, 2 * k, k, addition_carry ,1);


	free(t1);
	free(t2);
	free(t3);
	free(mont);
	free(M);
	free(addition_carry);

}
void load_modex(){
//	Py_Initialize();
//	PySys_SetPath(".");
}
void modex(int NUM){

	int bl = BITLENGTH*BASE*2;
	char *c = new char[NUM*bl+1];
	char *d = new char[NUM*bl+1];
	char *n = new char[NUM*bl+1];
	char *m = new char[NUM*bl+1];

	printf("Executing pyscript for modex(NUM=%d)\n",NUM);
	for(int i=0;i<NUM;i++){
		for(int j=0;j<bl-1;j++){
			if(j%2 == 0)
				c[i*bl + j] = '1';
			else
				c[i*bl + j] = '0';
			if(j%3 == 0)
				d[i*bl + j] = '1';
			else
				d[i*bl + j] = '0';
			if(j%5 == 0)
				n[i*bl + j] = '1';
			else
				n[i*bl + j] = '0';
		}
		c[i*bl + bl-1] = '1';
		d[i*bl + bl-1] = '1';
		n[i*bl + bl-1] = '1';
	}
	c[NUM*bl-1] = '\0';
	d[NUM*bl-1] = '\0';
	n[NUM*bl-1] = '\0';

	char ns[5];
	char bls[5];

	sprintf(ns,"%d",NUM);
	sprintf(bls,"%d",bl);

	FILE *pipe = popen("pypy modex.py","w");
	fputs(c, pipe);
	fputc('\n', pipe);
	fputs(d, pipe);
	fputc('\n', pipe);
	fputs(n, pipe);
	fputc('\n', pipe);
	fputs(ns, pipe);
	fputc('\n', pipe);
	fputs(bls, pipe);
	fputc('\n', pipe);

//	fgets(m,NUM*bl+2,pipe);
//	printf("M=%s\n",m);
	fclose(pipe);
//	char *cmd = new char[3*NUM*bl+128];
//	cmd[0] = '\0';
//	strcat(cmd,"python");
//	strcat(cmd," ");
//	strcat(cmd,"modex.py");
//	strcat(cmd," ");
//	strcat(cmd,c);
//	strcat(cmd," ");
//	strcat(cmd,d);
//	strcat(cmd," ");
//	strcat(cmd,n);
//	strcat(cmd," ");
//	strcat(cmd,ns);
//	strcat(cmd," ");
//	strcat(cmd,bls);
////	printf("COMMAND=%s\n",cmd);
//	system(cmd);
/*
	PyObject *module, *func, *prm, *ret;
	PyEval_InitThreads();
	PyEval_SaveThread();
	module = PyImport_ImportModule("modex");
	if (module != 0){
		func = PyObject_GetAttrString(module, "modex");

		prm = Py_BuildValue("(sssii)", c, d, n,NUM, bl);
		ret = PyObject_CallObject(func, prm);
		//Result m = c^d mod n
		m = PyString_AsString(ret);

		Py_DECREF(module);
		Py_DECREF(func);
		Py_DECREF(prm);
		Py_DECREF(ret);
		Py_Finalize();
	}
	*/
}

