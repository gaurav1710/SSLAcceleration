#include <jni.h>
#include "com_sslproxy_jni_CudaKernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
extern void execute_test();
//extern void decrypt(int k, char n[], char pd[], char qd[], char e[], char p[], char q[], char emsg[], char dmsg[], char qinv[]);
extern void decrypt_batch(JNIEnv *env, jobject jobj, jint jk, jobjectArray n, jobjectArray pd, jobjectArray qd, jobjectArray e, jobjectArray p, jobjectArray q, jobjectArray emsg, jobjectArray dmsg, jobjectArray qinv, jobjectArray rpinv, jobjectArray rqinv, jobjectArray mbarp, jobjectArray mbarq, jobjectArray r2p, jobjectArray r2q, jint jnum_req, jint jbase);
struct timeval start,stop;
#define RECORD_TIME_START gettimeofday(&start, NULL);

#define RECORD_TIME_END(func) gettimeofday(&stop, NULL);\
					printf("Time taken by %s = %lu\n",func, stop.tv_usec - start.tv_usec);

void convert(jchar* a, char b[], int len){
	for(int i=0;i<len;i++){
		b[i] = (char)(a[i]);
		b[i] = b[i]-'0';	
	}
	return;
}


JNIEXPORT jobjectArray JNICALL Java_com_sslproxy_jni_CudaKernels_decrypt
  (JNIEnv *env, jobject jobj, jint jk, jobjectArray n, jobjectArray pd, jobjectArray qd,
		  jobjectArray e, jobjectArray p, jobjectArray q, jobjectArray emsg, jobjectArray dmsg,
		  jobjectArray qinv, jobjectArray rpinv, jobjectArray rqinv, jobjectArray mbarp, jobjectArray mbarq,
		  jobjectArray r2p, jobjectArray r2q,jint num_req, jint jbase)
{

	decrypt_batch(env, jobj, jk, n, pd, qd, e, p, q, emsg, dmsg, qinv, rpinv, rqinv, mbarp, mbarq, r2p, r2q, num_req, jbase);

	return dmsg;
}
