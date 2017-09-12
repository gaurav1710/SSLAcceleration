package com.sslproxy.jni;

public class CudaKernels {
	static { System.loadLibrary("CudaKernels"); }
    	public native int[][] decrypt(int k, int n[][], int pd[][],int qd[][], int e[][], int p[][], int q[][], int emsg[][],
			int dmsg[][], int qinv[][], int rpinv[][], int rqinv[][], int mp[][], int mq[][], int r2p[][], int r2q[][], int reqs); 	
}
