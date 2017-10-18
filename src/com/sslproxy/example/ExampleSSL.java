package com.sslproxy.example;

import java.math.BigInteger;
import java.util.Random;

import com.sslproxy.jni.CudaKernels;

/**
 * Example/Test the SSL offloading code
 * 
 * @author gaurav(mcs132556)
 */
public class ExampleSSL {
	public static int BITLENGTH = 32; //=k-bit RSA encryption
	public static int BASE = 16; //Base 2^BASE representation  
	public static int BATCH_SIZE = 1;//Number of decryption requests

	public static void main(String[] args) {
		System.out.println("Setup Details:");
		System.out.println("BITLENGTH=" + BITLENGTH);
		System.out.println("BASE=2^" + BASE);
		SSLWorker.initVal();
		new SSLWorker(0).run(); //synchronous, no threading for now..
	}
}
class SSLWorker implements Runnable {
	public static BigInteger bn[];
	public static BigInteger bd[];
	public static BigInteger bp[];
	private static int n[][];
	private static int pd[][];
	private static int qd[][];
	private static int e[][];
	private static int p[][];
	private static int q[][];
	public static int emsg[][];
	private static int dmsg[][];
	private static int qinv[][];

	public static int rpinv[][];
	private static int rqinv[][];
	private static int mp[][];
	private static int mq[][];

	private static int r2p[][];
	private static int r2q[][];
	private int tid;

	public SSLWorker(int tid) {
		this.tid = tid;
	}

	@Override
	public void run() {
		new CudaKernels().decrypt(ExampleSSL.BITLENGTH, n, pd, qd, e, p, q, emsg, dmsg, qinv, rpinv,
				rqinv, mp, mq, r2p, r2q, ExampleSSL.BATCH_SIZE, ExampleSSL.BASE);
		validate();
	}
	
	public static void validate(){
		int base = (int)Math.pow(2, ExampleSSL.BASE);
		for(int i=0;i<ExampleSSL.BATCH_SIZE;i++){
			print_arr(dmsg[i], "DMSG");
		}
	} 
	
	public static void initVal() {
		int k = ExampleSSL.BITLENGTH / ExampleSSL.BASE;//length of array containing the number represented in BASE
		int kby2 = k / 2;
		bn = new BigInteger[ExampleSSL.BATCH_SIZE];
		bd = new BigInteger[ExampleSSL.BATCH_SIZE];
		bp = new BigInteger[ExampleSSL.BATCH_SIZE];
		n = new int[ExampleSSL.BATCH_SIZE][k];
		pd = new int[ExampleSSL.BATCH_SIZE][kby2];
		qd = new int[ExampleSSL.BATCH_SIZE][kby2];
		e = new int[ExampleSSL.BATCH_SIZE][k];
		p = new int[ExampleSSL.BATCH_SIZE][kby2];
		q = new int[ExampleSSL.BATCH_SIZE][kby2];
		emsg = new int[2 * ExampleSSL.BATCH_SIZE][kby2];
		dmsg = new int[ExampleSSL.BATCH_SIZE][k];
		qinv = new int[ExampleSSL.BATCH_SIZE][kby2];

		rpinv = new int[ExampleSSL.BATCH_SIZE][kby2];
		rqinv = new int[ExampleSSL.BATCH_SIZE][kby2];
		mp = new int[ExampleSSL.BATCH_SIZE][kby2];
		mq = new int[ExampleSSL.BATCH_SIZE][kby2];
		r2p = new int[ExampleSSL.BATCH_SIZE][kby2];
		r2q = new int[ExampleSSL.BATCH_SIZE][kby2];

		BigInteger R = new BigInteger("2").pow(kby2 * ExampleSSL.BASE);
		BigInteger R2 = R.pow(2);
		for (int kk = 0; kk < ExampleSSL.BATCH_SIZE; kk++) {
			bp[kk] = BigInteger.probablePrime(kby2 * ExampleSSL.BASE, new Random());
			BigInteger bq = BigInteger.probablePrime(kby2 * ExampleSSL.BASE, new Random());
			if (bp[kk].compareTo(bq) < 0) {
				// if p<q swap..
				BigInteger temp = new BigInteger(bp[kk].toString());
				bp[kk] = bq;
				bq = temp;
			}
			BigInteger phi = bp[kk].subtract(BigInteger.ONE).multiply(bq.subtract(BigInteger.ONE));
			bd[kk] = new BigInteger("65537").modInverse(phi);
			bn[kk] = bp[kk].multiply(bq);
			n[kk] = conv(
					pad(new StringBuilder(bp[kk].multiply(bq).toString(2)).reverse().toString(), k * ExampleSSL.BASE),
					k);
			BigInteger bpd = bd[kk].mod(bp[kk].subtract(BigInteger.ONE));
			BigInteger bqd = bd[kk].mod(bq.subtract(BigInteger.ONE));

			pd[kk] = conv(pad(new StringBuilder(bpd.toString(2)).reverse().toString(), kby2 * ExampleSSL.BASE), kby2);
			qd[kk] = conv(pad(new StringBuilder(bqd.toString(2)).reverse().toString(), kby2 * ExampleSSL.BASE), kby2);
			e[kk] = conv(pad(new StringBuilder(new BigInteger("65537").toString(2)).reverse().toString(),
					k * ExampleSSL.BASE), k);
			p[kk] = conv(pad(new StringBuilder(bp[kk].toString(2)).reverse().toString(), kby2 * ExampleSSL.BASE),
					kby2);
			q[kk] = conv(pad(new StringBuilder(bq.toString(2)).reverse().toString(), kby2 * ExampleSSL.BASE), kby2);
			emsg[kk] = conv(
					pad(new StringBuilder(bp[kk].mod(bpd).toString(2)).reverse().toString(), kby2 * ExampleSSL.BASE),
					kby2);
			emsg[kk + ExampleSSL.BATCH_SIZE] = conv(
					pad(new StringBuilder(bp[kk].mod(bqd).toString(2)).reverse().toString(), kby2 * ExampleSSL.BASE),
					kby2);

			dmsg[kk] = new int[k];
			qinv[kk] = conv(pad(new StringBuilder(bq.modInverse(bp[kk]).toString(2)).reverse().toString(),
					kby2 * ExampleSSL.BASE), kby2);

			BigInteger RinvP = R.modInverse(bp[kk]);
			BigInteger RinvQ = R.modInverse(bq);

			rpinv[kk] = conv(pad(new StringBuilder(RinvP.toString(2)).reverse().toString(), kby2 * ExampleSSL.BASE),
					kby2);
			rqinv[kk] = conv(pad(new StringBuilder((RinvQ).toString(2)).reverse().toString(), kby2 * ExampleSSL.BASE),
					kby2);
			;
			mp[kk] = conv(
					pad(new StringBuilder((((RinvP.multiply(R)).subtract(BigInteger.ONE)).divide(bp[kk])).toString(2))
							.reverse().toString(), kby2 * ExampleSSL.BASE),
					kby2);
			mq[kk] = conv(pad(new StringBuilder((((RinvQ.multiply(R)).subtract(BigInteger.ONE)).divide(bq)).toString(2))
					.reverse().toString(), kby2 * ExampleSSL.BASE), kby2);

			r2p[kk] = conv(
					pad(new StringBuilder(R2.mod(bp[kk]).toString(2)).reverse().toString(), kby2 * ExampleSSL.BASE),
					kby2);
			r2q[kk] = conv(
					pad(new StringBuilder(R2.mod(bq).toString(2)).reverse().toString(), kby2 * ExampleSSL.BASE),
					kby2);
			print_arr(p[kk], "P");
			print_arr(q[kk], "Q");
			print_arr(e[kk], "E");
			print_arr(bd, "D");
		}
		
	}

	public static void print_arr(int[] a, String label){
		System.out.print(label);
		System.out.print(": ");
		for(int i=0;i<a.length;i++){
			System.out.print(a[i]+" ");
		}
		System.out.println();
	}
	public static void print_arr(BigInteger[] a, String label){
		System.out.print(label);
		System.out.print(": ");
		for(int i=0;i<a.length;i++){
			System.out.print(a[i].toString()+" ");
		}
		System.out.println();
	}
	public static String pad(String str, int bl) {
		String paddedStr = str;
		for (int i = 0; i < bl - str.length(); i++) {
			paddedStr = paddedStr + "0";
		}
		return paddedStr;
	}

	public static int[] conv(String str, int bl) {
		int[] baseDigs = new int[bl];
		for (int i = 0; i < bl; i++) {
			baseDigs[i] = Integer
					.valueOf(new StringBuilder(str.substring(i * ExampleSSL.BASE, (i + 1) * ExampleSSL.BASE))
							.reverse().toString(), 2);
		}
		return baseDigs;
	}
	
	public static BigInteger toBI(int a[], int base){
		BigInteger bi = new BigInteger("0");
		BigInteger pow = new BigInteger("1");
		BigInteger bbase = new BigInteger(""+base);
		for(int i=0;i<a.length;i++){
			bi = bi.add(new BigInteger(""+a[i]).multiply(pow));
			pow = pow.multiply(bbase);
			
		}
		return bi;
	}
	public static BigInteger decrypt(int c[], BigInteger D, BigInteger N, int base){
		BigInteger C = toBI(c, base);
		BigInteger M = C.modPow(D, N);
		return M;
	}
}
