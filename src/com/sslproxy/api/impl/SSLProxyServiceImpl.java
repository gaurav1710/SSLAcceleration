package com.sslproxy.api.impl;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

import com.sslproxy.api.Message;
import com.sslproxy.api.Request;
import com.sslproxy.api.SSLProxyService;
import com.sslproxy.example.ExampleSSL;
import com.sslproxy.jni.CudaKernels;

public class SSLProxyServiceImpl implements SSLProxyService {

	private static int BASE = 16; //Base 2^(BASE)
	@Override
	public List<Message> decrypt(List<Request> requests, int bitlength) {
		System.out.println("INFO: Received request size:"+requests.size()+" bitlength:"+bitlength);
		List<Message> responses = new ArrayList<Message>();
		int k = bitlength / BASE;//length of array containing the number represented in BASE
		int kby2 = k / 2;
		int batchSize = requests.size();
		int [][] n = new int[batchSize][k];
		int [][] pd = new int[batchSize][kby2];
		int [][] qd = new int[batchSize][kby2];
		int [][] e = new int[batchSize][k];
		int [][] p = new int[batchSize][kby2];
		int [][] q = new int[batchSize][kby2];
		int [][] emsg = new int[2 * batchSize][kby2];
		int [][] dmsg = new int[batchSize][k];
		int [][] qinv = new int[batchSize][kby2];
		int [][] rpinv = new int[batchSize][kby2];
		int [][] rqinv = new int[batchSize][kby2];
		int [][] mp = new int[batchSize][kby2];
		int [][] mq = new int[batchSize][kby2];
		int [][] r2p = new int[batchSize][kby2];
		int [][] r2q = new int[batchSize][kby2];
		BigInteger R = new BigInteger("2").pow(kby2 * BASE);
		BigInteger R2 = R.pow(2);
		for (int kk = 0; kk < requests.size(); kk++) {
			Request req = requests.get(kk);
			System.out.println("INFO: Adding request("+kk+"):"+req.toString());
			BigInteger bp = new BigInteger(req.getP());
			BigInteger bq = new BigInteger(req.getQ());
			if (bp.compareTo(bq) < 0) {
				// if p<q swap..
				BigInteger temp = new BigInteger(bp.toString());
				bp = bq;
				bq = temp;
			}
			BigInteger bd = new BigInteger(req.getD());
			BigInteger bn = bp.multiply(bq);
			n[kk] = conv(
					pad(new StringBuilder(bp.multiply(bq).toString(2)).reverse().toString(), k * BASE),
					k);
			BigInteger bpd = bd.mod(bp.subtract(BigInteger.ONE));
			BigInteger bqd = bd.mod(bq.subtract(BigInteger.ONE));

			pd[kk] = conv(pad(new StringBuilder(bpd.toString(2)).reverse().toString(), kby2 * BASE), kby2);
			qd[kk] = conv(pad(new StringBuilder(bqd.toString(2)).reverse().toString(), kby2 * BASE), kby2);
			e[kk] = conv(pad(new StringBuilder(new BigInteger(req.getE()).toString(2)).reverse().toString(),
					k * BASE), k);
			p[kk] = conv(pad(new StringBuilder(bp.toString(2)).reverse().toString(), kby2 * BASE),
					kby2);
			q[kk] = conv(pad(new StringBuilder(bq.toString(2)).reverse().toString(), kby2 * BASE), kby2);
			emsg[kk] = conv(
					pad(new StringBuilder(new BigInteger(req.getCipherText()).mod(bpd).toString(2)).reverse().toString(), kby2 * BASE),
					kby2);
			emsg[kk + batchSize] = conv(
					pad(new StringBuilder(new BigInteger(req.getCipherText()).mod(bqd).toString(2)).reverse().toString(), kby2 * BASE),
					kby2);

			dmsg[kk] = new int[k];
			qinv[kk] = conv(pad(new StringBuilder(bq.modInverse(bp).toString(2)).reverse().toString(),
					kby2 * BASE), kby2);

			BigInteger RinvP = R.modInverse(bp);
			BigInteger RinvQ = R.modInverse(bq);

			rpinv[kk] = conv(pad(new StringBuilder(RinvP.toString(2)).reverse().toString(), kby2 * BASE),
					kby2);
			rqinv[kk] = conv(pad(new StringBuilder((RinvQ).toString(2)).reverse().toString(), kby2 * BASE),
					kby2);
			;
			mp[kk] = conv(
					pad(new StringBuilder((((RinvP.multiply(R)).subtract(BigInteger.ONE)).divide(bp)).toString(2))
							.reverse().toString(), kby2 * BASE),
					kby2);
			mq[kk] = conv(pad(new StringBuilder((((RinvQ.multiply(R)).subtract(BigInteger.ONE)).divide(bq)).toString(2))
					.reverse().toString(), kby2 * BASE), kby2);

			r2p[kk] = conv(
					pad(new StringBuilder(R2.mod(bp).toString(2)).reverse().toString(), kby2 * BASE),
					kby2);
			r2q[kk] = conv(
					pad(new StringBuilder(R2.mod(bq).toString(2)).reverse().toString(), kby2 * BASE),
					kby2);
		}
		System.out.println("INFO: Starting decryption on GPUs..");
		new CudaKernels().decrypt(bitlength, n, pd, qd, e, p, q, emsg, dmsg, qinv, rpinv,
				rqinv, mp, mq, r2p, r2q, requests.size(), BASE);
		System.out.println("INFO: Done decryption on GPUs..");
		int basePow = (int)Math.pow(2, BASE);
		for(int j=0;j<batchSize;j++){
			responses.add(new Message(toBI(dmsg[j], basePow).toString()));
		}
		return responses;
	}
	
	private void print_arr(int[] a, String label){
		System.out.print(label);
		System.out.print(": ");
		for(int i=0;i<a.length;i++){
			System.out.print(a[i]+" ");
		}
		System.out.println();
	}
	private void print_arr(BigInteger[] a, String label){
		System.out.print(label);
		System.out.print(": ");
		for(int i=0;i<a.length;i++){
			System.out.print(a[i].toString()+" ");
		}
		System.out.println();
	}
	private String pad(String str, int bl) {
		String paddedStr = str;
		for (int i = 0; i < bl - str.length(); i++) {
			paddedStr = paddedStr + "0";
		}
		return paddedStr;
	}

	private int[] conv(String str, int bl) {
		int[] baseDigs = new int[bl];
		for (int i = 0; i < bl; i++) {
			baseDigs[i] = Integer
					.valueOf(new StringBuilder(str.substring(i * BASE, (i + 1) * BASE))
							.reverse().toString(), 2);
		}
		return baseDigs;
	}
	
	private BigInteger toBI(int a[], int base){
		BigInteger bi = new BigInteger("0");
		BigInteger pow = new BigInteger("1");
		BigInteger bbase = new BigInteger(""+base);
		for(int i=0;i<a.length;i++){
			bi = bi.add(new BigInteger(""+a[i]).multiply(pow));
			pow = pow.multiply(bbase);
			
		}
		return bi;
	}
	private BigInteger decrypt(int c[], BigInteger D, BigInteger N, int base){
		BigInteger C = toBI(c, base);
		BigInteger M = C.modPow(D, N);
		return M;
	}
}
