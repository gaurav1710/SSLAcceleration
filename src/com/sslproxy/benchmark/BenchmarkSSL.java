package com.sslproxy.benchmark;

import java.math.BigInteger;
import java.util.Date;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import com.sslproxy.jni.CudaKernels;

/**
 * Bechmark the SSL offloading code
 * 
 * @author gaurav(mcs132556)
 */
public class BenchmarkSSL {
	public static int NUMCORES = 8;
	public static int NUMWORKERS = 1;
	public static int BITLENGTH = 32;
	public static int EXPS = 1;
	public static int BATCH_SIZE = 1;
	public static String IMAGEPATH = "./ResFiles/";
	public static String FILEPATH = "";
	public static int BASE = 16;
	public static boolean cpuExecution = false;
	public static boolean saveResults = false;

	public static void main(String[] args) {
		processArgs(args);
		System.out.println("Setup Details:");
		System.out.println("NUMWORKERS=" + NUMWORKERS);
		System.out.println("EXPS_NUM=" + EXPS);
		System.out.println("BITLENGTH_START=" + BITLENGTH);
		System.out.println("BATCH_SIZE=" + BATCH_SIZE);
		System.out.println("IMAGEPATH=" + IMAGEPATH);
		System.out.println("FILEPATH=" + FILEPATH);

		DataCollector.reset();
		BITLENGTH /= BASE;
		if (BITLENGTH % BASE != 0)
			BITLENGTH++;
		for (int test = 0; test < EXPS; test++) {
			// System.out.println("Starting experiment "+test);
			
			SSLWorker.initVal();
			ExecutorService threadpool = Executors.newFixedThreadPool(NUMCORES);
			long startTime = System.currentTimeMillis();
			for (int i = 0; i < NUMWORKERS; i++) {
				threadpool.execute(new Thread(cpuExecution ? new SSLCPUWorker() : new SSLWorker(i)));
			}
			threadpool.shutdown();
			double avgLatency = 0.0;
			try {
				threadpool.awaitTermination(1, TimeUnit.DAYS);
				startTime = System.currentTimeMillis() - startTime;
				avgLatency = (double) startTime / (double) (NUMWORKERS);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			System.out.println("Experiment " + test + " ends with lat=" + avgLatency);
			DataCollector.addPoint(new DataPoint(2 * BITLENGTH, avgLatency));
			BITLENGTH++;
		}
		if(saveResults){
			DataCollector.saveInFile(FILEPATH + "Res" + new Date().toString() + ".csv");
			Plotter.plot(IMAGEPATH + "latencyVsBitLength.jpeg", "BitLength Vs Latency", "Avg. Latency(ms)",
					"RSA Bit Length");
		}		
		
	}

	private static void processArgs(String[] args) {
		if (args.length > 0) {
			NUMWORKERS = Integer.parseInt(args[0]);
			if (args.length > 1) {
				EXPS = Integer.parseInt(args[1]);
			}
			if (args.length > 2) {
				BITLENGTH = Integer.parseInt(args[2]);
			}
			if (args.length > 3) {
				BATCH_SIZE = Integer.parseInt(args[3]);
			}
			if (args.length > 4) {
				BASE = Integer.parseInt(args[4]);
			}
			if (args.length > 5) {
				IMAGEPATH = args[5];
			}
		}
	}
}

class SSLCPUWorker implements Runnable {

	@Override
	public void run() {
		BenchmarkSSLCPU cpuBenchmark = new BenchmarkSSLCPU(SSLWorker.bn[0], SSLWorker.bd[0],
				SSLWorker.bp[0].toByteArray());
		cpuBenchmark.decrypt();
		System.out.println("Decrypted premaster secret:" + new String(cpuBenchmark.getDecryptMessage()));
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
		// System.out.println("T"+tid+": Starting offload now...");
		int[][] mesg = new CudaKernels().decrypt(BenchmarkSSL.BITLENGTH, n, pd, qd, e, p, q, emsg, dmsg, qinv, rpinv,
				rqinv, mp, mq, r2p, r2q, BenchmarkSSL.BATCH_SIZE, BenchmarkSSL.BASE);
		print_arr(emsg[0], "Msg");
		print_arr(dmsg[0], "M1M2");
		// System.out.println("Decrypted message:"+new String(mesg));
		// System.out.println("T"+tid+": Done.");
	}

	public static void initVal() {
		int k = BenchmarkSSL.BITLENGTH;
		int kby2 = BenchmarkSSL.BITLENGTH / 2;
		bn = new BigInteger[BenchmarkSSL.BATCH_SIZE];
		bd = new BigInteger[BenchmarkSSL.BATCH_SIZE];
		bp = new BigInteger[BenchmarkSSL.BATCH_SIZE];
		n = new int[BenchmarkSSL.BATCH_SIZE][k];
		pd = new int[BenchmarkSSL.BATCH_SIZE][kby2];
		qd = new int[BenchmarkSSL.BATCH_SIZE][kby2];
		e = new int[BenchmarkSSL.BATCH_SIZE][k];
		p = new int[BenchmarkSSL.BATCH_SIZE][kby2];
		q = new int[BenchmarkSSL.BATCH_SIZE][kby2];
		emsg = new int[2 * BenchmarkSSL.BATCH_SIZE][kby2];
		dmsg = new int[BenchmarkSSL.BATCH_SIZE][k];
		qinv = new int[BenchmarkSSL.BATCH_SIZE][kby2];

		rpinv = new int[BenchmarkSSL.BATCH_SIZE][kby2];
		rqinv = new int[BenchmarkSSL.BATCH_SIZE][kby2];
		mp = new int[BenchmarkSSL.BATCH_SIZE][kby2];
		mq = new int[BenchmarkSSL.BATCH_SIZE][kby2];
		r2p = new int[BenchmarkSSL.BATCH_SIZE][kby2];
		r2q = new int[BenchmarkSSL.BATCH_SIZE][kby2];

		BigInteger R = new BigInteger("2").pow(kby2 * BenchmarkSSL.BASE);
		BigInteger R2 = R.pow(2);
		for (int kk = 0; kk < BenchmarkSSL.BATCH_SIZE; kk++) {
			bp[kk] = BigInteger.probablePrime(kby2 * BenchmarkSSL.BASE, new Random());
			BigInteger bq = BigInteger.probablePrime(kby2 * BenchmarkSSL.BASE, new Random());
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
					pad(new StringBuilder(bp[kk].multiply(bq).toString(2)).reverse().toString(), k * BenchmarkSSL.BASE),
					k);
			BigInteger bpd = bd[kk].mod(bp[kk].subtract(BigInteger.ONE));
			BigInteger bqd = bd[kk].mod(bq.subtract(BigInteger.ONE));

			pd[kk] = conv(pad(new StringBuilder(bpd.toString(2)).reverse().toString(), kby2 * BenchmarkSSL.BASE), kby2);
			qd[kk] = conv(pad(new StringBuilder(bqd.toString(2)).reverse().toString(), kby2 * BenchmarkSSL.BASE), kby2);
			e[kk] = conv(pad(new StringBuilder(new BigInteger("65537").toString(2)).reverse().toString(),
					k * BenchmarkSSL.BASE), k);
			p[kk] = conv(pad(new StringBuilder(bp[kk].toString(2)).reverse().toString(), kby2 * BenchmarkSSL.BASE),
					kby2);
			q[kk] = conv(pad(new StringBuilder(bq.toString(2)).reverse().toString(), kby2 * BenchmarkSSL.BASE), kby2);
			emsg[kk] = conv(
					pad(new StringBuilder(bp[kk].mod(bpd).toString(2)).reverse().toString(), kby2 * BenchmarkSSL.BASE),
					kby2);
			emsg[kk + BenchmarkSSL.BATCH_SIZE] = conv(
					pad(new StringBuilder(bp[kk].mod(bqd).toString(2)).reverse().toString(), kby2 * BenchmarkSSL.BASE),
					kby2);

			dmsg[kk] = new int[k];
			qinv[kk] = conv(pad(new StringBuilder(bq.modInverse(bp[kk]).toString(2)).reverse().toString(),
					kby2 * BenchmarkSSL.BASE), kby2);

			BigInteger RinvP = R.modInverse(bp[kk]);
			BigInteger RinvQ = R.modInverse(bq);

			rpinv[kk] = conv(pad(new StringBuilder(RinvP.toString(2)).reverse().toString(), kby2 * BenchmarkSSL.BASE),
					kby2);
			rqinv[kk] = conv(pad(new StringBuilder((RinvQ).toString(2)).reverse().toString(), kby2 * BenchmarkSSL.BASE),
					kby2);
			;
			mp[kk] = conv(
					pad(new StringBuilder((((RinvP.multiply(R)).subtract(BigInteger.ONE)).divide(bp[kk])).toString(2))
							.reverse().toString(), kby2 * BenchmarkSSL.BASE),
					kby2);
			mq[kk] = conv(pad(new StringBuilder((((RinvQ.multiply(R)).subtract(BigInteger.ONE)).divide(bq)).toString(2))
					.reverse().toString(), kby2 * BenchmarkSSL.BASE), kby2);

			r2p[kk] = conv(
					pad(new StringBuilder(R2.mod(bp[kk]).toString(2)).reverse().toString(), kby2 * BenchmarkSSL.BASE),
					kby2);
			r2q[kk] = conv(
					pad(new StringBuilder(R2.mod(bq).toString(2)).reverse().toString(), kby2 * BenchmarkSSL.BASE),
					kby2);

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
					.valueOf(new StringBuilder(str.substring(i * BenchmarkSSL.BASE, (i + 1) * BenchmarkSSL.BASE))
							.reverse().toString(), 2);
		}
		return baseDigs;
	}
}
