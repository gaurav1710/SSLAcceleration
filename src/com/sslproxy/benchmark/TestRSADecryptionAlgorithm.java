package com.sslproxy.benchmark;

import java.math.BigInteger;

/**
 * Test and validate the RSA decryption algorithm using
 * {@link java.math.BigInteger}
 * 
 * @author mcs132556
 *
 */
public class TestRSADecryptionAlgorithm {

	public static void main(String[] args) {
		calM1M2();
	}

	public static void calM1M2() {
		BigInteger x = new BigInteger("612769785");
		BigInteger y = new BigInteger("1243321573");
		BigInteger m = new BigInteger("3099412931");
		BigInteger R = new BigInteger("4294967296");
		BigInteger Mbar = new BigInteger("1928401173");
		BigInteger R2modm = new BigInteger("440608158");
		BigInteger m1 = sqAndMul(x, y, m, R, Mbar, R2modm);

		x = new BigInteger("612769785");
		y = new BigInteger("1953616713");
		m = new BigInteger("2154188249");
		Mbar = new BigInteger("2163907991");
		R2modm = new BigInteger("913509272");
		BigInteger m2 = sqAndMul(x, y, m, R, Mbar, R2modm);

		BigInteger p = new BigInteger("3099412931");
		BigInteger q = new BigInteger("2154188249");
		BigInteger qinv = new BigInteger("1048679846");

		BigInteger t1 = m1.subtract(m2);
		System.out.println("M1-M2=" + t1);
		BigInteger t2 = (t1.multiply(qinv)).mod(p);
		System.out.println("M1-M2*q^inv mod p=" + t2);
		BigInteger t3 = t2.multiply(q);
		System.out.println("(M1-M2*q^inv mod p)*q=" + t3);
		System.out.println(m2.add(t3));
		System.out.println("M2+(M1-M2*q^inv mod p)*q=" + m2.add(t3));
	}

	public static BigInteger sqAndMul(BigInteger x, BigInteger y, BigInteger m, BigInteger R, BigInteger Mbar,
			BigInteger R2modm) {
		BigInteger a = new BigInteger("1");
		BigInteger g = new BigInteger("1");
		String ybits = y.toString(2);
		for (int i = 0; i < 32 - ybits.length(); i++) {
			ybits = "0" + ybits;
		}
		a = montgomery(BigInteger.ONE, R2modm, m, R, Mbar);
		g = montgomery(x, R2modm, m, R, Mbar);
		System.out.println(a);
		System.out.println(g.toString() + " " + x.multiply(R).mod(m));
		System.out.println(ybits);
		for (int i = 0; i < ybits.length(); i++) {
			a = montgomery(a, a, m, R, Mbar);
			if (ybits.charAt(i) == '1') {
				a = montgomery(a, g, m, R, Mbar);
			}
		}
		a = montgomery(a, BigInteger.ONE, m, R, Mbar);
		System.out.println(x.toString() + "^" + y.toString() + "%" + m.toString() + "=" + a.toString());
		basesqAndMul(x, y, m);
		return a;

	}

	public static BigInteger montgomery(BigInteger a, BigInteger b, BigInteger m, BigInteger R, BigInteger Mbar) {

		BigInteger T = new BigInteger("0");
		BigInteger M = new BigInteger("0");
		BigInteger U = new BigInteger("0");

		T = a.multiply(b);
		M = T.multiply(Mbar).mod(R);
		U = (T.add(M.multiply(m))).divide(R);
		if (U.compareTo(m) >= 0)
			U = U.subtract(m);
		return U;

	}

	public static BigInteger basesqAndMul(BigInteger a, BigInteger b, BigInteger m) {
		BigInteger res = BigInteger.ONE;
		String ybits = b.toString(2);

		for (int i = 0; i < ybits.length(); i++) {
			res = (res.multiply(res)).mod(m);
			if (ybits.charAt(i) == '1') {
				res = (res.multiply(a)).mod(m);
			}

		}
		System.out.println(res.toString());
		return res;

	}

	public static void print(int a[]) {

		for (int i = 0; i < a.length; i++) {
			System.out.print(a[i] + " ");
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
