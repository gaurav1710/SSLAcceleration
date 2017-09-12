package com.sslproxy.benchmark;

import java.math.BigInteger;
import java.security.SecureRandom;
import java.security.interfaces.RSAKey;
import java.security.interfaces.RSAPrivateCrtKey;
import java.security.interfaces.RSAPrivateKey;
import java.security.interfaces.RSAPublicKey;
import java.util.Map;
import java.util.WeakHashMap;

import javax.crypto.BadPaddingException;

import sun.security.jca.JCAUtil;

public final class RSACore {
	private static final boolean ENABLE_BLINDING = true;
	private static final Map<BigInteger, BlindingParameters> blindingCache;

	public static int getByteLength(BigInteger b) {
		int n = b.bitLength();
		return (n + 7 >> 3);
	}

	public static int getByteLength(RSAKey key) {
		return getByteLength(key.getModulus());
	}

	public static byte[] convert(byte[] b, int ofs, int len) {
		if ((ofs == 0) && (len == b.length)) {
			return b;
		}
		byte[] t = new byte[len];
		System.arraycopy(b, ofs, t, 0, len);
		return t;
	}

	public static byte[] rsa(byte[] msg, RSAPublicKey key)
			throws BadPaddingException {
		return crypt(msg, key.getModulus(), key.getPublicExponent());
	}

	@Deprecated
	public static byte[] rsa(byte[] msg, RSAPrivateKey key)
			throws BadPaddingException {
		return rsa(msg, key, true);
	}

	public static byte[] rsa(byte[] msg, RSAPrivateKey key, boolean verify)
			throws BadPaddingException {
		if (key instanceof RSAPrivateCrtKey) {
			return crtCrypt(msg, (RSAPrivateCrtKey) key, verify);
		}
		return priCrypt(msg, key.getModulus(), key.getPrivateExponent());
	}

	private static byte[] crypt(byte[] msg, BigInteger n, BigInteger exp)
			throws BadPaddingException {
		BigInteger m = parseMsg(msg, n);
		BigInteger c = m.modPow(exp, n);
		return toByteArray(c, getByteLength(n));
	}

	private static byte[] priCrypt(byte[] msg, BigInteger n, BigInteger exp)
			throws BadPaddingException {
		BigInteger c = parseMsg(msg, n);
		BlindingRandomPair brp = null;

		brp = getBlindingRandomPair(null, exp, n);
		c = c.multiply(brp.u).mod(n);
		BigInteger m = c.modPow(exp, n);
		m = m.multiply(brp.v).mod(n);

		return toByteArray(m, getByteLength(n));
	}

	private static byte[] crtCrypt(byte[] msg, RSAPrivateCrtKey key,
			boolean verify) throws BadPaddingException {
		BigInteger n = key.getModulus();
		BigInteger c0 = parseMsg(msg, n);
		BigInteger c = c0;
		BigInteger p = key.getPrimeP();
		BigInteger q = key.getPrimeQ();
		BigInteger dP = key.getPrimeExponentP();
		BigInteger dQ = key.getPrimeExponentQ();
		BigInteger qInv = key.getCrtCoefficient();
		BigInteger e = key.getPublicExponent();
		BigInteger d = key.getPrivateExponent();

		BlindingRandomPair brp = getBlindingRandomPair(e, d, n);
		c = c.multiply(brp.u).mod(n);

		BigInteger m1 = c.modPow(dP, p);

		BigInteger m2 = c.modPow(dQ, q);

		BigInteger mtmp = m1.subtract(m2);
		if (mtmp.signum() < 0) {
			mtmp = mtmp.add(p);
		}
		BigInteger h = mtmp.multiply(qInv).mod(p);

		BigInteger m = h.multiply(q).add(m2);

		m = m.multiply(brp.v).mod(n);

		if ((verify) && (!(c0.equals(m.modPow(e, n))))) {
			throw new BadPaddingException("RSA private key operation failed");
		}

		return toByteArray(m, getByteLength(n));
	}

	private static BigInteger parseMsg(byte[] msg, BigInteger n)
			throws BadPaddingException {
		BigInteger m = new BigInteger(1, msg);
		if (m.compareTo(n) >= 0) {
			throw new BadPaddingException("Message is larger than modulus");
		}
		return m;
	}

	private static byte[] toByteArray(BigInteger bi, int len) {
		byte[] b = bi.toByteArray();
		int n = b.length;
		if (n == len) {
			return b;
		}

		if ((n == len + 1) && (b[0] == 0)) {
			byte[] t = new byte[len];
			System.arraycopy(b, 1, t, 0, len);
			return t;
		}

		assert (n < len);
		byte[] t = new byte[len];
		System.arraycopy(b, 0, t, len - n, n);
		return t;
	}

	private static BlindingRandomPair getBlindingRandomPair(BigInteger e,
			BigInteger d, BigInteger n) {
		BlindingParameters bps = null;
		synchronized (blindingCache) {
			bps = (BlindingParameters) blindingCache.get(n);
		}

		if (bps == null) {
			bps = new BlindingParameters(e, d, n);
			synchronized (blindingCache) {
				if (blindingCache.get(n) == null) {
					blindingCache.put(n, bps);
				}
			}
		}

		BlindingRandomPair brp = bps.getBlindingRandomPair(e, d, n);
		if (brp == null) {
			bps = new BlindingParameters(e, d, n);
			synchronized (blindingCache) {
				if (blindingCache.get(n) != null) {
					blindingCache.put(n, bps);
				}
			}
			brp = bps.getBlindingRandomPair(e, d, n);
		}

		return ((BlindingRandomPair) brp);
	}

	static {
		blindingCache = new WeakHashMap();
	}

	private static final class BlindingParameters {
		private static final BigInteger BIG_TWO = BigInteger.valueOf(2L);
		private final BigInteger e;
		private final BigInteger d;
		private BigInteger u;
		private BigInteger v;

		BlindingParameters(BigInteger e, BigInteger d, BigInteger n) {
			this.u = null;
			this.v = null;
			this.e = e;
			this.d = d;

			int len = n.bitLength();
			SecureRandom random = JCAUtil.getSecureRandom();
			this.u = new BigInteger(len, random).mod(n);

			if (this.u.equals(BigInteger.ZERO)) {
				this.u = BigInteger.ONE;
			}

			try {
				this.v = this.u.modInverse(n);
			} catch (ArithmeticException ae) {
				this.u = BigInteger.ONE;
				this.v = BigInteger.ONE;
			}

			if (e != null) {
				this.u = this.u.modPow(e, n);
			} else {
				this.v = this.v.modPow(d, n);
			}
		}

		RSACore.BlindingRandomPair getBlindingRandomPair(BigInteger e,
				BigInteger d, BigInteger n) {
			if (((this.e != null) && (this.e.equals(e)))
					|| ((this.d != null) && (this.d.equals(d)))) {
				RSACore.BlindingRandomPair brp = null;
				synchronized (this) {
					if ((!(this.u.equals(BigInteger.ZERO)))
							&& (!(this.v.equals(BigInteger.ZERO)))) {
						brp = new RSACore.BlindingRandomPair(this.u, this.v);
						if ((this.u.compareTo(BigInteger.ONE) <= 0)
								|| (this.v.compareTo(BigInteger.ONE) <= 0)) {
							this.u = BigInteger.ZERO;
							this.v = BigInteger.ZERO;
						} else {
							this.u = this.u.modPow(BIG_TWO, n);
							this.v = this.v.modPow(BIG_TWO, n);
						}
					}
				}
				return brp;
			}

			return null;
		}
	}

	private static final class BlindingRandomPair {
		final BigInteger u;
		final BigInteger v;

		BlindingRandomPair(BigInteger u, BigInteger v) {
			this.u = u;
			this.v = v;
		}
	}
}
