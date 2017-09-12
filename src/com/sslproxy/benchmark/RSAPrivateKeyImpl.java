package com.sslproxy.benchmark;

import java.io.IOException;
import java.math.BigInteger;
import java.security.AccessController;
import java.security.InvalidKeyException;
import java.security.interfaces.RSAPrivateKey;

import sun.security.action.GetPropertyAction;
import sun.security.pkcs.PKCS8Key;
import sun.security.rsa.RSAKeyFactory;
import sun.security.util.DerOutputStream;
import sun.security.util.DerValue;
import sun.security.x509.AlgorithmId;

public final class RSAPrivateKeyImpl extends PKCS8Key implements RSAPrivateKey {
	private static final long serialVersionUID = -33106691987952810L;
	private final BigInteger n;
	private final BigInteger d;
	private static final boolean restrictExpLen = "true"
			.equalsIgnoreCase((String) AccessController
					.doPrivileged(new GetPropertyAction(
							"sun.security.rsa.restrictRSAExponent", "true")));
	public RSAPrivateKeyImpl(BigInteger n, BigInteger d) throws InvalidKeyException {
		this.n = n;
		this.d = d;
	    checkRSAProviderKeyLengths(n.bitLength(), null);

		this.algid = new AlgorithmId(
				AlgorithmId.RSAEncryption_oid);;
		try {
			DerOutputStream out = new DerOutputStream();
			out.putInteger(0);
			out.putInteger(n);
			out.putInteger(0);
			out.putInteger(d);
			out.putInteger(0);
			out.putInteger(0);
			out.putInteger(0);
			out.putInteger(0);
			out.putInteger(0);
			DerValue val = new DerValue((byte)48, out.toByteArray());

			this.key = val.toByteArray();
		} catch (IOException exc) {
			throw new InvalidKeyException(exc);
		}
	}
	static void checkRSAProviderKeyLengths(int modulusLen, BigInteger exponent)
			throws InvalidKeyException {
		checkKeyLengths(modulusLen + 7 & 0xFFFFFFF8, exponent, 0, 2147483647);
	}
	public static void checkKeyLengths(int modulusLen, BigInteger exponent,
			int minModulusLen, int maxModulusLen) throws InvalidKeyException {
		if ((minModulusLen > 0) && (modulusLen < minModulusLen)) {
			throw new InvalidKeyException("RSA keys must be at least "
					+ minModulusLen + " bits long");
		}

		int maxLen = Math.min(maxModulusLen, 16384);

		if (modulusLen > maxLen) {
			throw new InvalidKeyException("RSA keys must be no longer than "
					+ maxLen + " bits");
		}

		if ((!(restrictExpLen)) || (exponent == null) || (modulusLen <= 3072)
				|| (exponent.bitLength() <= 64)) {
			return;
		}
		throw new InvalidKeyException(
				"RSA exponents can be no longer than 64 bits  if modulus is greater than 3072 bits");
	}
	public String getAlgorithm() {
		return "RSA";
	}

	public BigInteger getModulus() {
		return this.n;
	}

	public BigInteger getPrivateExponent() {
		return this.d;
	}
}