package com.sslproxy.benchmark;

import java.math.BigInteger;
import java.security.InvalidKeyException;
import java.security.interfaces.RSAPrivateKey;

import javax.crypto.BadPaddingException;

public class BenchmarkSSLCPU {
	private byte[] dmesg;
	private byte[] mesg;
	private RSAPrivateKey privateKey;

	public BenchmarkSSLCPU(BigInteger n, BigInteger d, byte[] dmesg) {
		this.dmesg = dmesg;
		try {
			privateKey = new RSAPrivateKeyImpl(n, d);
		} catch (InvalidKeyException e) {
			e.printStackTrace();
		}
	}

	public void decrypt() {
		try {
			mesg = RSACore.rsa(dmesg, this.privateKey);
		} catch (BadPaddingException e) {
			e.printStackTrace();
		}
	}

	public byte[] getDecryptMessage() {
		return mesg;
	}
}
