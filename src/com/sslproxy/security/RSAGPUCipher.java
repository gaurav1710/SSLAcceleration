package com.sslproxy.security;

import java.security.AlgorithmParameters;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.Key;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.security.SecureRandom;
import java.security.interfaces.RSAKey;
import java.security.interfaces.RSAPrivateKey;
import java.security.interfaces.RSAPublicKey;
import java.security.spec.AlgorithmParameterSpec;
import java.security.spec.InvalidParameterSpecException;
import java.security.spec.MGF1ParameterSpec;
import java.util.BitSet;
import java.util.Locale;

import javax.crypto.BadPaddingException;
import javax.crypto.CipherSpi;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.NoSuchPaddingException;
import javax.crypto.ShortBufferException;
import javax.crypto.spec.OAEPParameterSpec;
import javax.crypto.spec.PSource;

import sun.security.jca.Providers;
import sun.security.rsa.RSACore;
import sun.security.rsa.RSAKeyFactory;
import sun.security.rsa.RSAPadding;
import sun.security.rsa.RSAPrivateCrtKeyImpl;

import com.sslproxy.jni.CudaKernels;



public class RSAGPUCipher extends CipherSpi{
	private static final byte[] B0 = new byte[0];
	private static final int MODE_ENCRYPT = 1;
	private static final int MODE_DECRYPT = 2;
	private static final int MODE_SIGN = 3;
	private static final int MODE_VERIFY = 4;
	private static final String PAD_NONE = "NoPadding";
	private static final String PAD_PKCS1 = "PKCS1Padding";
	private static final String PAD_OAEP_MGF1 = "OAEP";
	private int mode;
	private String paddingType;
	private RSAPadding padding;
	private OAEPParameterSpec spec = null;
	private byte[] buffer;
	private int bufOfs;
	private int outputSize;
	private RSAPublicKey publicKey;
	private RSAPrivateKey privateKey;
	private String oaepHashAlgorithm = "SHA-1";

	public RSAGPUCipher() {
		this.paddingType = "PKCS1Padding";
		System.out.println("New cipher invoked...........");
	}

	protected void engineSetMode(String mode) throws NoSuchAlgorithmException {
		if (!(mode.equalsIgnoreCase("ECB")))
			throw new NoSuchAlgorithmException("Unsupported mode " + mode);
	}

	protected void engineSetPadding(String paddingName)
			throws NoSuchPaddingException {
		if (paddingName.equalsIgnoreCase("NoPadding")) {
			this.paddingType = "NoPadding";
		} else if (paddingName.equalsIgnoreCase("PKCS1Padding")) {
			this.paddingType = "PKCS1Padding";
		} else {
			String lowerPadding = paddingName.toLowerCase(Locale.ENGLISH);
			if (lowerPadding.equals("oaeppadding")) {
				this.paddingType = "OAEP";
			} else {
				if ((lowerPadding.startsWith("oaepwith"))
						&& (lowerPadding.endsWith("andmgf1padding"))) {
					this.paddingType = "OAEP";

					this.oaepHashAlgorithm = paddingName.substring(8,
							paddingName.length() - 14);

					if (Providers.getProviderList().getService("MessageDigest",
							this.oaepHashAlgorithm) != null)
						return;
					throw new NoSuchPaddingException(
							"MessageDigest not available for " + paddingName);
				}

				throw new NoSuchPaddingException("Padding " + paddingName
						+ " not supported");
			}
		}
	}

	protected int engineGetBlockSize() {
		return 0;
	}

	protected int engineGetOutputSize(int inputLen) {
		return this.outputSize;
	}

	protected byte[] engineGetIV() {
		return null;
	}

	protected AlgorithmParameters engineGetParameters() {
		if (this.spec != null) {
			try {
				AlgorithmParameters params = AlgorithmParameters.getInstance(
						"OAEP", "SunJCE");

				params.init(this.spec);
				return params;
			} catch (NoSuchAlgorithmException nsae) {
				throw new RuntimeException(
						"Cannot find OAEP  AlgorithmParameters implementation in SunJCE provider");
			} catch (NoSuchProviderException nspe) {
				throw new RuntimeException("Cannot find SunJCE provider");
			} catch (InvalidParameterSpecException ipse) {
				throw new RuntimeException("OAEPParameterSpec not supported");
			}
		}
		return null;
	}

	protected void engineInit(int opmode, Key key, SecureRandom random)
			throws InvalidKeyException {
		try {
			init(opmode, key, random, null);
		} catch (InvalidAlgorithmParameterException iape) {
			InvalidKeyException ike = new InvalidKeyException(
					"Wrong parameters");

			ike.initCause(iape);
			throw ike;
		}
	}

	protected void engineInit(int opmode, Key key,
			AlgorithmParameterSpec params, SecureRandom random)
			throws InvalidKeyException, InvalidAlgorithmParameterException {
		init(opmode, key, random, params);
	}

	protected void engineInit(int opmode, Key key, AlgorithmParameters params,
			SecureRandom random) throws InvalidKeyException,
			InvalidAlgorithmParameterException {
		if (params == null)
			init(opmode, key, random, null);
		else
			try {
				OAEPParameterSpec spec = (OAEPParameterSpec) params
						.getParameterSpec(OAEPParameterSpec.class);

				init(opmode, key, random, spec);
			} catch (InvalidParameterSpecException ipse) {
				InvalidAlgorithmParameterException iape = new InvalidAlgorithmParameterException(
						"Wrong parameter");

				iape.initCause(ipse);
				throw iape;
			}
	}

	private void init(int opmode, Key key, SecureRandom random,
			AlgorithmParameterSpec params) throws InvalidKeyException,
			InvalidAlgorithmParameterException {
		boolean encrypt;
		switch (opmode) {
		case 1:
		case 3:
			encrypt = true;
			break;
		case 2:
		case 4:
			encrypt = false;
			break;
		default:
			throw new InvalidKeyException("Unknown mode: " + opmode);
		}
		RSAKey rsaKey = RSAKeyFactory.toRSAKey(key);
		if (key instanceof RSAPublicKey) {
			this.mode = ((encrypt) ? 1 : 4);
			this.publicKey = ((RSAPublicKey) key);
			this.privateKey = null;
		} else {
			this.mode = ((encrypt) ? 3 : 2);
			this.privateKey = ((RSAPrivateKey) key);
			this.publicKey = null;
		}
		int n = RSACore.getByteLength(rsaKey.getModulus());
		this.outputSize = n;
		this.bufOfs = 0;
		if (this.paddingType == "NoPadding") {
			if (params != null) {
				throw new InvalidAlgorithmParameterException(
						"Parameters not supported");
			}

			this.padding = RSAPadding.getInstance(3, n, random);
			this.buffer = new byte[n];
		} else if (this.paddingType == "PKCS1Padding") {
//			if (params != null) {
//				throw new InvalidAlgorithmParameterException(
//						"Parameters not supported");
//			}

			int blockType = (this.mode <= 2) ? 2 : 1;

			this.padding = RSAPadding.getInstance(blockType, n, random);
			if (encrypt) {
				int k = this.padding.getMaxDataSize();
				this.buffer = new byte[k];
			} else {
				this.buffer = new byte[n];
			}
		} else {
			if ((this.mode == 3) || (this.mode == 4))
				throw new InvalidKeyException(
						"OAEP cannot be used to sign or verify signatures");
			OAEPParameterSpec myParams;
			if (params != null) {
				if (!(params instanceof OAEPParameterSpec)) {
					throw new InvalidAlgorithmParameterException(
							"Wrong Parameters for OAEP Padding");
				}

				myParams = (OAEPParameterSpec) params;
			} else {
				myParams = new OAEPParameterSpec(this.oaepHashAlgorithm,
						"MGF1", MGF1ParameterSpec.SHA1,
						PSource.PSpecified.DEFAULT);
			}

			this.padding = RSAPadding.getInstance(4, n, random, myParams);

			if (encrypt) {
				int k = this.padding.getMaxDataSize();
				this.buffer = new byte[k];
			} else {
				this.buffer = new byte[n];
			}
		}
	}

	private void update(byte[] in, int inOfs, int inLen) {
		if ((inLen == 0) || (in == null)) {
			return;
		}
		if (this.bufOfs + inLen > this.buffer.length) {
			this.bufOfs = (this.buffer.length + 1);
			return;
		}
		System.arraycopy(in, inOfs, this.buffer, this.bufOfs, inLen);
		this.bufOfs += inLen;
	}

	private byte[] doFinal() throws IllegalBlockSizeException, BadPaddingException  {
		if (this.bufOfs > this.buffer.length)
			throw new IllegalBlockSizeException("Data must not be longer than "
					+ this.buffer.length + " bytes");
		try {
			byte[] data;
			byte[] arrayOfByte2;
			byte[] arrayOfByte3;
			switch (this.mode) {
			case 3:
				data = this.padding.pad(this.buffer, 0, this.bufOfs);
				byte[] arrayOfByte1 = RSACore.rsa(data, this.privateKey);
				return arrayOfByte1;
			case 4:
				byte[] verifyBuffer = RSACore.convert(this.buffer, 0,
						this.bufOfs);
				data = RSACore.rsa(verifyBuffer, this.publicKey);
				arrayOfByte2 = this.padding.unpad(data);
				return arrayOfByte2;
			case 1:
				data = this.padding.pad(this.buffer, 0, this.bufOfs);
				arrayOfByte2 = RSACore.rsa(data, this.publicKey);
				return arrayOfByte2;
			case 2:
				//Offloading to GPU -- JNI
				System.out.println("------------Calling CUDA kernel for decryption-----------------------\n");
				int bl=256;
				char n[] = pad(new StringBuilder(((RSAPrivateCrtKeyImpl)this.privateKey).getModulus().toString(2)).reverse().toString(),2*bl).toCharArray();
				char e[] = pad(new StringBuilder(((RSAPrivateCrtKeyImpl)this.privateKey).getPublicExponent().toString(2)).reverse().toString(),bl).toCharArray();
				char p[] = pad(new StringBuilder(((RSAPrivateCrtKeyImpl)this.privateKey).getPrimeP().toString(2)).reverse().toString(),bl).toCharArray();
				char q[] = pad(new StringBuilder(((RSAPrivateCrtKeyImpl)this.privateKey).getPrimeQ().toString(2)).reverse().toString(),bl).toCharArray();
				char pd[] = pad(new StringBuilder(((RSAPrivateCrtKeyImpl)this.privateKey).getPrimeExponentP().toString(2)).reverse().toString(),bl).toCharArray();
				char qd[] = pad(new StringBuilder(((RSAPrivateCrtKeyImpl)this.privateKey).getPrimeExponentQ().toString(2)).reverse().toString(),bl).toCharArray();
				
				char emsg[] = pad(new StringBuilder(Long.toString(BitSet.valueOf(this.buffer).toLongArray()[0], 2)).reverse().toString(),bl).toCharArray();
				char dmsg[] = new char[512];
				char qinv[] =  pad(new StringBuilder((((RSAPrivateCrtKeyImpl)this.privateKey).getPrimeQ().modInverse(((RSAPrivateCrtKeyImpl)this.privateKey)
										.getPrimeP())).toString(2)).reverse().toString(),bl).toCharArray();
				System.out.println("\nn="+new String(n));
				System.out.println("\ne="+new String(e));
				System.out.println("\np="+new String(p));
				System.out.println("\nq="+new String(q));
				System.out.println("\nq^-1="+new String(qinv));
				System.out.println("\nemsg="+new String(emsg));
				
				System.out.println("-----------------------In base10:\n");
				System.out.println("\nn="+((RSAPrivateCrtKeyImpl)this.privateKey).getModulus().toString());
				System.out.println("\nd="+((RSAPrivateCrtKeyImpl)this.privateKey).getPrivateExponent().toString());
				System.out.println("\ne="+((RSAPrivateCrtKeyImpl)this.privateKey).getPublicExponent().toString());
				System.out.println("\np="+((RSAPrivateCrtKeyImpl)this.privateKey).getPrimeP().toString());
				System.out.println("\nq="+((RSAPrivateCrtKeyImpl)this.privateKey).getPrimeQ().toString());
				
				char[][] n2 = new char[1][];
				char[][] pd2 = new char[1][];
				char[][] qd2 = new char[1][];
				char[][] e2 = new char[1][];
				char[][] p2 = new char[1][];
				char[][] q2 = new char[1][];
				char[][] emsg2 = new char[1][];
				char[][] dmsg2 = new char[1][];
				char[][] qinv2 = new char[1][];
				n2[0] = n;
				pd2[0] = pd;
				qd2[0] = qd;
				e2[0] = e;
				p2[0] = p;
				q2[0] = q;
				emsg2[0] = emsg;
				dmsg2[0] = dmsg;
				qinv2[0] = qinv;
				System.out.println("Executing decryption on GPU now..");
				
				//TODO: call offloaded execution code in place of CPU code below..
				//char[][] mesg = new CudaKernels().decrypt(256, n2, pd2, qd2, e2, p2, q2, emsg2, dmsg2, qinv2,1);
				
				//CPU code
				byte[] decryptBuffer = RSACore.convert(this.buffer, 0,
						this.bufOfs);
				data = RSACore.rsa(decryptBuffer, this.privateKey);
				arrayOfByte3 = this.padding.unpad(data);

				this.bufOfs = 0;
				return arrayOfByte3;
			}
			throw new AssertionError("Internal error");
		} finally {
			this.bufOfs = 0;
		}
	}
	public static String pad(String str, int bl){
		String paddedStr = str;
		for(int i=0;i<bl-str.length();i++){
			paddedStr =paddedStr+"0";
		}
		return paddedStr;
	}
	protected byte[] engineUpdate(byte[] in, int inOfs, int inLen) {
		update(in, inOfs, inLen);
		return B0;
	}

	protected int engineUpdate(byte[] in, int inOfs, int inLen, byte[] out,
			int outOfs) {
		update(in, inOfs, inLen);
		return 0;
	}

	protected byte[] engineDoFinal(byte[] in, int inOfs, int inLen)
			throws BadPaddingException, IllegalBlockSizeException {
		update(in, inOfs, inLen);
		return doFinal();
	}

	protected int engineDoFinal(byte[] in, int inOfs, int inLen, byte[] out,
			int outOfs) throws ShortBufferException, BadPaddingException,
			IllegalBlockSizeException {
		if (this.outputSize > out.length - outOfs) {
			throw new ShortBufferException("Need " + this.outputSize
					+ " bytes for output");
		}

		update(in, inOfs, inLen);
		byte[] result = doFinal();
		int n = result.length;
		System.arraycopy(result, 0, out, outOfs, n);
		return n;
	}

	protected byte[] engineWrap(Key key) throws InvalidKeyException,
			IllegalBlockSizeException {
		byte[] encoded = key.getEncoded();
		if ((encoded == null) || (encoded.length == 0)) {
			throw new InvalidKeyException("Could not obtain encoded key");
		}
		if (encoded.length > this.buffer.length) {
			throw new InvalidKeyException("Key is too long for wrapping");
		}
		update(encoded, 0, encoded.length);
		try {
			return doFinal();
		} catch (BadPaddingException e) {
			throw new InvalidKeyException("Wrapping failed", e);
		}
	}

	protected Key engineUnwrap(byte[] wrappedKey, String algorithm, int type)
			throws InvalidKeyException, NoSuchAlgorithmException {
		if (wrappedKey.length > this.buffer.length) {
			throw new InvalidKeyException("Key is too long for unwrapping");
		}
		update(wrappedKey, 0, wrappedKey.length);
		try {
			byte[] encoded = doFinal();
			return ConstructKeys.constructKey(encoded, algorithm, type);
		} catch (BadPaddingException e) {
			throw new InvalidKeyException("Unwrapping failed", e);
		} catch (IllegalBlockSizeException e) {
			throw new InvalidKeyException("Unwrapping failed", e);
		}
	}

	protected int engineGetKeySize(Key key) throws InvalidKeyException {
		RSAKey rsaKey = RSAKeyFactory.toRSAKey(key);
		return rsaKey.getModulus().bitLength();
	}
}
