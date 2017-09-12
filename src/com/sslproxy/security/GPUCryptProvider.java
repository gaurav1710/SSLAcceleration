package com.sslproxy.security;

import java.security.Provider;

public class GPUCryptProvider extends Provider{

	public static final String name = "GPUCrypt";
	public static final double version = 1.0;
	public static final String info = "Cryptography offloading to GPU";
	/**
	 * 
	 */
	private static final long serialVersionUID = 7053319084438473594L;

	public GPUCryptProvider(){
		super(name, version, info);
		doRegister();
	}
	
	private void doRegister() {	
		put("Cipher.RSA", "com.sslproxy.security.RSAGPUCipher");
	}

}
