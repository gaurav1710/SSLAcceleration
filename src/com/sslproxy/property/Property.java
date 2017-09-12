package com.sslproxy.property;

public class Property {
	// Proxy server properties
	public static String hostname = "127.0.0.1";
	public static int port = 443;
	// certificate,server public key etc.
	public static String keyStoreFilePath = "../ssl.keystore";
	public static String keyStorePass = "sslmtp";
	public static String[] enabledCipherSuites = { "TLS_RSA_WITH_AES_128_CBC_SHA" };

	// Backend server properties
	public static String backendHostname = "127.0.0.1";
	public static int bakendPort = 8080;
	public static boolean backendOn = true;

	// GPU Offload switch
	public static boolean gpuOffload = true;
}
