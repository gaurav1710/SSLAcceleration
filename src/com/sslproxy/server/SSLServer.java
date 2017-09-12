package com.sslproxy.server;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.ServerSocket;
import java.security.KeyManagementException;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.UnrecoverableKeyException;
import java.security.cert.CertificateException;

import javax.net.ssl.KeyManagerFactory;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLServerSocketFactory;
import javax.net.ssl.SSLSocket;
import javax.net.ssl.TrustManagerFactory;

import com.sslproxy.property.Property;

public class SSLServer {
	private ServerSocket serverSocket;

	public void start() {
		try {
			SSLContext sslContext = SSLContext.getInstance("TLSv1");
			KeyStore keyStore = KeyStore.getInstance("JKS");
			FileInputStream fis = new FileInputStream(Property.keyStoreFilePath);
			keyStore.load(fis, Property.keyStorePass.toCharArray());

			KeyManagerFactory keyManagerFactory = KeyManagerFactory
					.getInstance(KeyManagerFactory.getDefaultAlgorithm());
			keyManagerFactory.init(keyStore,
					Property.keyStorePass.toCharArray());

			TrustManagerFactory trustManagerFactory = TrustManagerFactory
					.getInstance(KeyManagerFactory.getDefaultAlgorithm());
			trustManagerFactory.init(keyStore);

			sslContext.init(keyManagerFactory.getKeyManagers(),
					trustManagerFactory.getTrustManagers(), null);
			SSLServerSocketFactory sslServerSocketFactory = (SSLServerSocketFactory) sslContext
					.getServerSocketFactory();
			serverSocket = sslServerSocketFactory
					.createServerSocket(Property.port + 10000);

		} catch (IOException | NoSuchAlgorithmException e) {
			e.printStackTrace();
		} catch (KeyStoreException e) {
			e.printStackTrace();
		} catch (CertificateException e) {
			e.printStackTrace();
		} catch (KeyManagementException e) {
			e.printStackTrace();
		} catch (UnrecoverableKeyException e) {
			e.printStackTrace();
		}
	}

	public void stop() {
		try {
			serverSocket.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void loop() {
		while (true) {
			try {
				SSLSocket socket = (SSLSocket) serverSocket.accept();
				//Enable only specific cipher suites (TLS_RSA_WITH_AES_128_CBC_SHA)
				socket.setEnabledCipherSuites(Property.enabledCipherSuites);
				socket.setSoTimeout(Integer.MAX_VALUE);
				socket.startHandshake();
				process(socket);

			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	private void process(SSLSocket sock) throws IOException {
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(
				sock.getOutputStream()));
		BufferedReader br = new BufferedReader(new InputStreamReader(
				sock.getInputStream()));
		String responseText = null;
		while ((responseText = br.readLine()) != null) {
			if (responseText.trim().equals(""))
				break;
		}
		responseText = "<HTML><H1>SSL Using CUDA</H1></HTML>";
		bw.write(responseText, 0, responseText.length());
		bw.newLine();
		bw.flush();
		//close streams and socket
		br.close();
		bw.close();
		sock.close();
	}

	public static void main(String[] args) {
		if(args.length>0){
			Property.keyStoreFilePath = args[0];
		}
		SSLServer server = new SSLServer();
		server.start();
		server.loop();
	}

}
