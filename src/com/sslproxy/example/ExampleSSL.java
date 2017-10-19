package com.sslproxy.example;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.sslproxy.api.Message;
import com.sslproxy.api.Request;
import com.sslproxy.api.SSLProxyService;
import com.sslproxy.api.impl.SSLProxyServiceImpl;
import com.sslproxy.jni.CudaKernels;

/**
 * Example/Test the SSL offloading code
 * 
 * @author gaurav(mcs132556)
 */
public class ExampleSSL {

	public static void main(String[] args) {
		SSLProxyService sslSvc = new SSLProxyServiceImpl();
		int batchSize = 2;
		int bitLength = 1024;
		List<Request> requests = new ArrayList<Request>();
		for(int i=0;i<batchSize;i++){
			Request req = new Request();
			BigInteger bp = BigInteger.probablePrime(bitLength/2, new Random());
			BigInteger bq = BigInteger.probablePrime(bitLength/2, new Random());
			BigInteger phi = bp.subtract(BigInteger.ONE).multiply(bq.subtract(BigInteger.ONE));
			BigInteger bd = new BigInteger("65537").modInverse(phi);
			String e = "65537";
			String ctext = BigInteger.probablePrime(bitLength, new Random()).toString();
			req.setCipherText(ctext);
			req.setD(bd.toString());
			req.setE(e);
			req.setP(bp.toString());
			req.setQ(bq.toString());
			requests.add(req);
		}
		List<Message> messages = sslSvc.decrypt(requests, bitLength);
		for(int i=0;i<messages.size();i++){
			System.out.println("INFO: Decrypted message("+i+"):"+messages.get(i).toString());
		}
	} 
}