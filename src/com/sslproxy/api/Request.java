package com.sslproxy.api;

/**
 * Decryption Request
 */
public class Request {
	String cipherText;
	String p;
	String q;
	String d;
	String e;
	
	public String getCipherText() {
		return cipherText;
	}
	public void setCipherText(String cipherText) {
		this.cipherText = cipherText;
	}
	public String getP() {
		return p;
	}
	public void setP(String p) {
		this.p = p;
	}
	public String getQ() {
		return q;
	}
	public void setQ(String q) {
		this.q = q;
	}
	public String getD() {
		return d;
	}
	public void setD(String d) {
		this.d = d;
	}
	public String getE() {
		return e;
	}
	public void setE(String e) {
		this.e = e;
	}
	@Override
	public String toString() {
		return "Request [cipherText=" + cipherText + ", p=" + p + ", q=" + q
				+ ", d=" + d + ", e=" + e + "]";
	}
}
