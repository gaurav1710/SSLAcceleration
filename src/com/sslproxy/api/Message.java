package com.sslproxy.api;

/**
 * Response - decrypted message
 *
 */
public class Message {
	String message;
	public Message(String message){
		this.message = message;
	}
	public String getMessage(){
		return message;
	}
	@Override
	public String toString() {
		return "Message [message=" + message + "]";
	}
}
