package com.sslproxy.security;

import java.security.Provider;
import java.security.Security;

public class TestProvider {

	public static void main(String[] args) {
		Provider[] p = Security.getProviders();
		for(Provider pr:p){
		    System.out.println("Provider provider name is " + pr.getName());
		    System.out.println("Provider provider version # is " + pr.getVersion());
		    System.out.println("Provider provider info is " + pr.getInfo());
		}
	}

}
