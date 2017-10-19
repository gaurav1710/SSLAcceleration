package com.sslproxy.api;

import java.util.List;

public interface SSLProxyService {
	List<Message> decrypt(List<Request> requests, int bitlength);
}
