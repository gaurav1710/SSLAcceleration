void test_padd(){
int m[K], n[K],s[K+1];
	int *dm, *dn,*ds;
	printf("Starting..\n\n");
	cudaMalloc((void **) &dm, K*sizeof(int));
	
	cudaMalloc((void **) &dn, K*sizeof(int));
	
	cudaMalloc((void **) &ds, (K+1)*sizeof(int));
	for(int i=0;i<K;i++){
		m[i] =9; n[i] = 0;
	}
	n[0]=1;
	for(int i=0;i<(K+1);i++){
		s[i] = 0;
	}
	cudaMemcpy(dm, m, K*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dn, n, K*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(ds,s, (K+1)*sizeof(int),cudaMemcpyHostToDevice);
	
	dim3 grid(1, 1);//1 block
	dim3 block(1,1);//K threads
	
	padd<<<grid,block>>>(dm, dn, K, ds,K);
	
	cudaDeviceSynchronize();
	cudaMemcpy(s, ds, (K+1)*sizeof(int),cudaMemcpyDeviceToHost);
	printf("Printing Results..\n");
	print_array(m,K);
	printf("+ \n");
	print_array(n,K);
	
	printf("=\n");
	
	printf("Net Sum: ");
	print_array(s,(K+1));
	cudaFree(dm);
	cudaFree(dn);
	cudaFree(ds);
}

void test_pdiv(){
	int m[K], n[K];
	int *dm, *dn;
	printf("Starting..\n\n");
	cudaMalloc((void **) &dm, K*sizeof(int));
	
	cudaMalloc((void **) &dn, K*sizeof(int));
	
	for(int i=0;i<K;i++){
		m[i] = i%10;
		n[i]=0;
	}
	cudaMemcpy(dm, m, K*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dn, n, K*sizeof(int),cudaMemcpyHostToDevice);
	
	dim3 grid(1, 1);//1 block
	dim3 block(K,1);//K threads
	
	pdiv<<<grid,block>>>(dm, K,dn);
	
	cudaError_t err = cudaDeviceSynchronize();
  	if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
	cudaMemcpy(n, dn, (K)*sizeof(int),cudaMemcpyDeviceToHost);
	printf("Printing Results..\n");
	print_array(m,K);
	printf("/2 \n");
	
	printf("=\n");
	
	printf("Res: ");
	print_array(n,(K));
	cudaFree(dm);
	cudaFree(dn);
	
}
void test_psub(){
int m[K], n[K],s[K];
	int *dm, *dn,*ds;
	printf("Starting..\n\n");
	cudaMalloc((void **) &dm, K*sizeof(int));
	
	cudaMalloc((void **) &dn, K*sizeof(int));
	
	cudaMalloc((void **) &ds, (K)*sizeof(int));
	for(int i=0;i<K;i++){
		m[i] = 0;n[i] = 0;
	}
	m[K-1]=1;
	n[0]=1;
	for(int i=0;i<(K+1);i++){
		s[i] = 0;
	}
	cudaMemcpy(dm, m, K*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dn, n, K*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(ds,s, K*sizeof(int),cudaMemcpyHostToDevice);
	
	dim3 grid(1, 1);//1 block
	dim3 block(1,1);//K threads
	
	psub<<<grid,block>>>(dm, dn, K, ds);
	
	cudaDeviceSynchronize();
	cudaMemcpy(s, ds, (K)*sizeof(int),cudaMemcpyDeviceToHost);
	printf("Printing Results..\n");
	print_array(m,K);
	printf("- \n");
	print_array(n,K);
	
	printf("=\n");
	
	printf("Res: ");
	print_array(s,K);
	cudaFree(dm);
	cudaFree(dn);
	cudaFree(ds);
}

void test_montgomery(){
	int m[K], n[K],ib[2*K],cb[2*K];
	int *dm, *dn,*dib,*dcb;
	printf("Starting..\n\n");
	cudaMalloc((void **) &dm, K*sizeof(int));
	
	cudaMalloc((void **) &dn, K*sizeof(int));
	
	cudaMalloc((void **) &dib, 2*K*sizeof(int));
	cudaMalloc((void **) &dcb, 2*K*sizeof(int));
	m[0] = 9;
	m[1] = 4;
	m[2] = 6;
	n[0] = 7;
	n[1] = 2;
	n[2] = 6;
	for(int i=0;i<K;i++){
		m[i] = n[i] = 1;
	}
	for(int i=0;i<2*K;i++){
		ib[i] = 0;cb[i]=0;
	}
	cudaMemcpy(dm, m, K*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dn, n, K*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dib, ib, 2*K*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dcb, cb, 2*K*sizeof(int),cudaMemcpyHostToDevice);
				
	dim3 grid(1, 1);//1 block
	dim3 block(K,1);//K threads
	
	pmul<<<grid,block>>>(dm, dn, K, dib,dcb);
	
	cudaDeviceSynchronize();
	cudaMemcpy(ib, dib, 2*K*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(cb, dcb, 2*K*sizeof(int),cudaMemcpyDeviceToHost);
	
	printf("Printing Results..\n");
	print_array(m,K);
	printf("X \n");
	print_array(n,K);
	
	printf("=\n");
	printf("Intermediate Buffer: ");			
	print_array(ib,2*K);
	printf("Carry Buffer: ");
	print_array(cb,2*K);
	add_carry(ib,cb,2*K);
	printf("Net Product: ");
	print_array(ib,2*K);
	cudaFree(dm);
	cudaFree(dn);
	cudaFree(dib);
	cudaFree(dcb);
}
