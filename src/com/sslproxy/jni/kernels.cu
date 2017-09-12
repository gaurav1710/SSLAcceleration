#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define K 300	

/*
 * MP parallel multiplication implimentation
 * Calculates m*n
 */
__global__ void pmul(int *m, int *n, int k,int *inter_buf, int *carry_buf)
{
	int x = threadIdx.x;
	int i;
	//digit of one operand that will be taken care of by this thread..
	int ndig = n[x];
	int product = 1;
	for(i=0;i<k;i++){
		product = m[i]*ndig;
		//low word
		int lword = product%10;
		//high word
		int hword = product/10;
		//store and add to partial results with carry handling
		int ind = x+i;
		//lword 
		int carry = (inter_buf[ind]+lword)/10;
		inter_buf[ind] = (inter_buf[ind]+lword)%10;
		carry_buf[ind+1] += carry;
		//hword 
		carry = (inter_buf[ind+1]+hword)/10;
		inter_buf[ind+1] = (inter_buf[ind+1]+hword)%10;
					
		carry_buf[ind+2] += carry;
		__syncthreads();//Only required here..
	}		
}

/*
 * Parallel addition of large numbers
 * m+n
 */
__global__ void padd(int *m, int *n, int k,int *res, int batch_size){
	int x = threadIdx.x*batch_size;
	int i;
	for(i=x;i<x+batch_size && i<k;i++){
		int sum = m[i] + n[i]; 
		res[i] = sum%10;
		res[i+1] += sum/10;
		__syncthreads();	
	}		
}

/*
 * Parallel division of large numbers by 2
 * m/2
 */
__global__ void pdiv(int *m, int k,int *res){
	int x = threadIdx.x;
	int carry = x==k-1?0:m[x+1]%2;
	res[x] = (10*carry+m[x])/2;
}

__device__ void dadd_carry(int sum[], int carry[], int k){

	for(int i=0;i<k;i++){
		sum[i] += carry[i];
		if((i+1)!=k){
			carry[i+1] += sum[i]/10;
		}
		sum[i] %= 10;		
	}
}

/*
 * Parallel subtraction of large numbers
 * m-n
 */
__global__ void psub(int *m, int *n, int k,int *res){
	int x = threadIdx.x;
	int carry = x==k-1?0:m[x+1]%2;
	res[x] = (10*carry+m[x])/2;
}

__global__ void montgomery(int *x, int *y, int k, int *m, int *p){
	int ind = threadIdx.x;
	int *xi = new int[k];
	for(int i=0;i<k;i++){
		xi[i] = ind==i?x[i]:0; 
	}
	int *s1 = new int[2*k];
	int *c1 = new int[2*k];
	
	int *res = new int[2*k];
	pmul<<<1,k>>>(x,y,k,s1,c1);
        cudaDeviceSynchronize();
        dadd_carry(s1,c1,k);
	padd<<<1,k/3>>>(p,s1,2*k,res,k/3);
	
        cudaDeviceSynchronize();	
	p = res;
	for(int i=1;i<k;i++){
		xi[i] =0; 
	}	
	xi[0] = p[0];
	pmul<<<1,k>>>(xi,m,k,s1,c1);


        cudaDeviceSynchronize();
	dadd_carry(s1,c1,2*k);
	padd<<<1,k/3>>>(p,s1,2*k,res,k/3);

        cudaDeviceSynchronize();
	p = res;
   	pdiv<<<1,k>>>(p,2*k,res);
	p=res;
        cudaDeviceSynchronize();
	//compare p and m
	int p_grt_m = 1;
	for(int i=k-1;i>=0;i--){
		if(p[i]!=m[i]){
			if(p[i]<m[i]){
				p_grt_m = 0;		
			}else{
				p_grt_m = 1;
			}
		}
	}
	if(p_grt_m == 1){
		//TODO:pad m k digits  to 0 and pass 2k digits
		psub<<<1,k>>>(p, m, 2*k, res);
        	cudaDeviceSynchronize();
		p = res;	
	}
	
}

__global__ void crt_m1(){
	
}

__global__ void crt_m2(){

}

void print_array(int a[], int l){
	for(int i=l-1;i>=0;i--){
		printf("%d",a[i]);
	}
	printf("\n");

}

void add_carry(int sum[], int carry[], int k){

	for(int i=0;i<k;i++){
		sum[i] += carry[i];
		if((i+1)!=k){
			carry[i+1] += sum[i]/10;
		}
		sum[i] %= 10;		
	}
}


void test_pmul(){
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

void test_padd(){
int m[K], n[K],s[K+1];
	int *dm, *dn,*ds;
	printf("Starting..\n\n");
	cudaMalloc((void **) &dm, K*sizeof(int));
	
	cudaMalloc((void **) &dn, K*sizeof(int));
	
	cudaMalloc((void **) &ds, (K+1)*sizeof(int));
	for(int i=0;i<K;i++){
		m[i] = n[i] = 1;
	}
	for(int i=0;i<(K+1);i++){
		s[i] = 0;
	}
	cudaMemcpy(dm, m, K*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dn, n, K*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(ds,s, (K+1)*sizeof(int),cudaMemcpyHostToDevice);
	
	dim3 grid(1, 1);//1 block
	dim3 block(K,1);//K threads
	
	padd<<<grid,block>>>(dm, dn, K, ds,1);
	
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
	
	cudaDeviceSynchronize();
	cudaMemcpy(n, dn, (K+1)*sizeof(int),cudaMemcpyDeviceToHost);
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
		m[i] = 2;n[i] = 1;
	}
	for(int i=0;i<(K+1);i++){
		s[i] = 0;
	}
	cudaMemcpy(dm, m, K*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dn, n, K*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(ds,s, K*sizeof(int),cudaMemcpyHostToDevice);
	
	dim3 grid(1, 1);//1 block
	dim3 block(K,1);//K threads
	
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

extern void execute_test(){
	test_pmul();
	test_padd();
	test_pdiv();
	test_psub();
}
