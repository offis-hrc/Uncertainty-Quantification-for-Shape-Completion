
#include <stdio.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>



__global__ void NmDistanceKernel(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i){
	const int batch=512;
	__shared__ float buf[batch*3];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int k2=0;k2<m;k2+=batch){
			int end_k=min(m,k2+batch)-k2;
			for (int j=threadIdx.x;j<end_k*3;j+=blockDim.x){
				buf[j]=xyz2[(i*m+k2)*3+j];
			}
			__syncthreads();
			for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
				float x1=xyz[(i*n+j)*4+0];
				float y1=xyz[(i*n+j)*4+1];
				float z1=xyz[(i*n+j)*4+2];
                float best_c=xyz[(i*n+j)*4+3];
				int best_i=0;
				float best=0;
				float best_x1=0;
				float best_y1=0;
				float best_z1=0;
				float best_x2=0;
				float best_y2=0;
				float best_z2=0;
				//float best_c=0;
				int end_ka=end_k-(end_k&3);
				if (end_ka==batch){
					for (int k=0;k<batch;k+=4){
						{
							float x2=buf[k*3+0]-x1;
							float y2=buf[k*3+1]-y1;
							float z2=buf[k*3+2]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (k==0 || d<best){
								best=d;
								best_x1 = x1;
								best_y1 = y1;
								best_z1 = z1;
								best_x2 = buf[k*3+0];
								best_y2 = buf[k*3+1];
								best_z2 = buf[k*3+2];
								//best_c  = buf[k*4+3];
								//best = (best_x2 - best_x1)*(best_x2 - best_x1) +  (best_y2 - best_y1)*(best_y2 - best_y1) + (best_z2 - best_z1)*(best_z2 - best_z1);
								best_i=k+k2;
							}
						}
						{
							float x2=buf[k*3+3]-x1;
							float y2=buf[k*3+4]-y1;
							float z2=buf[k*3+5]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_x1 = x1;
								best_y1 = y1;
								best_z1 = z1;
								best_x2 = buf[k*3+3];
								best_y2 = buf[k*3+4];
								best_z2 = buf[k*3+5];
								//best_c  = buf[k*3+7];
								//best = (best_x2 - best_x1)*(best_x2 - best_x1) +  (best_y2 - best_y1)*(best_y2 - best_y1) + (best_z2 - best_z1)*(best_z2 - best_z1);
								best_i=k+k2+1;
							}
						}
						{
							float x2=buf[k*3+6]-x1;
							float y2=buf[k*3+7]-y1;
							float z2=buf[k*3+8]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_x1 = x1;
								best_y1 = y1;
								best_z1 = z1;
								best_x2 = buf[k*3+6];
								best_y2 = buf[k*3+7];
								best_z2 = buf[k*3+8];
								//best_c  = buf[k*4+11];
								//best = (best_x2 - best_x1)*(best_x2 - best_x1) +  (best_y2 - best_y1)*(best_y2 - best_y1) + (best_z2 - best_z1)*(best_z2 - best_z1);
								best_i=k+k2+2;
							}
						}
						{
							float x2=buf[k*3+9]-x1;
							float y2=buf[k*3+10]-y1;
							float z2=buf[k*3+11]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_x1 = x1;
								best_y1 = y1;
								best_z1 = z1;
								best_x2 = buf[k*3+9];
								best_y2 = buf[k*3+10];
								best_z2 = buf[k*3+11];
								//best_c  = buf[k*4+15];
								//best = (best_x2 - best_x1)*(best_x2 - best_x1) +  (best_y2 - best_y1)*(best_y2 - best_y1) + (best_z2 - best_z1)*(best_z2 - best_z1);
								best_i=k+k2+3;
							}
						}
					}
				}else{
					for (int k=0;k<end_ka;k+=4){
						{
							float x2=buf[k*3+0]-x1;
							float y2=buf[k*3+1]-y1;
							float z2=buf[k*3+2]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (k==0 || d<best){
								best=d;
								best_x1 = x1;
								best_y1 = y1;
								best_z1 = z1;
								best_x2 = buf[k*3+0];
								best_y2 = buf[k*3+1];
								best_z2 = buf[k*3+2];
								//best_c  = buf[k*4+3];
								best_i=k+k2;
							}
						}
						{
							float x2=buf[k*3+3]-x1;
							float y2=buf[k*3+4]-y1;
							float z2=buf[k*3+5]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_x1 = x1;
								best_y1 = y1;
								best_z1 = z1;
								best_x2 = buf[k*3+3];
								best_y2 = buf[k*3+4];
								best_z2 = buf[k*3+5];
								//best_c  = buf[k*4+7];
								best_i=k+k2+1;
							}
						}
						{
							float x2=buf[k*3+6]-x1;
							float y2=buf[k*3+7]-y1;
							float z2=buf[k*3+8]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_x1 = x1;
								best_y1 = y1;
								best_z1 = z1;
								best_x2 = buf[k*3+6];
								best_y2 = buf[k*3+7];
								best_z2 = buf[k*3+8];
								//best_c  = buf[k*3+11];
								best_i=k+k2+2;
							}
						}
						{
							float x2=buf[k*3+9]-x1;
							float y2=buf[k*3+10]-y1;
							float z2=buf[k*3+11]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_x1 = x1;
								best_y1 = y1;
								best_z1 = z1;
								best_x2 = buf[k*3+9];
								best_y2 = buf[k*3+10];
								best_z2 = buf[k*3+11];
								//best_c  = buf[k*4+15];
								best_i=k+k2+3;
							}
						}
					}
				}
				for (int k=end_ka;k<end_k;k++){
					float x2=buf[k*3+0]-x1;
					float y2=buf[k*3+1]-y1;
					float z2=buf[k*3+2]-z1;
					float d=x2*x2+y2*y2+z2*z2;
					if (k==0 || d<best){
						best=d;
						best_x1 = x1;
						best_y1 = y1;
						best_z1 = z1;
						best_x2 = buf[k*3+0];
						best_y2 = buf[k*3+1];     
						best_z2 = buf[k*3+2];						
						//best_c  = buf[k*3+3];
						best_i=k+k2;
					}
				}
				if (k2==0 || result[(i*n+j)]>best){
					//best = (best_x1 - (best_c*best_x2 + (1-best_c)*best_x1))*(best_x1 - (best_c*best_x2 + (1-best_c)*best_x1)) + (best_y1 - (best_c*best_y2 + (1-best_c)*best_y1))*(best_y1 - (best_c*best_y2 + (1-best_c)*best_y1)) + (best_z1 - (best_c*best_z2 + (1-best_c)*best_z1))*(best_z1 - (best_c*best_z2 + (1-best_c)*best_z1));
					best = (best_x2 - ((1-best_c)*best_x1 + best_c*(best_x2)))*(best_x2 - ((1-best_c)*best_x1 + best_c*(best_x2))) + (best_x2 - ((1-best_c)*best_x1 + best_c*(best_x2)))*(best_x2 - ((1-best_c)*best_x1 + best_c*(best_x2))) + (best_x2 - ((1-best_c)*best_x1 + best_c*(best_x2)))*(best_x2 - ((1-best_c)*best_x1 + best_c*(best_x2))) +  0.003*(0-best_c)*(0-best_c);
					
					//scalar_t d=(x2-(x1 + c*(x2-x1)))*(x2-(x1 + c*(x2-x1))) + (y2-(y1 + c*(y2-y1)))*(y2-(y1 + c*(y2-y1))) + (z2-(z1 + c*(z2-z1)))*(z2-(z1 + c*(z2-z1)));				
					result[(i*n+j)]=best;
					result_i[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
	}
}

__global__ void NmDistanceKernelNormal(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i){
	const int batch=512;
	__shared__ float buf[batch*4];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int k2=0;k2<m;k2+=batch){
			int end_k=min(m,k2+batch)-k2;
			for (int j=threadIdx.x;j<end_k*4;j+=blockDim.x){
				buf[j]=xyz2[(i*m+k2)*4+j];
			}
			__syncthreads();
			for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
				float x1=xyz[(i*n+j)*3+0];
				float y1=xyz[(i*n+j)*3+1];
				float z1=xyz[(i*n+j)*3+2];
				int best_i=0;
				float best=0;
				float best_x1=0;
				float best_y1=0;
				float best_z1=0;
				float best_x2=0;
				float best_y2=0;
				float best_z2=0;
				float best_c=0;
				
				
				int end_ka=end_k-(end_k&3);
				if (end_ka==batch){
					for (int k=0;k<batch;k+=4){
						{
							float x2=buf[k*4+0]-x1;
							float y2=buf[k*4+1]-y1;
							float z2=buf[k*4+2]-z1;
							
							float d=x2*x2+y2*y2+z2*z2;
							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
								best_x1 = x1;
								best_y1 = y1;
								best_z1 = z1;
								best_x2 = buf[k*4+0];
								best_y2 = buf[k*4+1];
								best_z2 = buf[k*4+2];
								best_c  = buf[k*4+3];
                                
							}
						}
						{
							float x2=buf[k*4+4]-x1;
							float y2=buf[k*4+5]-y1;
							float z2=buf[k*4+6]-z1;
						
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+1;
								best_x1 = x1;
								best_y1 = y1;
								best_z1 = z1;
								best_x2 = buf[k*4+4];
								best_y2 = buf[k*4+5];
								best_z2 = buf[k*4+6];
								best_c  = buf[k*4+7];
                               
							}
						}
						{
							float x2=buf[k*4+8]-x1;
							float y2=buf[k*4+9]-y1;
							float z2=buf[k*4+10]-z1;
							
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+2;
								best_x1 = x1;
								best_y1 = y1;
								best_z1 = z1;
								best_x2 = buf[k*4+8];
								best_y2 = buf[k*4+9];
								best_z2 = buf[k*4+10];
								best_c  = buf[k*4+11];
                                
							}
						}
						{
							float x2=buf[k*4+12]-x1;
							float y2=buf[k*4+13]-y1;
							float z2=buf[k*4+14]-z1;
							
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+3;
								best_x1 = x1;
								best_y1 = y1;
								best_z1 = z1;
								best_x2 = buf[k*4+12];
								best_y2 = buf[k*4+13];
								best_z2 = buf[k*4+14];
								best_c  = buf[k*4+15];
                                
							}
						}
					}
				}else{
					for (int k=0;k<end_ka;k+=4){
						{
							

                            float x2=buf[k*4+0]-x1;
							float y2=buf[k*4+1]-y1;
							float z2=buf[k*4+2]-z1;
							
							float d=x2*x2+y2*y2+z2*z2;

							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
								best_x1 = x1;
								best_y1 = y1;
								best_z1 = z1;
								best_x2 = buf[k*4+0];
								best_y2 = buf[k*4+1];
								best_z2 = buf[k*4+2];
								best_c  = buf[k*4+3];
                                
							}
						}
						{
							float x2=buf[k*4+4]-x1;
							float y2=buf[k*4+5]-y1;
							float z2=buf[k*4+6]-z1;
							
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+1;
								best_x1 = x1;
								best_y1 = y1;
								best_z1 = z1;
								best_x2 = buf[k*4+4];
								best_y2 = buf[k*4+5];
								best_z2 = buf[k*4+6];
								best_c  = buf[k*4+7];
                                
							}
						}
						{
							float x2=buf[k*4+8]-x1;
							float y2=buf[k*4+9]-y1;
							float z2=buf[k*4+10]-z1;
							
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+2;
								best_x1 = x1;
								best_y1 = y1;
								best_z1 = z1;
								best_x2 = buf[k*4+8];
								best_y2 = buf[k*4+9];
								best_z2 = buf[k*4+10];
								best_c  = buf[k*4+11];
                                
							}
						}
						{
							float x2=buf[k*4+12]-x1;
							float y2=buf[k*4+13]-y1;
							float z2=buf[k*4+14]-z1;
							
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+3;
								best_x1 = x1;
								best_y1 = y1;
								best_z1 = z1;
								best_x2 = buf[k*4+12];
								best_y2 = buf[k*4+13];
								best_z2 = buf[k*4+14];
								best_c  = buf[k*4+15];
                                
							}
						}
					}
				}
				for (int k=end_ka;k<end_k;k++){
					float x2=buf[k*4+0]-x1;
					float y2=buf[k*4+1]-y1;
					float z2=buf[k*4+2]-z1;
					
					float d=x2*x2+y2*y2+z2*z2;
					if (k==0 || d<best){
						best=d;
						best_i=k+k2;
						best_x1 = x1;
						best_y1 = y1;
						best_z1 = z1;
						best_x2 = buf[k*4+0];
						best_y2 = buf[k*4+1];     
						best_z2 = buf[k*4+2];						
						best_c  = buf[k*4+3];
                        
					}
				}
				if (k2==0 || result[(i*n+j)]>best){
					//best = (best_x1 - (best_c*best_x2 + (1-best_c)*best_x1))*(best_x1 - (best_c*best_x2 + (1-best_c)*best_x1)) + (best_y1 - (best_c*best_y2 + (1-best_c)*best_y1))*(best_y1 - (best_c*best_y2 + (1-best_c)*best_y1)) + (best_z1 - (best_c*best_z2 + (1-best_c)*best_z1))*(best_z1 - (best_c*best_z2 + (1-best_c)*best_z1));
					//best = best + 1.*(best_c*(best_x1-best_x2)*(best_x1-best_x2) + best_d*(best_y1-best_y2)*(best_y1-best_y2) + best_e*(best_z1-best_z2)*(best_z1-best_z2));//(best_x1 - (best_x2 + best_c*(best_x1-best_x2)))*(best_x1 - (best_x2 + best_c*(best_x1-best_x2))) + (best_y1 - (best_y2 + best_c*(best_y1-best_y2)))*(best_y1 - (best_y2 + best_c*(best_y1-best_y2))) + (best_z1 - (best_z2 + best_c*(best_z1-best_z2)))*(best_z1 - (best_z2 + best_c*(best_z1-best_z2)));
					//best = best + 0.33*(best_c*(best_x1-best_x2)*(best_x1-best_x2) + best_d*(best_y1-best_y2)*(best_y1-best_y2) + best_e*(best_z1-best_z2)*(best_z1-best_z2)) + 0.1*(1-best_c)*(1-best_c) + 0.1*(1-best_d)*(1-best_d) + 0.1*(1-best_e)*(1-best_e);// + (best_y1-best_y2)*(best_y1-best_y2) + (best_z1-best_z2)*(best_z1-best_z2));//(best_x1 - (best_x2 + best_c*(best_x1-best_x2)))*(best_x1 - (best_x2 + best_c*(best_x1-best_x2))) + (best_y1 - (best_y2 + best_c*(best_y1-best_y2)))*(best_y1 - (best_y2 + best_c*(best_y1-best_y2))) + (best_z1 - (best_z2 + best_c*(best_z1-best_z2)))*(best_z1 - (best_z2 + best_c*(best_z1-best_z2)));
					//best = best + best_c*0.33*((best_x1-best_x2)*(best_x1-best_x2) + (best_y1-best_y2)*(best_y1-best_y2) + (best_z1-best_z2)*(best_z1-best_z2)) + 0.02*(1-best_c)*(1-best_c) ;//+ 0.1*(1-best_d)*(1-best_d) + 0.1*(1-best_e)*(1-best_e);// + (best_y1-best_y2)*(best_y1-best_y2) + (best_z1-best_z2)*(best_z1-best_z2));//(best_x1 - (best_x2 + best_c*(best_x1-best_x2)))*(best_x1 - (best_x2 + best_c*(best_x1-best_x2))) + (best_y1 - (best_y2 + best_c*(best_y1-best_y2)))*(best_y1 - (best_y2 + best_c*(best_y1-best_y2))) + (best_z1 - (best_z2 + best_c*(best_z1-best_z2)))*(best_z1 - (best_z2 + best_c*(best_z1-best_z2)));
                    best = (best_x1 - ((1-best_c)*best_x2 + best_c*(best_x1)))*(best_x1 - ((1-best_c)*best_x2 + best_c*(best_x1))) + (best_y1 - ((1-best_c)*best_y2 + best_c*(best_y1)))*(best_y1 - ((1-best_c)*best_y2 + best_c*(best_y1))) + (best_z1 - ((1-best_c)*best_z2 + best_c*(best_z1)))*(best_z1 - ((1-best_c)*best_z2 + best_c*(best_z1))) + 0.002*(0-best_c)*(0-best_c);

					result[(i*n+j)]=best;
					result_i[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
	}
}


//xyz1 : predicted
//xyz2 : GT
// int chamfer_cuda_forward(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i, cudaStream_t stream){
int chamfer_cuda_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist1, at::Tensor dist2, at::Tensor idx1, at::Tensor idx2){

	const auto batch_size = xyz1.size(0);
	const auto n = xyz1.size(1); //num_points point cloud A (pred?)
	const auto m = xyz2.size(1); //num_points point cloud B (GT?)

	NmDistanceKernel<<<dim3(32,16,1),512>>>(batch_size, n, xyz1.data<float>(), m, xyz2.data<float>(), dist1.data<float>(), idx1.data<int>());
	//float a = 5;
	//float b = 10;
	//float b = 15;
	NmDistanceKernelNormal<<<dim3(32,16,1),512>>>(batch_size, m, xyz2.data<float>(), n, xyz1.data<float>(), dist2.data<float>(), idx2.data<int>());

	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd updateOutput: %s\n", cudaGetErrorString(err));
	    //THError("aborting");n,xyz1.data<float>(
	    return 0;
	  }
	  return 1;

}


__global__ void NmDistanceGradKernelNormal(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,float * grad_xyz1,float * grad_xyz2){
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
			float x1=xyz1[(i*n+j)*3+0];
			float y1=xyz1[(i*n+j)*3+1];
			float z1=xyz1[(i*n+j)*3+2];
			//float c =xyz1[(i*n+j)*4+3];
			int j2=idx1[i*n+j];
			float x2=xyz2[(i*m+j2)*4+0];
			float y2=xyz2[(i*m+j2)*4+1];
			float z2=xyz2[(i*m+j2)*4+2];
			float c =xyz2[(i*m+j2)*4+3];
			
			float g=grad_dist1[i*n+j]*2;
			//continue here 16.7
			// atomicAdd(&(grad_xyz1[(i*n+j)*3+0]),g*((x1-x2) + 0.001*c*(x1-x2)/3));
			// atomicAdd(&(grad_xyz1[(i*n+j)*3+1]),g*((y1-y2) + 0.001*d*(y1-y2)/3));
			// atomicAdd(&(grad_xyz1[(i*n+j)*3+2]),g*((z1-z2) + 0.001*e*(z1-z2)/3));
			// //atomicAdd(&(grad_xyz1[(i*n+j)*4+3]),0*g*(z1-z2));
			// atomicAdd(&(grad_xyz2[(i*m+j2)*6+0]),-(g*((x1-x2) + 0.001*c*(x1-x2)/3)));
			// atomicAdd(&(grad_xyz2[(i*m+j2)*6+1]),-(g*((y1-y2) + 0.001*d*(y1-y2)/3)));
			// atomicAdd(&(grad_xyz2[(i*m+j2)*6+2]),-(g*((z1-z2) + 0.001*e*(z1-z2)/3)));
			// atomicAdd(&(grad_xyz2[(i*m+j2)*6+3]),(0.001*(g/6)*((x1-x2)*(x1-x2))));//+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2))));//((z1-z2)*(z1-z2) + (y1-y2)*(y1-y2) + (x1-x2)*(x1-x2))));
			// atomicAdd(&(grad_xyz2[(i*m+j2)*6+4]),(0.001*(g/6)*((y1-y2)*(y1-y2))));
			// atomicAdd(&(grad_xyz2[(i*m+j2)*6+5]),(0.001*(g/6)*((z1-z2)*(z1-z2))));


			//works axis-wise
			// atomicAdd(&(grad_xyz1[(i*n+j)*3+0]),g*((x1-x2) + c*(x1-x2)));
			// atomicAdd(&(grad_xyz1[(i*n+j)*3+1]),g*((y1-y2) + d*(y1-y2)));
			// atomicAdd(&(grad_xyz1[(i*n+j)*3+2]),g*((z1-z2) + e*(z1-z2)));
			// //atomicAdd(&(grad_xyz1[(i*n+j)*4+3]),0*g*(z1-z2));
			// atomicAdd(&(grad_xyz2[(i*m+j2)*6+0]),-(g*((x1-x2) + c*(x1-x2))));
			// atomicAdd(&(grad_xyz2[(i*m+j2)*6+1]),-(g*((y1-y2) + d*(y1-y2))));
			// atomicAdd(&(grad_xyz2[(i*m+j2)*6+2]),-(g*((z1-z2) + e*(z1-z2))));
			// atomicAdd(&(grad_xyz2[(i*m+j2)*6+3]),((g/6)*(2*(x1-x2)*(x1-x2) - 0.2*(1-c))));//+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2))));//((z1-z2)*(z1-z2) + (y1-y2)*(y1-y2) + (x1-x2)*(x1-x2))));
			// atomicAdd(&(grad_xyz2[(i*m+j2)*6+4]),(0*(g/6)*(2*(y1-y2)*(y1-y2) - 0.2*(1-d))));U having to wait for the GPU to finish and transfer the loss might have an impact. (My GPUs are busy, 
			// atomicAdd(&(grad_xyz2[(i*m+j2)*6+5]),(0*(g/6)*(2*(z1-z2)*(z1-z2) - 0.2*(1-e))));

			atomicAdd(&(grad_xyz1[(i*n+j)*3+0]),-g*((c - 1)*(c-1)*(x2 - x1)));
			atomicAdd(&(grad_xyz1[(i*n+j)*3+1]),-g*((c - 1)*(c-1)*(y2 - y1)));
			atomicAdd(&(grad_xyz1[(i*n+j)*3+2]),-g*((c - 1)*(c-1)*(z2 - z1)));
			//atomicAdd(&(grad_xyz1[(i*n+j)*4+3]),0*g*(z1-z2));
			atomicAdd(&(grad_xyz2[(i*m+j2)*4+0]),g*((c - 1)*(c-1)*(x2 - x1)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*4+1]),g*((c - 1)*(c-1)*(y2 - y1)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*4+2]),g*((c - 1)*(c-1)*(z2 - z1)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*4+3]),((g)*(((c-1)*(x1*x1 + x2*x2 + y1*y1 + y2*y2 + z1*z1 +z2*z2 - 2*x1*x2 - 2*y1*y2 - 2*z1*z2)) - 0.004*(0-c))));

            //+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2))));//((z1-z2)*(z1-z2) + (y1-y2)*(y1-y2) + (x1-x2)*(x1-x2))));
			//atomicAdd(&(grad_xyz2[(i*m+j2)*6+4]),(0*(g/6)*(2*(y1-y2)*(y1-y2) - 0.2*(1-d))));
			//atomicAdd(&(grad_xyz2[(i*m+j2)*6+5]),(0*(g/6)*(2*(z1-z2)*(z1-z2) - 0.2*(1-e))));








			// atomicAdd(&(grad_xyz1[(i*n+j)*3+0]),g*(c-1)*(c-1)*(x1-x2));
			// atomicAdd(&(grad_xyz1[(i*n+j)*3+1]),g*(c-1)*(c-1)*(y1-y2));
			// atomicAdd(&(grad_xyz1[(i*n+j)*3+2]),g*(c-1)*(c-1)*(z1-z2));
			// //atomicAdd(&(grad_xyz1[(i*n+j)*4+3]),0*g*(z1-z2));
			// atomicAdd(&(grad_xyz2[(i*m+j2)*4+0]),-g*(c-1)*(c-1)*(x1-x2));
			// atomicAdd(&(grad_xyz2[(i*m+j2)*4+1]),-g*(c-1)*(c-1)*(y1-y2));
			// atomicAdd(&(grad_xyz2[(i*m+j2)*4+2]),-g*(c-1)*(c-1)*(z1-z2));
			// atomicAdd(&(grad_xyz2[(i*m+j2)*4+3]),0.01*g*(c-1)*(x1*x1 - 2*x1*x2 + y1*y1 -2*y1*y2 + z1*z1 - 2*z1*z2 + x2*x2 + y2*y2 + z2*z2));
		}
	}
}


__global__ void NmDistanceGradKernel(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,float * grad_xyz1,float * grad_xyz2){
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
			float x1=xyz1[(i*n+j)*4+0];
			float y1=xyz1[(i*n+j)*4+1];
			float z1=xyz1[(i*n+j)*4+2];
			float  c=xyz1[(i*n+j)*4+3];
			int j2=idx1[i*n+j];
			float x2=xyz2[(i*m+j2)*3+0];
			float y2=xyz2[(i*m+j2)*3+1];
			float z2=xyz2[(i*m+j2)*3+2];
			//float  c=xyz2[(i*m+j2)*4+3];
			float g=grad_dist1[i*n+j]*2;
			//continue here//continue here
			
			atomicAdd(&(grad_xyz1[(i*n+j)*4+0]),-g*((c - 1)*(c-1)*(x2 - x1)));
			atomicAdd(&(grad_xyz1[(i*n+j)*4+1]),-g*((c - 1)*(c-1)*(y2 - y1)));
			atomicAdd(&(grad_xyz1[(i*n+j)*4+2]),-g*((c - 1)*(c-1)*(z2 - z1)));
			atomicAdd(&(grad_xyz1[(i*n+j)*4+3]),((g)*(((c-1)*(x1*x1 + x2*x2 + y1*y1 + y2*y2 + z1*z1 +z2*z2 - 2*x1*x2 - 2*y1*y2 - 2*z1*z2)) - 0.006*(0-c))));
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+0]),g*((c - 1)*(c-1)*(x2 - x1)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+1]),g*((c - 1)*(c-1)*(y2 - y1)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+2]),g*((c - 1)*(c-1)*(z2 - z1)));
			//atomicAdd(&(grad_xyz2[(i*m+j2)*4+3]),0*g*c*(x1*x1 - 2*x1*x2 + y1*y1 -2*y1*y2 + z1*z1 - 2*z1*z2 + x2*x2 + y2*y2 + z2*z2));

		}
	}
}

// int chamfer_cuda_backward(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,const float * grad_dist2,const int * idx2,float * grad_xyz1,float * grad_xyz2, cudaStream_t stream){
int chamfer_cuda_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz1, at::Tensor gradxyz2, at::Tensor graddist1, at::Tensor graddist2, at::Tensor idx1, at::Tensor idx2){
	// cudaMemset(grad_xyz1,0,b*n*3*4);
	// cudaMemset(grad_xyz2,0,b*m*3*4);
	
	const auto batch_size = xyz1.size(0);
	const auto n = xyz1.size(1); //num_points point cloud A (is pred)
	const auto m = xyz2.size(1); //num_points point cloud B

	NmDistanceGradKernel<<<dim3(1,16,1),256>>>(batch_size,n,xyz1.data<float>(),m,xyz2.data<float>(),graddist1.data<float>(),idx1.data<int>(),gradxyz1.data<float>(),gradxyz2.data<float>());
	NmDistanceGradKernelNormal<<<dim3(1,16,1),256>>>(batch_size,m,xyz2.data<float>(),n,xyz1.data<float>(),graddist2.data<float>(),idx2.data<int>(),gradxyz2.data<float>(),gradxyz1.data<float>());
	
	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd get grad: %s\n", cudaGetErrorString(err));
	    //THError("aborting");
	    return 0;
	  }
	  return 1;
	
}
