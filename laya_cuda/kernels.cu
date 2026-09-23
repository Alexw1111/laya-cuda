#include <cuda_fp16.h>
#define INFINITY __int_as_float(0x7f800000)

__device__ float reduce_sum(float x) {
    __shared__ float sums[32];
    int lane=threadIdx.x%32, warp=threadIdx.x/32;
    for(int d=16;d;d/=2) x+=__shfl_down_sync(0xffffffff,x,d);
    if(!lane) sums[warp]=x;
    __syncthreads();
    x=threadIdx.x<blockDim.x/32?sums[lane]:0;
    if(!warp) for(int d=16;d;d/=2) x+=__shfl_down_sync(0xffffffff,x,d);
    if(!threadIdx.x) sums[0]=x;
    __syncthreads();
    return sums[0];
}
extern "C" __global__ void embed(const int* ids,const float* weights,float* out,int n,int w) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n*w) out[i]=weights[(long long)ids[i/w]*w+i%w];
}
extern "C" __global__ void layer_norm(float* residual,const half* branch,const float* gamma,
                                 const float* beta,void* target,int w,float eps,int out32) {
    int row=blockIdx.x, t=threadIdx.x;
    if(!gamma) {
        for(int j=t;j<w;j+=blockDim.x) {
            int i=row*w+j;
            float v=residual[i]+(branch?__half2float(branch[i]):0);
            residual[i]=v;
            if(out32) ((float*)target)[i]=v; else ((half*)target)[i]=__float2half(v);
        }
        return;
    }
    float s=0;
    for(int j=t;j<w;j+=blockDim.x) {
        int i=row*w+j;
        float v=residual[i]+(branch?__half2float(branch[i]):0);
        residual[i]=v; s+=v;
    }
    float mean=reduce_sum(s)/w, sq=0;
    for(int j=t;j<w;j+=blockDim.x) {
        float v=residual[row*w+j]-mean;
        sq+=v*v;
    }
    float inv=rsqrtf(reduce_sum(sq)/w+eps);
    for(int j=t;j<w;j+=blockDim.x) {
        int i=row*w+j;
        float v=(residual[i]-mean)*inv*gamma[j]+(beta?beta[j]:0);
        if(out32) ((float*)target)[i]=v; else ((half*)target)[i]=__float2half(v);
    }
}
extern "C" __global__ void bias(const float* x,const half* b,void* out,int n,int w,int out32) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float v=x[i]+__half2float(b[i%w]);
    if(out32) ((float*)out)[i]=v; else ((half*)out)[i]=__float2half(v);
}
extern "C" __global__ void add(float* x,const half* y,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=__half2float(y[i]);
}
extern "C" __global__ void activate(const half* x,half* y,int n,int w,int mode) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    int pos=mode==2?(i/w)*(2*w)+i%w:i;
    float v=__half2float(x[pos]);
    if(mode==1) v=fmaxf(v,0); else v=0.5f*v*(1+erff(v*0.7071067811865475f));
    half a=__float2half(v);
    y[i]=mode==2?__hmul(a,x[pos+w]):a;
}
extern "C" __global__ void layout(const half* x,half* qkv,const float* cosines,const float* sines,
                                   const int* positions,int m,int h) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, w=h*64;
    if(i>=m*3*w) return;
    int d=i%64, head=(i/64)%h, part=(i/w)%3, token=i/(3*w), pos=positions[token];
    half v=x[i];
    if(part<2 && cosines) {
        float other=__half2float(x[i+(d<32?32:-32)]); if(d<32) other=-other;
        v=__float2half(__half2float(v)*cosines[pos*32+d%32]+other*sines[pos*32+d%32]);
    }
    // Q and K are [h][m][64]. V interleaves token pairs, [h][m/2][64][2], so the attention
    // MMA loads two consecutive keys of one dimension as a single 32-bit word.
    long long base=((long long)part*h+head)*m*64;
    qkv[base+(part<2?token*64+d:(token/2*64+d)*2+token%2)]=v;
}
__device__ __forceinline__ void mma16816(float* c,const unsigned* a,unsigned b0,unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
__device__ __forceinline__ unsigned pack2(float x,float y) {
    half2 v=__floats2half2_rn(x,y); return *(unsigned*)&v;
}
// One warp per 16 packed queries and head, on tensor cores. Rows start at multiples of 16 tokens;
// rows[token] is -1 past the last row. One pass over the keys keeps each query's running maximum
// and rescales its accumulated exp(s-max)*V (online softmax). Probabilities enter the FP16 MMA as
// high+low parts, which keeps FP32-level probability precision. Padding tokens return zeros.
extern "C" __global__ void attention(const half* qkv,half* out,const int* rows,const int* starts,
                                     const int* lengths,int m,int h,int window) {
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/32, lane=threadIdx.x%32, g=lane/4, t=lane%4;
    if(warp>=h*(m/16)) return;
    int head=warp/(m/16), tile=warp%(m/16)*16, row=rows[tile];
    int first=row<0?tile:starts[row], valid=row<0?0:lengths[row], q0=tile-first;
    half* o0=out+((long long)(tile+g)*h+head)*64+t*2;
    half* o1=o0+8LL*h*64;
    if(q0>=valid) {for(int n=0;n<8;n++) {*(unsigned*)(o0+n*8)=0;*(unsigned*)(o1+n*8)=0;} return;}
    long long plane=(long long)h*m*64;
    const half *Q=qkv+head*(long long)m*64+first*64LL, *K=Q+plane, *V=Q+2*plane;
    int r0=q0+g, r1=r0+8;
    int start=window<0?0:max(0,q0-window)/16*16;
    int end=(min(valid,window<0?valid:q0+16+window)+15)/16*16;
    unsigned qa[4][4];
    for(int c=0;c<4;c++) {
        const half* p=Q+(q0+g)*64+c*16+t*2;
        qa[c][0]=*(const unsigned*)p; qa[c][1]=*(const unsigned*)(p+512);
        qa[c][2]=*(const unsigned*)(p+8); qa[c][3]=*(const unsigned*)(p+520);
    }
    float m0=-INFINITY,m1=-INFINITY,sum0=0,sum1=0,acc[8][4]={};
    for(int kb=start;kb<end;kb+=16) {
        float s[2][4]={};
        for(int u=0;u<2;u++) for(int c=0;c<4;c++) {
            const half* p=K+(kb+u*8+g)*64+c*16+t*2;
            mma16816(s[u],qa[c],*(const unsigned*)p,*(const unsigned*)(p+8));
        }
        for(int u=0;u<2;u++) for(int e=0;e<4;e++) {
            int key=kb+u*8+t*2+e%2, row=e<2?r0:r1;
            bool keep=key<valid && (window<0 || abs(key-row)<=window);
            s[u][e]=keep?s[u][e]*0.125f:-INFINITY;
        }
        // Raise each row's running maximum, then rescale what the row has accumulated.
        float n0=fmaxf(fmaxf(s[0][0],s[0][1]),fmaxf(s[1][0],s[1][1])), n1=fmaxf(fmaxf(s[0][2],s[0][3]),fmaxf(s[1][2],s[1][3]));
        for(int d=1;d<4;d*=2) {n0=fmaxf(n0,__shfl_xor_sync(0xffffffff,n0,d)); n1=fmaxf(n1,__shfl_xor_sync(0xffffffff,n1,d));}
        n0=fmaxf(m0,n0); n1=fmaxf(m1,n1);
        float c0=m0==-INFINITY?0:expf(m0-n0), c1=m1==-INFINITY?0:expf(m1-n1);
        m0=n0; m1=n1; sum0*=c0; sum1*=c1;
        for(int n=0;n<8;n++) {acc[n][0]*=c0; acc[n][1]*=c0; acc[n][2]*=c1; acc[n][3]*=c1;}
        unsigned hi[4],lo[4];
        for(int u=0;u<2;u++) for(int e=0;e<4;e+=2) {
            float m=e<2?m0:m1;
            float x=s[u][e]==-INFINITY?0:expf(s[u][e]-m), y=s[u][e+1]==-INFINITY?0:expf(s[u][e+1]-m);
            if(e<2) sum0+=x+y; else sum1+=x+y;
            half2 top=__floats2half2_rn(x,y);
            hi[u*2+e/2]=*(unsigned*)&top;
            lo[u*2+e/2]=pack2(x-__low2float(top),y-__high2float(top));
        }
        for(int n=0;n<8;n++) {
            const half* p=V+((kb/2+t)*64+n*8+g)*2;
            unsigned b0=*(const unsigned*)p, b1=*(const unsigned*)(p+512);
            mma16816(acc[n],hi,b0,b1); mma16816(acc[n],lo,b0,b1);
        }
    }
    for(int d=1;d<4;d*=2) {sum0+=__shfl_xor_sync(0xffffffff,sum0,d); sum1+=__shfl_xor_sync(0xffffffff,sum1,d);}
    float i0=r0<valid?1/sum0:0, i1=r1<valid?1/sum1:0;
    for(int n=0;n<8;n++) {
        *(unsigned*)(o0+n*8)=pack2(acc[n][0]*i0,acc[n][1]*i0);
        *(unsigned*)(o1+n*8)=pack2(acc[n][2]*i1,acc[n][3]*i1);
    }
}
extern "C" __global__ void type_add(float* x,const float* types,const int* kinds,int m,int w) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*w) x[i]+=types[kinds[i/w]*w+i%w];
}
extern "C" __global__ void gather(const float* x,const int* indices,float* y,int rows,int w) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<rows*w) y[i]=x[indices[i/w]*w+i%w];
}
extern "C" __global__ void gather_half(const half* x,const int* indices,half* y,int rows,int w) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<rows*w) y[i]=x[indices[i/w]*w+i%w];
}
extern "C" __global__ void features(const float* selected,const float* logits,const int* counts,
                                     half* output,int b,int k,int w) {
    int row=blockIdx.x,t=threadIdx.x,count=counts[row];
    for(int d=t;d<w;d+=blockDim.x) output[row*(w+4)+d]=__float2half(selected[row*(k+1)*w+d]);
    if(!t) {
        float top=-INFINITY,total=0,p1=0,p2=0,entropy=0;
        for(int j=0;j<count;j++) top=fmaxf(top,logits[row*(k+1)+j+1]);
        for(int j=0;j<count;j++) total+=expf(logits[row*(k+1)+j+1]-top);
        for(int j=0;j<count;j++) {
            float p=expf(logits[row*(k+1)+j+1]-top)/total;
            entropy-=p*logf(fmaxf(p,1e-9f)); if(p>p1) {p2=p1;p1=p;} else p2=fmaxf(p2,p);
        }
        output[row*(w+4)+w]=__float2half(p1);output[row*(w+4)+w+1]=__float2half(p1-p2);
        output[row*(w+4)+w+2]=__float2half(entropy/logf((float)max(count,2)));
        output[row*(w+4)+w+3]=__float2half(max(count,2)/255.f);
    }
}
