#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "helper_cuda.h"
#include "kernel.h"

// =============================================
// Constant Memory
__constant__ int imageD_const_[2];
__constant__ int imageH_const_[2];
__constant__ int imageW_const_;
// =============================================

// =============================================
#define MAX_KERNEL_SIZE_CU 100
#define COLUMNS_HALO_STEPS 1
#define COLUMNS_RESULT_STEPS 8
#define MAX_C(i,j) ( (i)<(j) ? (j):(i) )

const float GaussTruncate_cu = 4.0;

__device__ __constant__ float kernel_c[MAX_KERNEL_SIZE_CU];
// =============================================

//Maps to a single instruction on G8x / G9x / G10x
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b) {
    return (a % b != 0) ? (a - a % b + b) : a;
}

extern "C" cudaEvent_t * create_cudaEvent_array(const int num) {
    cudaEvent_t* event_ = (cudaEvent_t*)malloc(num * sizeof(cudaEvent_t));
    for (int i = 0; i < num; i++) checkCudaErrors(cudaEventCreateWithFlags(&(event_[i]), cudaEventBlockingSync | cudaEventDisableTiming));
    return event_;
}

extern "C" cudaStream_t * create_cudaStream_array(const int num) {
    cudaStream_t* stream_ = (cudaStream_t*)malloc(num * sizeof(cudaStream_t));
    for (int i = 0; i < num; i++) checkCudaErrors(cudaStreamCreate(&(stream_[i])));
    return stream_;
}

extern "C" cudaArray** allocate_cudaArray_array_(const int num, const int imageH, const int imageW, const int imageD) {
    /* Allocate texture memory with largest chunk size
    */
    cudaArray** cudaArray_array = new cudaArray* [num];
    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();
    if (imageD <= 0) for (int i = 0; i < num; i++) checkCudaErrors(cudaMallocArray(&cudaArray_array[i], &floatTex, imageW, imageH));
    else {
        cudaExtent extent = make_cudaExtent(imageW, imageH, imageD);
        for (int i = 0; i < num; i++) checkCudaErrors(cudaMalloc3DArray(&cudaArray_array[i], &floatTex, extent));
    }
    return cudaArray_array;
}

extern "C" cudaMemcpy3DParms setup_3DCopy_params(
    cudaArray * texSrc, float* input_host, const int imageH,
    const int imageW, const int imageD, const int offset
) {
    // Note that for 3D texture, there is no cudaMemcpyTo3DArray. Have to use cudaMemcpy3D instead
    cudaMemcpy3DParms p = { 0 };
    p.srcPtr   = make_cudaPitchedPtr((void*)&input_host[offset], imageW * sizeof(float), imageW, imageH);
    p.dstArray = texSrc;
    p.extent   = make_cudaExtent(imageW, imageH, imageD);
    p.kind     = cudaMemcpyHostToDevice;
    return p;
}

extern "C" void free_cudaArray_array(cudaArray** cudaArray_array, const int num) {
    for (int i = 0; i < num; i++) checkCudaErrors(cudaFreeArray(cudaArray_array[i]));
    delete cudaArray_array;
}

extern "C" void free_cudaEvent_array(cudaEvent_t * events, const int num) {
    for (int i = 0; i < num; i++) checkCudaErrors(cudaEventDestroy(events[i]));
    delete events;
}

extern "C" void free_cudaStream_array(cudaStream_t * streams, const int num) {
    for (int i = 0; i < num; i++) checkCudaErrors(cudaStreamDestroy(streams[i]));
    delete streams;
}

// Transfer to constant memory
extern "C" void setBasicParams_const_(std::vector<std::pair<int, int>>& section, int imageW, int imageH) {
    /* The function takes in padded image size and automatically subtracts by 2
    */
    int imageW_adjusted = imageW - 2;
    int imageH_adjusted = imageH - 2;
    const int chunk_num = section.size();
    if (chunk_num < 1) { printf("Invalid chunk_num\n"); exit(1); }
    std::vector<int> h(chunk_num), h_(2);
    for (int i = 0; i < chunk_num; i++) h[i] = section[i].second - section[i].first - 1;
    h_[0] = h[0];
    h_[1] = h[chunk_num - 1];
    cudaMemcpyToSymbol(imageW_const_, &imageW_adjusted, sizeof(int));
    if (imageH_adjusted > 0) {
        // 3D case
        std::vector<int> t = { imageH_adjusted, 0 };
        cudaMemcpyToSymbol(imageH_const_, &t[0], 2 * sizeof(int));
        cudaMemcpyToSymbol(imageD_const_, &h_[0], 2 * sizeof(int));
        t.clear();
    }
    else cudaMemcpyToSymbol(imageH_const_, &h_[0], 2 * sizeof(int));
    h.clear();
    h_.clear();
}

extern "C" cudaTextureObject_t* create_cudaTextureObject_array_(cudaArray** cudaArray_array, const int num) {
    cudaTextureObject_t* texSrc_array = new cudaTextureObject_t[num];

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.readMode = cudaReadModeElementType;

    cudaResourceDesc             texResrc;
    memset(&texResrc, 0, sizeof(cudaResourceDesc));
    texResrc.resType           = cudaResourceTypeArray;

    for (int i = 0; i < num; i++) {
        texResrc.res.array.array = cudaArray_array[i];
        checkCudaErrors(cudaCreateTextureObject(&texSrc_array[i], &texResrc, &texDesc, NULL));
    }
    return texSrc_array;
}

__device__ int binary_search(float* d, int size, float v) {
    /*
    * @d: unique array sorted in asending order
    * @size: size of the array
    * @v: target value to be found
    * Assume v in d
    */
    int left = 0, right = size - 1;
    while (v != d[(left + right) / 2]) {
        if (v < d[(left + right) / 2]) right = (left + right) / 2 - 1;
        else left = (left + right) / 2 + 1;
    }
    return (left + right) / 2;
}

// SG: Readable 2D version of code.
__global__ void ECC_kernel_v20(
    cudaTextureObject_t texSrc,
    int binNum,
    int* VCEC_device,
    float* ascend_unique_arr_device_,
    int chunk_type_id
)
{   // V2.0 is much more readable than V3.0, start from here
    /*
    SG: binNum is the len/size of the LOCAL VCEC (by local we mean the local VCEC for a block). It is not 255 by default. 
    Since they handle floating point numbers, the size of the VCEC is the number of unique floating point values.
    Thus the maximum size of the GLOBAL VCEC is H*W (in the case that all values are unique).
    The maximum size of the LOCAL VCEC is the number of unique values handled by the block (collection of threads).

    The VCEC has size 2*binNum ; The first binNum values track changes in EC; the remaining binNum contain the float intensity values in ascending order (eg: 1.23, 3.47, 8.96 etc). There is a 1:1 correspondence between hist_local_[i] & hist_local_[i+binNum] : basically hist_local_[i] contains EC changes for the intensity value stored at hist_local_[i+binNum]. They set the intensity value at hist_local_[i+binNum] using the ascend_unique_arr_device_ function.

    Each thread initializes one location in the local VCEC to 0. If there are more binNum than there are threads, then one thread initializes multiple locations to 0 (each location is blockDim.x*blockDim.y apart). The multiple locations are given by the block_pos update line. So basically the currently executing thread sets block_pos locations in the VCEC to 0. 

    blockDim : contains grid size of each block (how many threads in x direction and how many in y direction)
    threadIdx : contains the x,y location of a thread within a block
    */
    extern __shared__ float hist_local_[];  // SG: the VCEC in shared memory (local to a block)
    int block_pos = blockDim.x * threadIdx.y + threadIdx.x;
    while (block_pos < binNum) {
        hist_local_[block_pos] = 0;
        hist_local_[block_pos + binNum] = ascend_unique_arr_device_[block_pos];
        block_pos = block_pos + blockDim.x * blockDim.y;
    }
    __syncthreads(); // SG: making sure entire initialization of local VCEC is done (waiting for all threads to finish executing upto this line). Only then does each thread start execution of below code.

    /* 
    SG: ix and iy are variables declared in this thread-code and so are stored in registers. If register spills/overflows, global (aka slowest) memory will be used and so we should be careful not to declare too man variables.
    Since there is 1:1 correspondence between thread location and the pixel coordinate it works on, ix & iy are the pixel coordinates.

    IMAD declared and defined above. It does product followed by addition. 
    */
    const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x) + 1;
    const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y) + 1;
    if (ix < imageW_const_ + 1 && iy < imageH_const_[chunk_type_id] + 1) {

        // SG: In some cases it is < while others <= (similarly, >, >=). This is carefully written to break ties. The rule is that lexicographically lower position is preferred.
        float change = 1; // SG: Each pixel introduces the top-dim cell (in 2D it's the square while in 3D it's the cube). So change has +1 by default.
        float c = tex2D<float>(texSrc, ix, iy); // SG: extracting float pixel/intensity value from texture memory ; stored in register
        float t = tex2D<float>(texSrc, ix, iy - 1); // SG: extracting top neighbour intensity value from texture memory ; stored in register
        float b = tex2D<float>(texSrc, ix, iy + 1); // SG: bottom
        float l = tex2D<float>(texSrc, ix - 1, iy); // SG: left
        float r = tex2D<float>(texSrc, ix + 1, iy); // SG: right

        // Deciding introduced vertices
        change += (c < l&& c < t&& c < tex2D<float>(texSrc, ix - 1, iy - 1)); // SG: diagonal neighbours not stored in register, are accessed on-the-fly
        change += (c < t&& c <= r && c < tex2D<float>(texSrc, ix + 1, iy - 1));
        change += (c < l&& c <= b && c <= tex2D<float>(texSrc, ix - 1, iy + 1));
        change += (c <= b && c <= r && c <= tex2D<float>(texSrc, ix + 1, iy + 1));

        // Deciding introduced edges
        change -= (c < t);
        change -= (c < l);
        change -= (c <= r);
        change -= (c <= b);

        /*
        SG: To update the EC change value in the local VCEC, we first use the intensity value c and find it in the hist_local_[binNum] onwards array. Find the idx where c is using binary search. Then use this idx in the hist_local_[0] onwards and update the change there.

        atomicAdd used to prevent race conditions
        */
        atomicAdd(&hist_local_[binary_search(&hist_local_[binNum], binNum, c)], change); 
    }
    __syncthreads(); // SG: make sure all threads update their local VCEC

     // SG: Start writing local VCEC change to global VCEC (VCEC_device) ; using atomicAdd to prevent race conditions
    if (blockDim.x * threadIdx.y + threadIdx.x < binNum)
        atomicAdd(&VCEC_device[blockDim.x * threadIdx.y + threadIdx.x], (int)hist_local_[blockDim.x * threadIdx.y + threadIdx.x]);
}

// SG: does same work as ECC_kernel_v20 function, but is less readable and more optimized.
__global__ void ECC_kernel_v30(
    cudaTextureObject_t texSrc,
    int binNum,
    int* VCEC_device,
    float* ascend_unique_arr_device_,
    int chunk_type_id
)
{
    extern __shared__ float hist_local_[]; 

    int block_pos = blockDim.x * threadIdx.y + threadIdx.x;
    while (block_pos < binNum) {
        hist_local_[block_pos] = 0;
        hist_local_[block_pos + binNum] = ascend_unique_arr_device_[block_pos];
        block_pos = block_pos + blockDim.x * blockDim.y;
    }
    __syncthreads(); 

    const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x) + 1;
    const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y) + 1;
    if (ix < imageW_const_ + 1 && iy < imageH_const_[chunk_type_id] + 1) {

        float change = 1; 
        float c = tex2D<float>(texSrc, ix, iy); 
        bool c_l_l = (c < tex2D<float>(texSrc, ix - 1, iy));
        bool c_lq_r = (c <= tex2D<float>(texSrc, ix + 1, iy));
        if (c < tex2D<float>(texSrc, ix, iy - 1)) {
            change -= 1;
            if (c_l_l && c < tex2D<float>(texSrc, ix - 1, iy - 1)) change += 1;
            if (c_lq_r && c < tex2D<float>(texSrc, ix + 1, iy - 1)) change += 1;
        }
        if (c <= tex2D<float>(texSrc, ix, iy + 1)) {
            change -= 1;
            if (c_l_l && c <= tex2D<float>(texSrc, ix - 1, iy + 1)) change += 1;
            if (c_lq_r && c <= tex2D<float>(texSrc, ix + 1, iy + 1)) change += 1;
        }

        change -= c_l_l;
        change -= c_lq_r;
        atomicAdd(&hist_local_[binary_search(&hist_local_[binNum], binNum, c)], change);
    }
    __syncthreads();

    block_pos = blockDim.x * threadIdx.y + threadIdx.x;
    while (block_pos < binNum) {
        atomicAdd(&VCEC_device[block_pos], (int)hist_local_[block_pos]);
        block_pos = block_pos + blockDim.x * blockDim.y;
    } 
}

// SG: 3D version of code
__global__ void ECC_kernel_3D(
    cudaTextureObject_t texSrc,
    int binNum,
    int* VCEC_device,
    float* ascend_unique_arr_device_,
    int chunk_type_id
)
{
    // Declare and initialize local histogram
    extern __shared__ float hist_local_[];
    int block_pos = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;  // SG: Doubt: should it be blockDim.z instead of blockDim.x ? Otherwise, similar logic to the 2D case. Each thread initializes multiple locations to 0 (each location is blockDim.x*blockDim.y*blockDim.z apart) in the hist_local_
    while (block_pos < binNum) {
        hist_local_[block_pos] = 0;
        hist_local_[block_pos + binNum] = ascend_unique_arr_device_[block_pos];
        block_pos = block_pos + blockDim.x * blockDim.y * blockDim.z;
    }
    __syncthreads();  // SG: wait for all threads to finish initialization

    // SG: get coordinate of pixel using blockIdx and threadIdx
    const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x) + 1;
    const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y) + 1;
    const int iz = IMAD(blockDim.z, blockIdx.z, threadIdx.z) + 1;
    if (ix < imageW_const_ + 1 && iy < imageH_const_[0] + 1 && iz < imageD_const_[chunk_type_id] + 1) {
        float change = -1; // SG: Each pixel introduces top-dim cell (cube in 3D). So change is -1 by default.

        // SG: In some cases it is < while others <= (similarly, >, >=). This is carefully written to break ties. The rule is that lexicographically lower position is preferred.
        float ccc = tex3D<float>(texSrc, ix, iy, iz); // SG: center pixel
        // Variables used 9 times
        float ccf = tex3D<float>(texSrc, ix, iy, iz - 1); // SG: z denotes front/back directions
        float ccb = tex3D<float>(texSrc, ix, iy, iz + 1);
        float ctc = tex3D<float>(texSrc, ix, iy - 1, iz); // SG: y denotes top/down directions
        float cdc = tex3D<float>(texSrc, ix, iy + 1, iz);
        float lcc = tex3D<float>(texSrc, ix - 1, iy, iz); // SG: x denotes left/right directions
        float rcc = tex3D<float>(texSrc, ix + 1, iy, iz);
        // Variables used 3 times
        float lcf = tex3D<float>(texSrc, ix - 1, iy, iz - 1);
        float ctf = tex3D<float>(texSrc, ix, iy - 1, iz - 1);
        float rcf = tex3D<float>(texSrc, ix + 1, iy, iz - 1);
        float cdf = tex3D<float>(texSrc, ix, iy + 1, iz - 1);
        float ltc = tex3D<float>(texSrc, ix - 1, iy - 1, iz);
        float rtc = tex3D<float>(texSrc, ix + 1, iy - 1, iz);
        float ldc = tex3D<float>(texSrc, ix - 1, iy + 1, iz);
        float rdc = tex3D<float>(texSrc, ix + 1, iy + 1, iz);
        float lcb = tex3D<float>(texSrc, ix - 1, iy, iz + 1);
        float ctb = tex3D<float>(texSrc, ix, iy - 1, iz + 1);
        float rcb = tex3D<float>(texSrc, ix + 1, iy, iz + 1);
        float cdb = tex3D<float>(texSrc, ix, iy + 1, iz + 1);

        // Below operations have inherent total order enforced  // SG: See Figure 4 in paper to visualize
        // Introduced vertices V (8)
        // SG: Each top-dim cell (cube in 3D) has eight 0-dim cells (vertices). And so, we need to check which of these 8 vertices are introduced by (ix, iy, iz). Furthermore, for each vertex, we need to compare against seven other neighbours which share this vertex with us.
        change += (ccc < tex3D<float>(texSrc, ix - 1, iy - 1, iz - 1) && ccc < ctf&& ccc < lcf&& ccc < ccf&& ccc < ltc&& ccc < ctc&& ccc < lcc);             // top left front vertex
        change += (ccc <= tex3D<float>(texSrc, ix - 1, iy - 1, iz + 1) && ccc <= ctb && ccc <= lcb && ccc <= ccb && ccc < ltc&& ccc < ctc&& ccc < lcc);      // top left back vertex
        change += (ccc <= tex3D<float>(texSrc, ix + 1, iy - 1, iz + 1) && ccc <= rcb && ccc <= ccb && ccc <= ctb && ccc < ctc&& ccc < rtc&& ccc <= rcc);     // top right back vertex
        change += (ccc < tex3D<float>(texSrc, ix + 1, iy - 1, iz - 1) && ccc < rcf&& ccc < ccf&& ccc < ctf&& ccc < ctc&& ccc < rtc&& ccc <= rcc);            // top right front vertex

        change += (ccc < tex3D<float>(texSrc, ix - 1, iy + 1, iz - 1) && ccc < lcf&& ccc < ccf&& ccc < cdf&& ccc < lcc&& ccc <= cdc && ccc <= ldc);          // down left front vertex
        change += (ccc <= tex3D<float>(texSrc, ix - 1, iy + 1, iz + 1) && ccc <= lcb && ccc <= ccb && ccc <= cdb && ccc < lcc&& ccc <= cdc && ccc <= ldc);   // down left back vertex
        change += (ccc <= tex3D<float>(texSrc, ix + 1, iy + 1, iz + 1) && ccc <= cdb && ccc <= ccb && ccc <= rcb && ccc <= rcc && ccc <= rdc && ccc <= cdc); // down right back vertex
        change += (ccc < tex3D<float>(texSrc, ix + 1, iy + 1, iz - 1) && ccc < cdf&& ccc < ccf&& ccc < rcf&& ccc <= rcc && ccc <= rdc && ccc <= cdc);        // down right front vertex

        // Introduced edges E (12)
        // SG: Each top-dim cell (cube in 3D) has twelve 1-dim cells (edges). And so, we need to check which of these 12 edges are introduced by (ix, iy, iz). Furthermore, for each edge, we need to compare against three other neighbours which share this edge with us.
        change -= (ccc < ctf&& ccc < ccf&& ccc < ctc);       // top front edge
        change -= (ccc < ltc&& ccc < ctc&& ccc < lcc);       // top left edge
        change -= (ccc < ctc&& ccc <= ctb && ccc <= ccb);    // top back edge
        change -= (ccc < ctc&& ccc < rtc&& ccc <= rcc);      // top right edge

        change -= (ccc < ccf&& ccc < cdf&& ccc <= cdc);      // down front edge
        change -= (ccc < lcc&& ccc <= ldc && ccc <= cdc);    // down left edge
        change -= (ccc <= cdc && ccc <= ccb && ccc <= cdb);  // down back edge
        change -= (ccc <= rcc && ccc <= rdc && ccc <= cdc);  // down right edge

        change -= (ccc < lcf&& ccc < ccf&& ccc < lcc);       // front left edge
        change -= (ccc < lcc&& ccc <= lcb && ccc <= ccb);    // back left edge
        change -= (ccc <= rcc && ccc <= ccb && ccc <= rcb);  // back right edge
        change -= (ccc < ccf&& ccc < rcf&& ccc <= rcc);      // front right edge

        // Introduced faces F (6)
        // SG: Each top-dim cell (cube in 3D) has six 2-dim cells (square/face). And so, we need to check which of these 6 faces are introduced by (ix, iy, iz). Furthermore, for each face, we need to compare against only one other neighbour which shares this face with us.
        change += (ccc < ccf);   // front face
        change += (ccc < lcc);   // left face
        change += (ccc <= ccb);  // back face
        change += (ccc <= rcc);  // right face
        change += (ccc < ctc);   // top face
        change += (ccc <= cdc);  // down face

        // SG: updating the local VCEC; atomicAdd to prevent race conditions
        atomicAdd(&hist_local_[binary_search(&hist_local_[binNum], binNum, ccc)], change);
    }
    __syncthreads(); // SG: wait for all threads to update local VCEC; then update global VCEC
    if (blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x < binNum)
        atomicAdd(&VCEC_device[blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x], (int)hist_local_[blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x]);
}

__global__ void ECC_kernel_3D_v20(
    cudaTextureObject_t texSrc,
    int binNum,
    int* VCEC_device,
    float* ascend_unique_arr_device_,
    int chunk_type_id
)
{
    // Declare and initialize local histogram
    extern __shared__ float hist_local_[];
    int block_pos = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    while (block_pos < binNum) {
        hist_local_[block_pos] = 0;
        hist_local_[block_pos + binNum] = ascend_unique_arr_device_[block_pos];
        block_pos = block_pos + blockDim.x * blockDim.y * blockDim.z;
    }
    __syncthreads();

    const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x) + 1;
    const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y) + 1;
    const int iz = IMAD(blockDim.z, blockIdx.z, threadIdx.z) + 1;
    if (ix < imageW_const_ + 1 && iy < imageH_const_[0] + 1 && iz < imageD_const_[chunk_type_id] + 1) {
        float change = -1;
        float ccc = tex3D<float>(texSrc, ix, iy, iz);
        // Variables used 9 times
        float ccf = tex3D<float>(texSrc, ix, iy, iz - 1);
        float ccb = tex3D<float>(texSrc, ix, iy, iz + 1);
        float ctc = tex3D<float>(texSrc, ix, iy - 1, iz);
        float cdc = tex3D<float>(texSrc, ix, iy + 1, iz);
        float lcc = tex3D<float>(texSrc, ix - 1, iy, iz);
        float rcc = tex3D<float>(texSrc, ix + 1, iy, iz);

        // Below operations have inherent total order enforced
        // Introduced vertices V (8)
        change += (ccc < tex3D<float>(texSrc, ix - 1, iy - 1, iz - 1) && ccc < tex3D<float>(texSrc, ix, iy - 1, iz - 1) && ccc < tex3D<float>(texSrc, ix - 1, iy, iz - 1) && ccc < ccf&& ccc < tex3D<float>(texSrc, ix - 1, iy - 1, iz) && ccc < ctc&& ccc < lcc);             // top left front vertex
        change += (ccc <= tex3D<float>(texSrc, ix - 1, iy - 1, iz + 1) && ccc <= tex3D<float>(texSrc, ix, iy - 1, iz + 1) && ccc <= tex3D<float>(texSrc, ix - 1, iy, iz + 1) && ccc <= ccb && ccc < tex3D<float>(texSrc, ix - 1, iy - 1, iz) && ccc < ctc&& ccc < lcc);      // top left back vertex
        change += (ccc <= tex3D<float>(texSrc, ix + 1, iy - 1, iz + 1) && ccc <= tex3D<float>(texSrc, ix + 1, iy, iz + 1) && ccc <= ccb && ccc <= tex3D<float>(texSrc, ix, iy - 1, iz + 1) && ccc < ctc&& ccc < tex3D<float>(texSrc, ix + 1, iy - 1, iz) && ccc <= rcc);     // top right back vertex
        change += (ccc < tex3D<float>(texSrc, ix + 1, iy - 1, iz - 1) && ccc < tex3D<float>(texSrc, ix + 1, iy, iz - 1) && ccc < ccf&& ccc < tex3D<float>(texSrc, ix, iy - 1, iz - 1) && ccc < ctc&& ccc < tex3D<float>(texSrc, ix + 1, iy - 1, iz) && ccc <= rcc);            // top right front vertex

        change += (ccc < tex3D<float>(texSrc, ix - 1, iy + 1, iz - 1) && ccc < tex3D<float>(texSrc, ix - 1, iy, iz - 1) && ccc < ccf&& ccc < tex3D<float>(texSrc, ix, iy + 1, iz - 1) && ccc < lcc&& ccc <= cdc && ccc <= tex3D<float>(texSrc, ix - 1, iy + 1, iz));          // down left front vertex
        change += (ccc <= tex3D<float>(texSrc, ix - 1, iy + 1, iz + 1) && ccc <= tex3D<float>(texSrc, ix - 1, iy, iz + 1) && ccc <= ccb && ccc <= tex3D<float>(texSrc, ix, iy + 1, iz + 1) && ccc < lcc&& ccc <= cdc && ccc <= tex3D<float>(texSrc, ix - 1, iy + 1, iz));   // down left back vertex
        change += (ccc <= tex3D<float>(texSrc, ix + 1, iy + 1, iz + 1) && ccc <= tex3D<float>(texSrc, ix, iy + 1, iz + 1) && ccc <= ccb && ccc <= tex3D<float>(texSrc, ix + 1, iy, iz + 1) && ccc <= rcc && ccc <= tex3D<float>(texSrc, ix + 1, iy + 1, iz) && ccc <= cdc); // down right back vertex
        change += (ccc < tex3D<float>(texSrc, ix + 1, iy + 1, iz - 1) && ccc < tex3D<float>(texSrc, ix, iy + 1, iz - 1) && ccc < ccf&& ccc < tex3D<float>(texSrc, ix + 1, iy, iz - 1) && ccc <= rcc && ccc <= tex3D<float>(texSrc, ix + 1, iy + 1, iz) && ccc <= cdc);        // down right front vertex

        // Introduced edges E (12)
        change -= (ccc < tex3D<float>(texSrc, ix, iy - 1, iz - 1) && ccc < ccf&& ccc < ctc);       // top front edge
        change -= (ccc < tex3D<float>(texSrc, ix - 1, iy - 1, iz) && ccc < ctc&& ccc < lcc);       // top left edge
        change -= (ccc < ctc&& ccc <= tex3D<float>(texSrc, ix, iy - 1, iz + 1) && ccc <= ccb);    // top back edge
        change -= (ccc < ctc&& ccc < tex3D<float>(texSrc, ix + 1, iy - 1, iz) && ccc <= rcc);      // top right edge

        change -= (ccc < ccf&& ccc < tex3D<float>(texSrc, ix, iy + 1, iz - 1) && ccc <= cdc);      // down front edge
        change -= (ccc < lcc&& ccc <= tex3D<float>(texSrc, ix - 1, iy + 1, iz) && ccc <= cdc);    // down left edge
        change -= (ccc <= cdc && ccc <= ccb && ccc <= tex3D<float>(texSrc, ix, iy + 1, iz + 1));  // down back edge
        change -= (ccc <= rcc && ccc <= tex3D<float>(texSrc, ix + 1, iy + 1, iz) && ccc <= cdc);  // down right edge

        change -= (ccc < tex3D<float>(texSrc, ix - 1, iy, iz - 1) && ccc < ccf&& ccc < lcc);       // front left edge
        change -= (ccc < lcc&& ccc <= tex3D<float>(texSrc, ix - 1, iy, iz + 1) && ccc <= ccb);    // back left edge
        change -= (ccc <= rcc && ccc <= ccb && ccc <= tex3D<float>(texSrc, ix + 1, iy, iz + 1));  // back right edge
        change -= (ccc < ccf&& ccc < tex3D<float>(texSrc, ix + 1, iy, iz - 1) && ccc <= rcc);      // front right edge

        // Introduced faces F (6)
        change += (ccc < ccf);   // front face
        change += (ccc < lcc);   // left face
        change += (ccc <= ccb);  // back face
        change += (ccc <= rcc);  // right face
        change += (ccc < ctc);   // top face
        change += (ccc <= cdc);  // down face

        atomicAdd(&hist_local_[binary_search(&hist_local_[binNum], binNum, ccc)], change);
    }
    __syncthreads();
    if (blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x < binNum)
        atomicAdd(&VCEC_device[blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x], (int)hist_local_[blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x]);
}

__global__ void init_VCEC_kernel(int binNum, int* VCEC_device) {
    const int idx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    if (idx < binNum) VCEC_device[idx] = 0;
    __syncthreads();
}

// ===================================================================================================================== //
// Lanch kernel

extern "C" void computeECC(
    int imageH,
    int imageW,
    int binNum,
    cudaTextureObject_t texSrc,
    int* VCEC_device,
    float* ascend_unique_arr_device_,
    int chunk_type_id,
    cudaStream_t * stream
)
{
    // Kernel configurations and launch
    dim3 threads;
    int imageH_rounded = (pow(2, ceil(log(binNum) / log(2))) > 512) ? 512 : pow(2, ceil(log(binNum) / log(2)));
    switch (imageH_rounded) {
    case 1: threads = dim3(512, 1); break;
    case 2: threads = dim3(256, 2); break;
    case 4: threads = dim3(128, 4); break;
    case 8: threads = dim3(64, 8); break;
    case 16: threads = dim3(32, 16); break;
    default: threads = dim3(16, 32);
    }
    if (binNum >= 1024) printf("Too much shared memory used, consider reducing chunk size\n");
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));
    if (stream == nullptr)
        ECC_kernel_v30 << <blocks, threads, sizeof(float)* binNum* 2 >> > (texSrc, binNum, VCEC_device, ascend_unique_arr_device_, chunk_type_id);
    else
        ECC_kernel_v30 << <blocks, threads, sizeof(float)* binNum* 2, *stream >> > (texSrc, binNum, VCEC_device, ascend_unique_arr_device_, chunk_type_id);
    getLastCudaError("ECC_kernel() execution failed\n");
}

extern "C" void computeECC_3D(
    int imageH,
    int imageW,
    int imageD,
    int binNum,
    cudaTextureObject_t texSrc,
    int* VCEC_device,
    float* ascend_unique_arr_device_,
    int chunk_type_id,
    cudaStream_t * stream
)
{
    // Kernel configurations and launch
    dim3 threads;
    int imageD_rounded = (pow(2, ceil(log(binNum) / log(2))) > 512) ? 512 : pow(2, ceil(log(binNum) / log(2)));
    switch (imageD_rounded) {
    case 1: threads = dim3(32, 16, 1); break;
    case 2: threads = dim3(32, 8, 2); break;
    default: threads = dim3(16, 8, 4);
    }
    if (binNum >= 1024) printf("Too much shared memory used, consider reducing chunk size\n");
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y), iDivUp(imageD, threads.z));
    if (stream == nullptr)
        ECC_kernel_3D << <blocks, threads, sizeof(float)* binNum* 2 >> > (texSrc, binNum, VCEC_device, ascend_unique_arr_device_, chunk_type_id);
    else
        ECC_kernel_3D << <blocks, threads, sizeof(float)* binNum* 2, *stream >> > (texSrc, binNum, VCEC_device, ascend_unique_arr_device_, chunk_type_id);
    getLastCudaError("ECC_kernel() execution failed\n");
}

extern "C" void init_VCEC_device(
    int binNum,
    int* VCEC_device, 
    cudaStream_t* stream
) 
{
    // Kernel configurations and launch
    int size = (pow(2, ceil(log(binNum) / log(2))) > 512) ? 512 : pow(2, ceil(log(binNum) / log(2)));
    dim3 threads(size);
    dim3 blocks(iDivUp(binNum, threads.x));
    if (stream == nullptr)
        init_VCEC_kernel << <blocks, threads, 0 >> > (binNum, VCEC_device);
    else
        init_VCEC_kernel << <blocks, threads, 0, *stream >> > (binNum, VCEC_device);
    getLastCudaError("ECC_kernel() execution failed\n");
}

// =======================================================================================================
template<int COLUMNS_BLOCKDIM_X, int COLUMNS_BLOCKDIM_Y>
__global__ void convolutionColumnsKernel(
    float* d_Dst,
    float* d_Src,
    int imageW,
    int imageH,
    int pitch
)
{
    __shared__ int image_size;
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];
    int tmp;
    if (threadIdx.x == 0 && threadIdx.y == 0) image_size = imageW * imageH;
    __syncthreads();

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    if (baseX >= imageW) goto out_of_bound;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    const int upper_initial = baseY * pitch + baseX;
    d_Src += upper_initial;
    d_Dst += upper_initial;

    //Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        tmp = i * COLUMNS_BLOCKDIM_Y * pitch;
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (upper_initial + tmp < image_size) ? d_Src[tmp] : 0;
        //if (upper_initial + tmp < image_size) printf("%f %d\n", s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y], upper_initial + tmp);
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
    {
        tmp = i * COLUMNS_BLOCKDIM_Y * pitch;
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (upper_initial + tmp < image_size) ? d_Src[tmp] : 0;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        tmp = i * COLUMNS_BLOCKDIM_Y * pitch;
        if (upper_initial + tmp >= image_size) break;
        float sum = 0;
#pragma unroll

        for (int j = -COLUMNS_BLOCKDIM_Y; j <= COLUMNS_BLOCKDIM_Y; j++)
        {
            sum += kernel_c[COLUMNS_BLOCKDIM_Y - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
        }

        d_Dst[tmp] = (int)sum;
    }
out_of_bound:
}

extern "C" int generate_kernel(float sigma) {
    float x, sum = 0.0;
    int ksize = (int)(2.0 * GaussTruncate_cu * sigma + 1.0);
    ksize = MAX_C(3, ksize);
    if (ksize % 2 == 0) ksize++;
    printf("ksize is %d\n", ksize);
    assert(ksize < MAX_KERNEL_SIZE_CU);
    float* kernel_h = (float*)malloc(ksize * sizeof(float));
    for (int i = 0; i < ksize; i++) {
        x = i - ksize / 2;
        kernel_h[i] = exp(-x * x / (2.0 * sigma * sigma));
        sum += kernel_h[i];
    }
    for (int i = 0; i < ksize; i++) kernel_h[i] /= sum;
    cudaMemcpyToSymbol(kernel_c, kernel_h, ksize * sizeof(float));
    free(kernel_h);
    return ksize;
}

template<int COLUMNS_BLOCKDIM_X, int COLUMNS_BLOCKDIM_Y>
void convolutionVerticalCuda(float*& source_d, float*& blurred_d, int width, int height) {
    dim3 blocks((width + COLUMNS_BLOCKDIM_X - 1) / COLUMNS_BLOCKDIM_X, (height + COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y - 1) / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
    convolutionColumnsKernel<COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y> << <blocks, threads >> > (blurred_d, source_d, width, height, width);
}

extern "C" void gaussianVerticalCuda(float*& source_d, float*& blurred_d, int width, int height, int size_kernel) {
    switch (size_kernel) {
    case 11:	convolutionVerticalCuda<32, 5>(source_d, blurred_d, width, height); break;
    case 13:	convolutionVerticalCuda<32, 6>(source_d, blurred_d, width, height); break;
    case 19:	convolutionVerticalCuda<16, 9>(source_d, blurred_d, width, height); break;
    case 27:	convolutionVerticalCuda<16, 13>(source_d, blurred_d, width, height); break;
    case 37:	convolutionVerticalCuda<16, 18>(source_d, blurred_d, width, height); break;
    case 53:	convolutionVerticalCuda<8, 26>(source_d, blurred_d, width, height); break;
    default:	break;
    }
}