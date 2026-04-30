#include <stdint.h>
#include <cuda_runtime.h>

__device__ __forceinline__ uint64_t toy_hash_u64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__global__ void toy_search_kernel(
    uint64_t base,
    uint64_t stride,
    uint64_t target_hash,
    uint32_t iterations_per_thread,
    uint64_t *found_nonce,
    uint32_t *found_flag
) {
    uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    uint64_t total_threads = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
    uint64_t nonce = base + tid * stride;
    uint64_t step = total_threads * stride;

    for (uint32_t i = 0; i < iterations_per_thread; i++) {
        if (atomicAdd(found_flag, 0) != 0) {
            return;
        }
        if (toy_hash_u64(nonce) == target_hash) {
            if (atomicCAS(found_flag, 0, 1) == 0) {
                *found_nonce = nonce;
            }
            return;
        }
        nonce += step;
    }
}

extern "C" int satoshi_toy_cuda_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count;
}

extern "C" int satoshi_toy_cuda_device_name(int device, char *out, int out_len) {
    if (out == 0 || out_len <= 0) {
        return -1;
    }
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        out[0] = 0;
        return -2;
    }

    int i = 0;
    for (; i < out_len - 1 && prop.name[i] != 0; i++) {
        out[i] = prop.name[i];
    }
    out[i] = 0;
    return 0;
}

extern "C" int satoshi_toy_cuda_search_device(
    int device,
    uint64_t base,
    uint64_t stride,
    uint64_t target_hash,
    uint32_t blocks,
    uint32_t threads_per_block,
    uint32_t iterations_per_thread,
    uint64_t *host_found_nonce,
    uint64_t *host_searched
) {
    if (host_found_nonce == 0 || host_searched == 0 || blocks == 0 || threads_per_block == 0 || iterations_per_thread == 0) {
        return -1;
    }

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return -2;
    }

    uint64_t *device_found_nonce = 0;
    uint32_t *device_found_flag = 0;
    err = cudaMalloc((void **)&device_found_nonce, sizeof(uint64_t));
    if (err != cudaSuccess) {
        return -3;
    }
    err = cudaMalloc((void **)&device_found_flag, sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(device_found_nonce);
        return -4;
    }

    uint64_t zero64 = 0;
    uint32_t zero32 = 0;
    cudaMemcpy(device_found_nonce, &zero64, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_found_flag, &zero32, sizeof(uint32_t), cudaMemcpyHostToDevice);

    toy_search_kernel<<<blocks, threads_per_block>>>(
        base,
        stride,
        target_hash,
        iterations_per_thread,
        device_found_nonce,
        device_found_flag
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(device_found_nonce);
        cudaFree(device_found_flag);
        return -5;
    }

    uint32_t found_flag = 0;
    cudaMemcpy(&found_flag, device_found_flag, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_found_nonce, device_found_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(device_found_nonce);
    cudaFree(device_found_flag);

    *host_searched = (uint64_t)blocks * (uint64_t)threads_per_block * (uint64_t)iterations_per_thread;
    return found_flag ? 1 : 0;
}
