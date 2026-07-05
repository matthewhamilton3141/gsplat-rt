// tsdf_integrate.cu — custom CUDA kernel for the TSDF hot-path integrate stage.
//
// This replaces the vectorised-numpy `TSDFVolume.integrate` (the one pipeline
// stage still over its per-call budget, ~13 ms for a 64^3 grid on CPU). One
// CUDA thread owns one voxel: it re-derives the voxel's world coordinate from
// its flat index (so no N^3x3 coordinate buffer is streamed from memory),
// projects it into the current depth frame, and folds the truncated signed
// distance into the running weighted average in-place.
//
// Numerics are a line-for-line match of the numpy reference in
// `src/mapping/collision_proxy.py::TSDFVolume.integrate` (nearest-pixel
// sampling, weight += 1, running mean, clamp to +/-1) so the two paths are
// bit-comparable up to float rounding — see tests/test_tsdf_cuda.py.
//
// Build (on a CUDA box, from repo root):
//     python setup.py build_ext --inplace
// which compiles this into the `gaussian_kernels` extension module.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Flat index convention (C-order, matches numpy reshape/ravel):
//     idx = (i * N + j) * N + k     with world = origin + (i,j,k) * voxel_size
__global__ void tsdf_integrate_kernel(
        float* __restrict__ tsdf,          // (N^3,) in/out
        float* __restrict__ weight,        // (N^3,) in/out
        const float* __restrict__ depth,   // (H*W,) metres, row-major
        const int   N,
        const float voxel_size,
        const float ox, const float oy, const float oz,      // grid origin
        // R_wc = world <- camera rotation, row-major (r<row><col>)
        const float r00, const float r01, const float r02,
        const float r10, const float r11, const float r12,
        const float r20, const float r21, const float r22,
        const float tx, const float ty, const float tz,      // camera position
        const float fx, const float fy, const float cx, const float cy,
        const int   width, const int height,
        const float trunc) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * N * N;
    if (idx >= total) return;

    // Decode (i, j, k) from the flat C-order index.
    const int k = idx % N;
    const int j = (idx / N) % N;
    const int i = idx / (N * N);

    // Voxel-centre world coordinate.
    const float wx = ox + i * voxel_size;
    const float wy = oy + j * voxel_size;
    const float wz = oz + k * voxel_size;

    // Camera frame: vox_cam = R_wc^T @ (world - t). The numpy reference writes
    // this as the row-vector product (world - t) @ R_wc, i.e. each camera-axis
    // component is a dot with a *column* of R_wc.
    const float dx = wx - tx, dy = wy - ty, dz = wz - tz;
    const float cxc = r00 * dx + r10 * dy + r20 * dz;   // column 0
    const float cyc = r01 * dx + r11 * dy + r21 * dz;   // column 1
    const float czc = r02 * dx + r12 * dy + r22 * dz;   // column 2 (depth)

    if (czc <= 0.01f) return;                            // behind / on camera

    // Pinhole projection + nearest-pixel (matches np.rint).
    const int ui = __float2int_rn(fx * cxc / czc + cx);
    const int vi = __float2int_rn(fy * cyc / czc + cy);
    if (ui < 0 || ui >= width || vi < 0 || vi >= height) return;

    const float d_obs = depth[vi * width + ui];
    if (d_obs <= 0.01f) return;                          // no valid observation

    // Truncated signed distance (positive in front of the surface) + clamp.
    float sdf = (d_obs - czc) / trunc;
    sdf = fminf(1.0f, fmaxf(-1.0f, sdf));

    // Weighted running average, weight += 1 per observation.
    const float w_old = weight[idx];
    const float w_new = w_old + 1.0f;
    tsdf[idx] = (tsdf[idx] * w_old + sdf) / w_new;
    weight[idx] = w_new;
}

// Host launcher. `tsdf`, `weight`, `depth`, `R_wc` (3x3) and `t_wc` (3,) are
// expected to be contiguous float32 CUDA tensors; the volume tensors are
// updated in place. Kept deliberately thin — parameter marshalling only.
void tsdf_integrate_cuda(
        torch::Tensor tsdf,
        torch::Tensor weight,
        torch::Tensor depth,
        torch::Tensor R_wc,
        torch::Tensor t_wc,
        int N,
        double voxel_size,
        double ox, double oy, double oz,
        double fx, double fy, double cx, double cy,
        int width, int height,
        double trunc) {

    TORCH_CHECK(tsdf.is_cuda() && weight.is_cuda() && depth.is_cuda(),
                "tsdf, weight and depth must be CUDA tensors");
    TORCH_CHECK(tsdf.scalar_type() == torch::kFloat32, "tsdf must be float32");
    // Updated in place, so they must be contiguous (no silent copy).
    TORCH_CHECK(tsdf.is_contiguous() && weight.is_contiguous(),
                "tsdf and weight must be contiguous");

    auto R = R_wc.contiguous().cpu();       // 9 scalars — pull to host once
    auto t = t_wc.contiguous().cpu();
    const float* r = R.data_ptr<float>();
    const float* tt = t.data_ptr<float>();

    // Hold the contiguous depth in a named tensor: the kernel launch is async,
    // so a temporary from depth.contiguous() could be freed before it runs.
    auto depth_c = depth.contiguous();

    const int total = N * N * N;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    tsdf_integrate_kernel<<<blocks, threads>>>(
        tsdf.data_ptr<float>(),
        weight.data_ptr<float>(),
        depth_c.data_ptr<float>(),
        N, (float)voxel_size,
        (float)ox, (float)oy, (float)oz,
        r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8],
        tt[0], tt[1], tt[2],
        (float)fx, (float)fy, (float)cx, (float)cy,
        width, height, (float)trunc);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "tsdf_integrate kernel launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tsdf_integrate", &tsdf_integrate_cuda,
          "In-place TSDF integrate of one depth frame (CUDA)");
}
