/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/im2col.h>
#include <ops/declarable/helpers/col2im.h>
#include <exceptions/cuda_exception.h>
#include <NDArrayFactory.h>
#include <MmulHelper.h>
#include <PointersManager.h>

namespace nd4j {
namespace ops  {

//////////////////////////////////////////////////////////////////////////
// vol [bS, iC, iD, iH, iW] is convoluted to col [bS, iC, kD, kH, kW, oD, oH, oW]
template <typename T>
static __global__ void vol2colCuda(const void* volume, const Nd4jLong* volShapeInfo, void* column, const Nd4jLong* colShapeInfo,  const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    const T* vol = reinterpret_cast<const T*>(volume);
          T* col = reinterpret_cast<T*>(column);

    __shared__ int colRank, volRank;
    __shared__ Nd4jLong colLen, iD, iH, iW;
    __shared__ Nd4jLong *sharedMem;

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        volRank = 5;
        colRank = 8;

        colLen = shape::length(colShapeInfo);

        iD = volShapeInfo[3];
        iH = volShapeInfo[4];
        iW = volShapeInfo[5];
    }

    __syncthreads();

    const auto colInd = threadIdx.x + blockIdx.x * blockDim.x;

    if(colInd >= colLen)
        return;

    auto coords = sharedMem + threadIdx.x * colRank;

    shape::index2coords(colRank, colShapeInfo + 1, colInd, colLen, coords);

    // const auto colW = coords[7];
    // const auto colH = coords[6];
    // const auto colD = coords[5];
    // const auto kCol = coords[4];
    // const auto kRow = coords[3];
    // const auto kDep = coords[2];
    // const auto c    = coords[1];
    // const auto b    = coords[0];

    const auto colOffset = shape::getOffset(0, colShapeInfo + 1, colShapeInfo + colRank + 1, coords, colRank);

    coords[2] = (-pD + coords[2] * dD) + coords[5] * sD;     // const auto volDep = (-pD + kDep * dD) + colD * sD;
    coords[3] = (-pH + coords[3] * dH) + coords[6] * sH;     // const auto volRow = (-pH + kRow * dH) + colH * sH;
    coords[4] = (-pW + coords[4] * dW) + coords[7] * sW;     // const auto volCol = (-pW + kCol * dW) + colW * sW;

    if (static_cast<unsigned>(coords[2]) >= static_cast<unsigned>(iD) || static_cast<unsigned>(coords[3]) >= static_cast<unsigned>(iH) || static_cast<unsigned>(coords[4]) >= static_cast<unsigned>(iW))
        col[colOffset] = static_cast<T>(0.);
    else
        col[colOffset] = vol[shape::getOffset(0, volShapeInfo + 1, volShapeInfo + volRank + 1, coords, volRank)];
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void vol2colCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                const void* volume, const Nd4jLong* volShapeInfo,
                                      void* column, const Nd4jLong* colShapeInfo,
                                const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    vol2colCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(volume, volShapeInfo, column, colShapeInfo,  sD, sH, sW, pD, pH, pW, dD, dH, dW);
}
BUILD_SINGLE_TEMPLATE(template void vol2colCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t* stream, const void *vol, const Nd4jLong *volShapeInfo, void *col, const Nd4jLong *colShapeInfo, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW), FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::vol2col(nd4j::graph::Context& block, const NDArray& vol, NDArray& col, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

    PointersManager manager(block.launchContext(), "vol2col");

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (col.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = col.rankOf() * sizeof(Nd4jLong) * threadsPerBlock  + 128;

    NDArray::prepareSpecialUse({&col}, {&vol});
    BUILD_SINGLE_SELECTOR(vol.dataType(), vol2colCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, block.launchContext()->getCudaStream(), vol.getSpecialBuffer(), vol.getSpecialShapeInfo(), col.specialBuffer(), col.specialShapeInfo(), sD, sH, sW, pD, pH, pW, dD, dH, dW), FLOAT_TYPES);
    NDArray::registerSpecialUse({&col}, {&vol});

    manager.synchronize();
}


        void ConvolutionUtils::conv2d(nd4j::graph::Context & block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

        }

        void ConvolutionUtils::conv2d(nd4j::graph::Context & block, const std::vector<NDArray*>& inArrs, NDArray* output, const std::vector<int>& intArgs) {

        }

        void ConvolutionUtils::conv2dBP(nd4j::graph::Context & block, const std::vector<NDArray*>& inArrs, const std::vector<NDArray*>& outArrs, const std::vector<int>& intArgs) {

        }

        void ConvolutionUtils::conv2dBP(nd4j::graph::Context & block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

        }

        void ConvolutionUtils::depthwiseConv2d(nd4j::graph::Context & block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

        }

        void ConvolutionUtils::depthwiseConv2dBP(nd4j::graph::Context & block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

        }

        void ConvolutionUtils::sconv2d(nd4j::graph::Context & block, const NDArray* input, const NDArray* weightsDepth, const NDArray* weightsPoint, const NDArray* bias,  NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

        }



        void ConvolutionUtils::col2vol(nd4j::graph::Context & block, const NDArray& col, NDArray& vol, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {

        }

        void ConvolutionUtils::upsampling2d(nd4j::graph::Context & block, const NDArray& input, NDArray& output, const int factorH, const int factorW, const bool isNCHW) {

        }

        void ConvolutionUtils::upsampling3d(nd4j::graph::Context & block, const NDArray& input, NDArray& output, const int factorD, const int factorH, const int factorW, const bool isNCDHW) {

        }

        void ConvolutionUtils::upsampling2dBP(nd4j::graph::Context & block, const NDArray& gradO, NDArray& gradI, const bool isNCHW) {

        }

        void ConvolutionUtils::upsampling3dBP(nd4j::graph::Context & block, const NDArray& gradO, NDArray& gradI, const bool isNCDHW) {

        }

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static __global__ void avgPooling2dCuda(const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {

    // input is  [bS, iC, iH, iW]
    // output is [bS, iC, oH, oW]

    const auto x = reinterpret_cast<const X*>(vx);
          auto z = reinterpret_cast<Z*>(vz);

    __shared__ int bS, iC, oH, oW, iH, iW, strideB, strideC, strideY, strideX, strideOB, strideOC, strideOY, strideOX, length, kHEff, kWEff;

    if (threadIdx.x == 0) {

        bS = shape::sizeAt(xShapeInfo, 0);
        iC = shape::sizeAt(xShapeInfo, 1);
        oH = shape::sizeAt(zShapeInfo, 2);
        oW = shape::sizeAt(zShapeInfo, 3);
        iH = shape::sizeAt(xShapeInfo, 2);
        iW = shape::sizeAt(xShapeInfo, 3);

        strideB = shape::stride(xShapeInfo)[0];
        strideC = shape::stride(xShapeInfo)[1];
        strideY = shape::stride(xShapeInfo)[2];
        strideX = shape::stride(xShapeInfo)[3];

        strideOB = shape::stride(zShapeInfo)[0];
        strideOC = shape::stride(zShapeInfo)[1];
        strideOY = shape::stride(zShapeInfo)[2];
        strideOX = shape::stride(zShapeInfo)[3];

        length = shape::length(zShapeInfo);

        //Replace kernel H/W with *effective* kernel H/W accounting for dilatyon
        kHEff = kH + (kH-1)*(dH-1);
        kWEff = kW + (kW-1)*(dW-1);
    }

    __syncthreads();

    int tid = blockIdx.x * gridDim.x + threadIdx.x;

    for (int index = tid; index < length; index += blockDim.x * gridDim.x) {

        const int pw = index % oW;
        const int ph = (index / oW) % oH;
        const int c = (index / oW / oH) % iC;
        const int n = index / oW / oH / iC;

        int hstart = sH * ph - pH;
        int wstart = sW * pw - pW;
        int hend = hstart + kHEff;
        int wend = wstart + kWEff;

        if(hstart < 0){
            int f = nd4j::math::nd4j_ceil<Z,int>((Z) -hstart / (Z)dH);
            hstart += f * dH;
        }
        if(wstart < 0){
            int f = nd4j::math::nd4j_ceil<Z,int>((Z) -wstart / (Z) dW);
            wstart += f * dW;
        }
        if(hend > iH){
            int f = nd4j::math::nd4j_ceil<Z,int>((Z) (hend-iH) / (Z) dH);
            hend -= f * dH;
        }
        if(wend > iW){
            int f = nd4j::math::nd4j_ceil<Z,int>((Z) (wend-iW) / (Z) dW);
            wend -= f * dW;
        }

        //Accounts for dilation
        int pool_size = nd4j::math::nd4j_ceil<double,int>((double) (hend-hstart) / (double) dH) * nd4j::math::nd4j_ceil<double,int>((double) (wend-wstart) / (double) dW);

        Z sum = 0.0f;

        const X *inSlice = x + (n * strideB + c * strideC);

        for (int h = hstart; h < hend; h += dH)
            for (int w = wstart; w < wend; w += dW)
                sum += static_cast<Z>(inSlice[h * strideY + w * strideX]);

        int divide_factor = pool_size;  //Case 0: exclude padding
        if (extraParam0 == 1)     //Case 1: include padding
            divide_factor = kH * kW;

        z[n * strideOB + c * strideOC + pw * strideOX + ph * strideOY] = sum / static_cast<Z>(divide_factor);
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void avgPooling2dCudaLauncher(nd4j::LaunchContext & block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
    avgPooling2dCuda<X, Z><<<512, 512, 4192, *block.getCudaStream()>>>(vx, vxShapeInfo, vz, vzShapeInfo, kH, kW, sH, sW, pH, pW, dH, dW, extraParam0);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static __global__ void pnormPooling2dCuda(const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {

    // input is  [bS, iC, iH, iW]
    // output is [bS, iC, oH, oW]

    const auto x = reinterpret_cast<const X*>(vx);
          auto z = reinterpret_cast<Z*>(vz);

    __shared__ int bS, iC, oH, oW, iH, iW, strideB, strideC, strideY, strideX, strideOB, strideOC, strideOY, strideOX, length, kHEff, kWEff;
    __shared__ bool fOrder;

    if (threadIdx.x == 0) {

        bS = shape::sizeAt(xShapeInfo, 0);
        iC = shape::sizeAt(xShapeInfo, 1);
        oH = shape::sizeAt(zShapeInfo, 2);
        oW = shape::sizeAt(zShapeInfo, 3);
        iH = shape::sizeAt(xShapeInfo, 2);
        iW = shape::sizeAt(xShapeInfo, 3);

        strideB = shape::stride(xShapeInfo)[0];
        strideC = shape::stride(xShapeInfo)[1];
        strideY = shape::stride(xShapeInfo)[2];
        strideX = shape::stride(xShapeInfo)[3];

        strideOB = shape::stride(zShapeInfo)[0];
        strideOC = shape::stride(zShapeInfo)[1];
        strideOY = shape::stride(zShapeInfo)[2];
        strideOX = shape::stride(zShapeInfo)[3];

        length = shape::length(zShapeInfo);

        //Replace kernel H/W with *effective* kernel H/W accounting for dilatyon
        kHEff = kH + (kH-1)*(dH-1);
        kWEff = kW + (kW-1)*(dW-1);
    }

    __syncthreads();

    int tid = blockIdx.x * gridDim.x + threadIdx.x;

    for (int index = tid; index < length; index += blockDim.x * gridDim.x) {

        const int pw = index % oW;
        const int ph = (index / oW) % oH;
        const int c = (index / oW / oH) % iC;
        const int n = index / oW / oH / iC;

        int hstart = sH * ph - pH;
        int wstart = sW * pw - pW;
        int hend = hstart + kHEff;
        int wend = wstart + kWEff;

        if (hstart < 0) {
            int f = nd4j::math::nd4j_ceil<Z, int>((Z) -hstart / (Z) dH);
            hstart += f * dH;
        }
        if (wstart < 0) {
            int f = nd4j::math::nd4j_ceil<Z, int>((Z) -wstart / (Z) dW);
            wstart += f * dW;
        }
        if (hend > iH) {
            int f = nd4j::math::nd4j_ceil<Z, int>((Z) (hend - iH) / (Z) dH);
            hend -= f * dH;
        }
        if (wend > iW) {
            int f = nd4j::math::nd4j_ceil<Z, int>((Z) (wend - iW) / (Z) dW);
            wend -= f * dW;
        }
        //Accounts for dilation
        int pool_size = nd4j::math::nd4j_ceil<double, int>((double) (hend - hstart) / (double) dH) *
                        nd4j::math::nd4j_ceil<double, int>((double) (wend - wstart) / (double) dW);

        Z sum = 0.f;

        const X *inSlice = x + (n * strideB + c * strideC);

        for (int h = hstart; h < hend; h += dH)
            for (int w = wstart; w < wend; w += dW)
                sum += nd4j::math::nd4j_pow<Z, Z, Z>(static_cast<Z>(nd4j::math::nd4j_abs<X>(inSlice[h * strideY + w * strideX])), extraParam0);

        z[n * strideOB + c * strideOC + pw * strideOX + ph * strideOY] = nd4j::math::nd4j_pow<Z, Z, Z>(sum, (Z) 1.0f / extraParam0);
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void pnormPooling2dCudaLauncher(nd4j::LaunchContext & block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
    pnormPooling2dCuda<X, Z><<<512, 512, 4192, *block.getCudaStream()>>>(vx, vxShapeInfo, vz, vzShapeInfo, kH, kW, sH, sW, pH, pW, dH, dW, extraParam0);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static __global__ void maxPooling2dCuda(const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {

    // input is  [bS, iC, iH, iW]
    // output is [bS, iC, oH, oW]

    const auto x = reinterpret_cast<const X*>(vx);
          auto z = reinterpret_cast<Z*>(vz);

    __shared__ int bS, iC, oH, oW, iH, iW, strideB, strideC, strideY, strideX, strideOB, strideOC, strideOY, strideOX, length, kHEff, kWEff;
    __shared__ bool fOrder;

    if (threadIdx.x == 0) {

        bS = shape::sizeAt(xShapeInfo, 0);
        iC = shape::sizeAt(xShapeInfo, 1);
        oH = shape::sizeAt(zShapeInfo, 2);
        oW = shape::sizeAt(zShapeInfo, 3);
        iH = shape::sizeAt(xShapeInfo, 2);
        iW = shape::sizeAt(xShapeInfo, 3);

        strideB = shape::stride(xShapeInfo)[0];
        strideC = shape::stride(xShapeInfo)[1];
        strideY = shape::stride(xShapeInfo)[2];
        strideX = shape::stride(xShapeInfo)[3];

        strideOB = shape::stride(zShapeInfo)[0];
        strideOC = shape::stride(zShapeInfo)[1];
        strideOY = shape::stride(zShapeInfo)[2];
        strideOX = shape::stride(zShapeInfo)[3];

        length = shape::length(zShapeInfo);

        //Replace kernel H/W with *effective* kernel H/W accounting for dilatyon
        kHEff = kH + (kH-1)*(dH-1);
        kWEff = kW + (kW-1)*(dW-1);
    }

    __syncthreads();

    int tid = blockIdx.x * gridDim.x + threadIdx.x;

    for (int index = tid; index < length; index += blockDim.x * gridDim.x) {

        const int pw = index % oW;
        const int ph = (index / oW) % oH;
        const int c = (index / oW / oH) % iC;
        const int n = index / oW / oH / iC;

        int hstart = sH * ph - pH;
        int wstart = sW * pw - pW;
        int hend = hstart + kHEff;
        int wend = wstart + kWEff;

        if(hstart < 0){
            int f = nd4j::math::nd4j_ceil<Z,int>((Z) -hstart / (Z)dH);
            hstart += f * dH;
        }
        if(wstart < 0){
            int f = nd4j::math::nd4j_ceil<Z,int>((Z) -wstart / (Z) dW);
            wstart += f * dW;
        }
        if(hend > iH){
            int f = nd4j::math::nd4j_ceil<Z,int>((Z) (hend-iH) / (Z) dH);
            hend -= f * dH;
        }
        if(wend > iW){
            int f = nd4j::math::nd4j_ceil<Z,int>((Z) (wend-iW) / (Z) dW);
            wend -= f * dW;
        }
        //Accounts for dilation
        int pool_size = nd4j::math::nd4j_ceil<double,int>((double) (hend-hstart) / (double) dH) * nd4j::math::nd4j_ceil<double,int>((double) (wend-wstart) / (double) dW);

        Z max = -nd4j::DataTypeUtils::max<Z>();

        const X *inSlice = x + (n * strideB + c * strideC);

        for (int h = hstart; h < hend; h += dH) {
            for (int w = wstart; w < wend; w += dW) {
                Z v = static_cast<Z>(inSlice[h * strideY + w * strideX]);
                if (v > max)
                    max = v;
            }
        }

        z[n * strideOB + c * strideOC + pw * strideOX + ph * strideOY] = max;
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void maxPooling2dCudaLauncher(nd4j::LaunchContext & block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0) {
    maxPooling2dCuda<X,Z><<<512, 512, 4192, *block.getCudaStream()>>>(vx, vxShapeInfo, vz, vzShapeInfo, kH, kW, sH, sW, pH, pW, dH, dW, extraParam0);
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::pooling2d(nd4j::graph::Context& block, const NDArray& input, NDArray& output, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const PoolingType poolingMode, const int extraParam0) {

    if(!input.isActualOnDeviceSide()) input.syncToDevice();

    switch (poolingMode) {

        case MAX_POOL: {
                BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), maxPooling2dCudaLauncher, (*block.launchContext(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0), LIBND4J_TYPES, FLOAT_TYPES);
            }
            break;
        case AVG_POOL: {
                BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), avgPooling2dCudaLauncher, (*block.launchContext(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0), LIBND4J_TYPES, FLOAT_TYPES);
            }
            break;
        case PNORM_POOL: {
                BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), pnormPooling2dCudaLauncher, (*block.launchContext(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0), LIBND4J_TYPES, FLOAT_TYPES);
            }
            break;
        default:
            throw std::runtime_error("Pooling2D: Unknown PoolingType used");
    }

    output.tickWriteDevice();
    input.tickReadDevice();

    auto result = cudaStreamSynchronize(*block.launchContext()->getCudaStream());
    if (result != 0)
        throw cuda_exception::build("Pooling2D failed", result);
}




        void ConvolutionUtils::pooling3d(nd4j::graph::Context & block, const NDArray& input, NDArray& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {

        }

        void ConvolutionUtils::pooling2dBP(nd4j::graph::Context & block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int poolingMode, const int extraParam0) {

        }

        void ConvolutionUtils::pooling3dBP(nd4j::graph::Context  &block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int poolingMode, const int extraParam0) {

        }



BUILD_DOUBLE_TEMPLATE(template void maxPooling2dCudaLauncher, (nd4j::LaunchContext & block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0), LIBND4J_TYPES, FLOAT_TYPES);
BUILD_DOUBLE_TEMPLATE(template void pnormPooling2dCudaLauncher, (nd4j::LaunchContext & block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0), LIBND4J_TYPES, FLOAT_TYPES);
BUILD_DOUBLE_TEMPLATE(template void avgPooling2dCudaLauncher, (nd4j::LaunchContext & block, void *vx, Nd4jLong *vxShapeInfo, void *vz, Nd4jLong *vzShapeInfo, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int extraParam0), LIBND4J_TYPES, FLOAT_TYPES);



}
}