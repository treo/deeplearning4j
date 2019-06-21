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

//
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/top_k.h>
#include <PointersManager.h>
#include <ConstantTadHelper.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template<typename X, typename Y>
__global__ static void inTopKCuda(const void* vx, const Nd4jLong* xShapeInfo,
                                  const void* vy, const Nd4jLong* yShapeInfo,
                                        void* vz, const Nd4jLong* zShapeInfo,
                                  const Nd4jLong* xTadShapeInfo, const Nd4jLong* xTadOffsets,
                                  const uint k) {


    const auto y = reinterpret_cast<const Y*>(vy);
          auto z = reinterpret_cast<bool*>(vz);

    __shared__ uint* sharedMem;
    __shared__ X elemToCompare;
    __shared__ const X* xTad;
    __shared__ Nd4jLong idx, xTadLen;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<uint*>(shmem);

        xTadLen = shape::length(xTadShapeInfo);

        xTad = reinterpret_cast<const X*>(vx) + xTadOffsets[blockIdx.x];
        idx = y[shape::getIndexOffset(blockIdx.x, yShapeInfo, shape::length(yShapeInfo))]; // shape::length(yShapeInfo) == numTads
        elemToCompare = xTad[shape::getIndexOffset(idx, xTadShapeInfo, xTadLen)];
    }

    __syncthreads();

    sharedMem[threadIdx.x] = 0;
    for (Nd4jLong i = threadIdx.x; i < xTadLen; i += blockDim.x)
        if(elemToCompare < xTad[shape::getIndexOffset(i, xTadShapeInfo, xTadLen)])
            ++sharedMem[threadIdx.x];

    __syncthreads();

    // aggregate sum
    for (uint activeThreads = blockDim.x / 2; activeThreads > 0; activeThreads /= 2) {
        if (threadIdx.x < activeThreads)
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + activeThreads];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        z[shape::getIndexOffset(blockIdx.x, zShapeInfo, shape::length(zShapeInfo))] = *sharedMem < k;
}

///////////////////////////////////////////////////////////////////
template<typename X, typename Y>
static void inTopKCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                               const void *vx, const Nd4jLong *xShapeInfo,
                               const void *vy, const Nd4jLong *yShapeInfo,
                                     void *vz, const Nd4jLong *zShapeInfo,
                               const Nd4jLong* xTadShapeInfo, const Nd4jLong* xTadOffsets,
                               const uint k) {

    inTopKCuda<X,Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, xTadShapeInfo, xTadOffsets, k);
}

///////////////////////////////////////////////////////////////////
int inTopKFunctor(nd4j::LaunchContext * context, const NDArray* predictions, const NDArray* targets, NDArray* output, const uint k) {

    PointersManager manager(context, "in_top_k");

    const auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(predictions->getShapeInfo(), {1});

    const int threadsPerBlock = MAX_NUM_THREADS;
    const int blocksPerGrid = static_cast<int>(packX.numberOfTads());
    const int sharedMem = sizeof(uint) * threadsPerBlock + 128;

    const auto xType = predictions->dataType();
    const auto yType = targets->dataType();

    NDArray::prepareSpecialUse({output}, {predictions, targets});
    BUILD_DOUBLE_SELECTOR(xType, yType, inTopKCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), predictions->getSpecialBuffer(), predictions->getSpecialShapeInfo(), targets->getSpecialBuffer(), targets->getSpecialShapeInfo(), output->getSpecialBuffer(), output->getSpecialShapeInfo(), packX.specialShapeInfo(), packX.specialOffsets(), k), FLOAT_TYPES, INTEGER_TYPES);
    NDArray::registerSpecialUse({output}, {predictions, targets});

    manager.synchronize();

    return Status::OK();
}

    template <typename X, typename Y>
    static _CUDA_G void topValuesMover(void *vx, Nd4jLong *xTadShapeInfo, Nd4jLong *xTadOffsets, void *vi, Nd4jLong *iTadShapeInfo, Nd4jLong *iTadOffsets, void *vz, Nd4jLong *zTadShapeInfo, Nd4jLong *zTadOffsets, Nd4jLong tadLength, int numTads, int k) {
        for (int t = blockIdx.x; t < numTads; t += gridDim.x) {
            auto x = reinterpret_cast<X*>(vx) + xTadOffsets[t];
            auto i = reinterpret_cast<Y*>(vi) + iTadOffsets[t];
            auto z = reinterpret_cast<X*>(vz) + zTadOffsets[t];

            for (int e = threadIdx.x; e < k; e += blockDim.x) {
                auto idx = i[shape::getIndexOffset(e, iTadShapeInfo, k)];

                z[shape::getIndexOffset(e, zTadShapeInfo, k)] = x[shape::getIndexOffset(idx, xTadShapeInfo, tadLength)];
            }
        }
    }


    template <typename X, typename Y>
    static int topKFunctor_(nd4j::LaunchContext * context, const NDArray* input, NDArray* values, NDArray* indices, const uint k, bool needSort) {

        auto packX = ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), {input->rankOf() - 1});
        auto packI = ConstantTadHelper::getInstance()->tadForDimensions(indices->shapeInfo(), {input->rankOf() - 1});
        auto packZ = ConstantTadHelper::getInstance()->tadForDimensions(values->shapeInfo(), {input->rankOf() - 1});

        auto tadLength = shape::length(packX.primaryShapeInfo());

        // we get top K values first
        if (k == 1) {
            input->applyIndexReduce(indexreduce::IndexMax, indices, {input->rankOf() - 1});
        } else {

        }

        // copy values on specified indices
        topValuesMover<X,Y><<<256, 256, 1024, *context->getCudaStream()>>>(input->getSpecialBuffer(), packX.platformShapeInfo(), packX.platformOffsets(), indices->specialBuffer(), packI.platformShapeInfo(), packI.platformOffsets(), values->specialBuffer(), packZ.platformShapeInfo(), packZ.platformOffsets(), tadLength, packX.numberOfTads(), k);

        // optional sort
        if (k > 1 && needSort) {
            //
        }

        return Status::OK();
    }

    int topKFunctor(nd4j::LaunchContext * context, const NDArray* input, NDArray* values, NDArray* indices, const uint k, bool needSort) {
        input->syncToDevice();

        BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), topKFunctor_, (context, input, values, indices, k, needSort), LIBND4J_TYPES, INTEGER_TYPES);

        values->tickWriteDevice();
        indices->tickWriteDevice();

        return Status::OK();
    }


    BUILD_DOUBLE_TEMPLATE(template int topKFunctor_, (nd4j::LaunchContext * context, const NDArray* input, NDArray* values, NDArray* indices, const uint k, bool needSort), LIBND4J_TYPES, INTEGER_TYPES);

}
}
}