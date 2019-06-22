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
// @author Yurii Shyrma, created on 25.02.2018
//


#include<ops/declarable/helpers/batchnorm.h>
#include <helpers/ShapeUtils.h>
#include <OmpLaunchHelper.h>
#include <ConstantTadHelper.h>
#include <PointersManager.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void batchnormCuda(const void* vx, const Nd4jLong* xShapeInfo,
									const void* vMean, const Nd4jLong* meanShapeInfo,
									const void* vVariance, const Nd4jLong* varianceShapeInfo,
									const void* vGamma, const Nd4jLong* gammaShapeInfo,
									const void* vBeta, const Nd4jLong* betaShapeInfo,
										  void* vz, const Nd4jLong* zShapeInfo,
									const Nd4jLong* xTadShapeInfo, const Nd4jLong* xTadOffsets,
									const Nd4jLong* zTadShapeInfo, const Nd4jLong* zTadOffsets,
									const T epsilon) {

	const auto x    	= reinterpret_cast<const T*>(vx);
          auto z        = reinterpret_cast<T*>(vz);
	const auto mean 	= reinterpret_cast<const T*>(vMean);
	const auto variance = reinterpret_cast<const T*>(vVariance);
	const auto gamma    = reinterpret_cast<const T*>(vGamma);
	const auto beta     = reinterpret_cast<const T*>(vBeta);

    // maxRank = xRank = zRank, minRank = meanRank = varianceRank = gammaRank = betaRank
    __shared__ Nd4jLong minLen, tadLen, totalThreads;

    if (threadIdx.x == 0) {

        totalThreads = gridDim.x * blockDim.x;

        minLen = shape::length(meanShapeInfo);
        tadLen = shape::length(xShapeInfo) / minLen;
    }
    __syncthreads();

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint i = tid; i < minLen; i += totalThreads) {

		const auto meanOffset     = shape::getIndexOffset(i, meanShapeInfo, minLen);
    	const auto varianceOffset = shape::getIndexOffset(i, varianceShapeInfo, minLen);

    	T sigmaInvGam = 1. / nd4j::math::nd4j_sqrt<T, T>(variance[varianceOffset] + epsilon);

    	if(gamma != nullptr)
    		sigmaInvGam *= gamma[shape::getIndexOffset(i, gammaShapeInfo, minLen)];

		auto betaOffset = 0;
    	if(beta != nullptr)
    		betaOffset = shape::getIndexOffset(i, betaShapeInfo, minLen);

    	const auto xTad = x + xTadOffsets[i];
    		  auto zTad = z + zTadOffsets[i];

    	for (uint j = 0; j < tadLen; ++j) {

    		const auto xTadOffset = shape::getIndexOffset(j, xTadShapeInfo, tadLen);
    		const auto zTadOffset = shape::getIndexOffset(j, zTadShapeInfo, tadLen);

    		zTad[zTadOffset] = (xTad[xTadOffset] - mean[meanOffset]) * sigmaInvGam;

    		if(beta != nullptr)
				zTad[zTadOffset] += beta[betaOffset];
    	}
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void batchnormCuda2(const void* vx, const Nd4jLong* xShapeInfo,
                                    const void* vMean, const Nd4jLong* meanShapeInfo,
                                    const void* vVariance, const Nd4jLong* varianceShapeInfo,
                                    const void* vGamma, const Nd4jLong* gammaShapeInfo,
                                    const void* vBeta, const Nd4jLong* betaShapeInfo,
                                          void* vz, const Nd4jLong* zShapeInfo,
                                    const int numDims, const int* dims,
                                    const T epsilon) {

    const auto x        = reinterpret_cast<const T*>(vx);
          auto z        = reinterpret_cast<T*>(vz);
    const auto mean     = reinterpret_cast<const T*>(vMean);
    const auto variance = reinterpret_cast<const T*>(vVariance);
    const auto gamma    = reinterpret_cast<const T*>(vGamma);
    const auto beta     = reinterpret_cast<const T*>(vBeta);

    __shared__ int xRank, minRank;       // xRank == zRank. minRank = meanRank = varianceRank = gammaRank = betaRank
    __shared__ Nd4jLong xLen, totalThreads, *sharedMem; // xLen = zLen


    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        sharedMem    = reinterpret_cast<Nd4jLong*>(shmem);
        totalThreads = gridDim.x * blockDim.x;

        xLen    = shape::length(xShapeInfo);
        xRank   = shape::rank(xShapeInfo);
        minRank = shape::rank(meanShapeInfo);
    }
    __syncthreads();

    auto coords = sharedMem + threadIdx.x * xRank;
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint i = tid; i < xLen; i += totalThreads) {

        shape::index2coords(xRank, shape::shapeOf(const_cast<Nd4jLong*>(xShapeInfo)), i, xLen, coords);

        const auto xOffset = shape::getOffset(0, shape::shapeOf(const_cast<Nd4jLong*>(xShapeInfo)), shape::stride(const_cast<Nd4jLong*>(xShapeInfo)), coords, xRank);
        const auto zOffset = shape::getOffset(0, shape::shapeOf(const_cast<Nd4jLong*>(zShapeInfo)), shape::stride(const_cast<Nd4jLong*>(zShapeInfo)), coords, xRank);

        if(minRank == xRank) {
            for (uint i = 0, j = 0; i < xRank; ++i) {
                if(j < numDims && i != dims[j])
                    coords[i] = 0;
                else
                    ++j;
            }
        }
        else    // minRank = numDims = 1 in this case
            coords[0] = coords[dims[0]];

        const auto meanOffset     = shape::getOffset(0, shape::shapeOf(const_cast<Nd4jLong*>(meanShapeInfo)), shape::stride(const_cast<Nd4jLong*>(meanShapeInfo)), coords, minRank);
        const auto varianceOffset = shape::getOffset(0, shape::shapeOf(const_cast<Nd4jLong*>(varianceShapeInfo)), shape::stride(const_cast<Nd4jLong*>(varianceShapeInfo)), coords, minRank);

        T sigmaInvGam = 1. / nd4j::math::nd4j_sqrt<T, T>(variance[varianceOffset] + epsilon);

        if(gamma != nullptr) {
            const auto gammaOffset = shape::getOffset(0, shape::shapeOf(const_cast<Nd4jLong*>(gammaShapeInfo)), shape::stride(const_cast<Nd4jLong*>(gammaShapeInfo)), coords, minRank);
            sigmaInvGam *= gamma[gammaOffset];
        }

        z[zOffset] = (x[xOffset] - mean[meanOffset]) * sigmaInvGam;

        if(beta != nullptr) {
            const auto betaOffset = shape::getOffset(0, shape::shapeOf(const_cast<Nd4jLong*>(betaShapeInfo)), shape::stride(const_cast<Nd4jLong*>(betaShapeInfo)), coords, minRank);
            z[zOffset] += beta[betaOffset];
        }
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
__host__ static void batchnormCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream,
											const void* vx, const Nd4jLong* xShapeInfo,
                                           	const void* vMean, const Nd4jLong* meanShapeInfo,
											const void* vVariance, const Nd4jLong* varianceShapeInfo,
											const void* vGamma, const Nd4jLong* gammaShapeInfo,
											const void* vBeta, const Nd4jLong* betaShapeInfo,
												  void* vz, const Nd4jLong* zShapeInfo,
											const Nd4jLong* xTadShapeInfo, const Nd4jLong* xTadOffsets,
											const Nd4jLong* zTadShapeInfo, const Nd4jLong* zTadOffsets,
											const double epsilon) {

    batchnormCuda<T><<<blocksPerGrid, threadsPerBlock, 1024, *stream>>>(vx, xShapeInfo, vMean, meanShapeInfo, vVariance, varianceShapeInfo, vGamma, gammaShapeInfo, vBeta, betaShapeInfo, vz, zShapeInfo, xTadShapeInfo, xTadOffsets, zTadShapeInfo, zTadOffsets, static_cast<T>(epsilon));
}
BUILD_SINGLE_TEMPLATE(template void batchnormCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void* vx, const Nd4jLong* xShapeInfo, const void* vMean, const Nd4jLong* meanShapeInfo, const void* vVariance, const Nd4jLong* varianceShapeInfo, const void* vGamma, const Nd4jLong* gammaShapeInfo, const void* vBeta, const Nd4jLong* betaShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const Nd4jLong* xTadShapeInfo, const Nd4jLong* xTadOffsets, const Nd4jLong* zTadShapeInfo, const Nd4jLong* zTadOffsets, const double epsilon), FLOAT_TYPES);

///////////////////////////////////////////////////////////////////
template<typename T>
__host__ static void batchnormCudaLauncher2(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                            const void* vx, const Nd4jLong* xShapeInfo,
                                            const void* vMean, const Nd4jLong* meanShapeInfo,
                                            const void* vVariance, const Nd4jLong* varianceShapeInfo,
                                            const void* vGamma, const Nd4jLong* gammaShapeInfo,
                                            const void* vBeta, const Nd4jLong* betaShapeInfo,
                                                  void* vz, const Nd4jLong* zShapeInfo,
                                            const int numDims, const int* dims,
                                            const double epsilon) {

    batchnormCuda2<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vMean, meanShapeInfo, vVariance, varianceShapeInfo, vGamma, gammaShapeInfo, vBeta, betaShapeInfo, vz, zShapeInfo, numDims, dims, static_cast<T>(epsilon));
}
BUILD_SINGLE_TEMPLATE(template void batchnormCudaLauncher2, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, const void* vx, const Nd4jLong* xShapeInfo, const void* vMean, const Nd4jLong* meanShapeInfo, const void* vVariance, const Nd4jLong* varianceShapeInfo, const void* vGamma, const Nd4jLong* gammaShapeInfo, const void* vBeta, const Nd4jLong* betaShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const int numDims, const int* dims, const double epsilon), FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
void batchnorm(const NDArray* input, const NDArray* mean, const NDArray* variance, const NDArray* gamma, const NDArray* beta, NDArray* output, const std::vector<int>& axes, const double epsilon) {

	std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(input->rankOf(), axes);

	auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimsToExclude);
    auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->shapeInfo(), dimsToExclude);

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (mean->lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

    PointersManager manager(input->getContext(), "batchnorm");

    NDArray::prepareSpecialUse({output}, {input, mean, variance, gamma, beta});
    BUILD_SINGLE_SELECTOR(input->dataType(), batchnormCudaLauncher, (blocksPerGrid, threadsPerBlock, input->getContext()->getCudaStream(), input->getSpecialBuffer(), input->getSpecialShapeInfo(), mean->getSpecialBuffer(), mean->getSpecialShapeInfo(), variance->getSpecialBuffer(), variance->getSpecialShapeInfo(), gamma ? gamma->getSpecialBuffer() : nullptr, gamma ? gamma->getSpecialShapeInfo() : nullptr, beta ? beta->getSpecialBuffer() : nullptr, beta ? beta->getSpecialShapeInfo() : nullptr, output->specialBuffer(), output->specialShapeInfo(), packX.platformShapeInfo(), packX.platformOffsets(), packZ.platformShapeInfo(), packZ.platformOffsets(), epsilon), FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input, mean, variance, gamma, beta});

    manager.synchronize();


    // const int threadsPerBlock = MAX_NUM_THREADS / 4;
    // const int blocksPerGrid = (input->lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    // const int sharedMem = sizeof(Nd4jLong) * threadsPerBlock * input->rankOf() + 128;

    // PointersManager manager(input->getContext(), "batchnorm");

    // const int* dims = reinterpret_cast<int*>(manager.replicatePointer(axes.data(), axes.size() * sizeof(int)));

    // NDArray::prepareSpecialUse({output}, {input, mean, variance, gamma, beta});
    // BUILD_SINGLE_SELECTOR(input->dataType(), batchnormCudaLauncher2, (blocksPerGrid, threadsPerBlock, sharedMem, input->getContext()->getCudaStream(), input->getSpecialBuffer(), input->getSpecialShapeInfo(), mean->getSpecialBuffer(), mean->getSpecialShapeInfo(), variance->getSpecialBuffer(), variance->getSpecialShapeInfo(), gamma ? gamma->getSpecialBuffer() : nullptr, gamma ? gamma->getSpecialShapeInfo() : nullptr, beta ? beta->getSpecialBuffer() : nullptr, beta ? beta->getSpecialShapeInfo() : nullptr, output->specialBuffer(), output->specialShapeInfo(), axes.size(), dims, epsilon), FLOAT_TYPES);
    // NDArray::registerSpecialUse({output}, {input, mean, variance, gamma, beta});

    // manager.synchronize();
}


}
}
}

