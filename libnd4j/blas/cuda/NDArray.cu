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

#ifndef NDARRAY_CPP
#define NDARRAY_CPP

#include "../NDArray.h"
#include "../NDArrayFactory.h"
#include "NativeOpExecutioner.h"
#include <memory/Workspace.h>
#include <memory/MemoryRegistrator.h>
#include <ops.h>
#include <ops/gemm.h>
#include <pointercast.h>
#include <stdexcept>
#include <memory>
#include <helpers/logger.h>
#include <loops/pairwise_transform.h>
#include <loops/transform_same.h>
#include <loops/random.h>
#include <loops/broadcasting.h>
#include <indexing/NDIndex.h>
#include <indexing/IndicesList.h>
#include <helpers/ShapeUtils.h>
#include <sstream>
#include <helpers/ArrayUtils.h>
#include <MmulHelper.h>
#include <helpers/threshold.h>
#include <exceptions/datatype_exception.h>
#include <exceptions/cuda_exception.h>
#include <specials_cuda.h>
#include <loops/special_kernels.h>
#include <PointersManager.h>
#include "../NDArray.hpp"
#include <ConstantShapeHelper.h>

namespace nd4j {

void* NDArray::platformBuffer()             { return specialBuffer();    }
void* NDArray::getPlatformBuffer() const    { return getSpecialBuffer(); }

Nd4jLong* NDArray::getPlatformShapeInfo() const { return getSpecialShapeInfo(); }
Nd4jLong* NDArray::platformShapeInfo()          { return specialShapeInfo(); }

void NDArray::syncToDevice() const          { _buffer->syncToSpecial();  }
void NDArray::syncToHost() const            { _buffer->syncToPrimary(getContext()); }
void NDArray::tickWriteHost() const         { _buffer->writePrimary();   }
void NDArray::tickWriteDevice() const       { _buffer->writeSpecial();   }
void NDArray::tickReadHost() const          { _buffer->readPrimary();    }
void NDArray::tickReadDevice() const        { _buffer->readSpecial();    }
void NDArray::tickBothActual() const        { _buffer->writePrimary(); _buffer->readSpecial(); }
bool NDArray::isActualOnHostSide() const    { return _buffer->isPrimaryActual(); }
bool NDArray::isActualOnDeviceSide() const  { return _buffer->isSpecialActual(); }
void NDArray::makeBothBuffersActual() const { if(!isActualOnHostSide()) syncToHost(); if(!isActualOnDeviceSide()) syncToDevice(); }


///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void fillAsTriangularCuda(const void* vx, const Nd4jLong* xShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const T val, const int lower, const int upper) {

    const auto x = reinterpret_cast<const T*>(vx);
          auto z = reinterpret_cast<T*>(vz);

    __shared__ int zRank, xRank, areSameOffsets;        // xRank == zRank always, except when xRank = 1, in this case zRank = 2
    __shared__ Nd4jLong zLen, totalThreads, *sharedMem;  // xLen == zLen, except when xRank = 1, in this case zLen = 2*xLen

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);
        areSameOffsets = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
        xRank = shape::rank(xShapeInfo);
        zRank = shape::rank(zShapeInfo);
        zLen  = shape::length(zShapeInfo);
        totalThreads = gridDim.x * blockDim.x;
    }

    __syncthreads();

    auto coords = sharedMem + threadIdx.x * zRank;

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < zLen; i += totalThreads) {

        shape::index2coords(zRank, shape::shapeOf(const_cast<Nd4jLong*>(zShapeInfo)), i, zLen, coords);
        const auto zOffset = shape::getOffset(0, shape::shapeOf(const_cast<Nd4jLong*>(zShapeInfo)), shape::stride(const_cast<Nd4jLong*>(zShapeInfo)), coords, zRank);

        // if( (row + upper < col) || (row + lower > col) )
        if((coords[zRank - 2] + upper < coords[zRank - 1]) || (coords[zRank - 2] + lower > coords[zRank - 1]))
            z[zOffset] = val;
        else if(vx != vz) {      // when x and z are different arrays
            if(xRank != zRank)
                coords[0] = coords[1];
            const auto xOffset = areSameOffsets ? zOffset : shape::getOffset(0, shape::shapeOf(const_cast<Nd4jLong*>(xShapeInfo)), shape::stride(const_cast<Nd4jLong*>(xShapeInfo)), coords, xRank);
            z[zOffset] = x[xOffset];
        }
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
void NDArray::fillAsTriangular(const float val, int lower, int upper, const char direction, NDArray* target) {

    if (isS())
        throw std::runtime_error("NDArray::fillAsTriangular: you can't use this method on String array!");

    if(target == nullptr)
        target = this;

    if(!isSameShape(target) && !(rankOf() == 1 && target->rankOf() == 2 && sizeAt(0) == target->sizeAt(0) && sizeAt(0) == target->sizeAt(1)))
        throw std::string("NDArray::fillAsTriangular method: wrong shape of target array !");

     if (direction == 'u')
        lower = -target->sizeAt(-2);
    else if (direction == 'l')
        upper = target->sizeAt(-1);

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (target->lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = threadsPerBlock * sizeof(decltype(*target->getShapeInfo())) * target->rankOf() + 128;

    PointersManager manager(getContext(), "NDArray::fillAsTriangular");

    NDArray::prepareSpecialUse({target}, {this});
    fillAsTriangularCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *getContext()->getCudaStream()>>>(getPlatformBuffer(), getPlatformShapeInfo(), target->getPlatformBuffer(), target->getPlatformShapeInfo(), static_cast<T>(val), lower, upper);
    NDArray::registerSpecialUse({target}, {this});

    manager.synchronize();
}
BUILD_SINGLE_TEMPLATE(template void NDArray::fillAsTriangular, (const float val, int lower, int upper, const char direction, NDArray* target), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void identityMatrixCuda(void* vx, const Nd4jLong* xShapeInfo, const T val) {

    auto x = reinterpret_cast<T*>(vx);

    __shared__ int rank;
    __shared__ Nd4jLong len, totalThreads, *sharedMem;  // xLen == zLen, except when xRank = 1, in this case zLen = 2*xLen

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);
        rank = shape::rank(xShapeInfo);
        len  = shape::length(xShapeInfo);
        totalThreads = gridDim.x * blockDim.x;
    }

    __syncthreads();

    auto coords = sharedMem + threadIdx.x * rank;

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < len; i += totalThreads) {

        shape::index2coords(rank, shape::shapeOf(const_cast<Nd4jLong*>(xShapeInfo)), i, len, coords);
        const auto offset = shape::getOffset(0, shape::shapeOf(const_cast<Nd4jLong*>(xShapeInfo)), shape::stride(const_cast<Nd4jLong*>(xShapeInfo)), coords, rank);

        if(coords[rank - 2] == coords[rank - 1]) // row == col -> on diagonal
            x[offset] = val;
        else
            x[offset] = static_cast<T>(0);
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
static void identityMatrixCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, void* vx, const Nd4jLong *xShapeInfo, const float val) {

    identityMatrixCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, static_cast<T>(val));
}
BUILD_SINGLE_TEMPLATE(template void identityMatrixCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, void* vx, const Nd4jLong *xShapeInfo, const float val), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
void NDArray::setIdentity() {
    if (isS())
        throw std::runtime_error("NDArray::setIdentity: you can't use this method on String array!");

    // if (rankOf() != 2)
    //     throw std::runtime_error("NDArray::setIdentity: method should work only for 2D tensors. But " + toStringValue(rankOf()) + " was given.");

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = threadsPerBlock * sizeof(decltype(getShapeInfo())) * rankOf() + 128;

    PointersManager manager(getContext(), "NDArray::setIdentity");

    syncToDevice();
    BUILD_SINGLE_SELECTOR(dataType(), identityMatrixCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, getContext()->getCudaStream(), getPlatformBuffer(), getPlatformShapeInfo(), 1.f), LIBND4J_TYPES);
    tickWriteDevice();

    manager.synchronize();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NDArray::swapUnsafe(NDArray& other) {
    auto xType = this->dataType();

    if (xType != other.dataType())
        throw std::runtime_error("NDArray::swapUnsage method: both arrays must have the same data type");

    if(specialBuffer() == nullptr || other.specialBuffer() == nullptr)
        throw std::runtime_error("NDArray::swapUnsafe method: input array should not be empty!");

    if(lengthOf() != other.lengthOf())
        throw std::runtime_error("NDArray::swapUnsafe method: input arrays should have the same length!");

    BUILD_SINGLE_SELECTOR(xType, templatedSwapUnsafe, (specialBuffer(), specialShapeInfo(), other.specialBuffer(), other.specialShapeInfo(), getContext()->getCudaStream()), LIBND4J_TYPES);
}

////////////////////////////////////////////////////////////////////////
void NDArray::synchronize(const char* msg) const {
    auto res = cudaStreamSynchronize(*(getContext()->getCudaStream()));
    if (res != 0)
        throw std::runtime_error(msg + std::string(": synchronization failed !"));
}
////////////////////////////////////////////////////////////////////////
void NDArray::prepareSpecialUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList, bool synchronizeWritables) {

    for (const auto& a : readList)
        if(a != nullptr)
            a->syncToDevice();

    for (const auto& a : writeList) {
        a->getDataBuffer()->allocateSpecial();
        if (synchronizeWritables)
            a->syncToDevice();
    }
}

////////////////////////////////////////////////////////////////////////
void NDArray::registerSpecialUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList) {

    for (const auto& p : readList)
        if(p != nullptr)
            p->tickReadDevice();

    for (const auto& p : writeList)
        p->tickWriteDevice();
}

////////////////////////////////////////////////////////////////////////
void NDArray::preparePrimaryUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList, bool synchronizeWritables) {

    for (const auto& a : readList)
        if(a != nullptr)
            a->syncToHost();

    for (const auto& a : writeList) {
        a->getDataBuffer()->allocatePrimary();
        if (synchronizeWritables)
            a->syncToHost();
    }
}

////////////////////////////////////////////////////////////////////////
void NDArray::registerPrimaryUse(const std::initializer_list<const NDArray*>& writeList, const std::initializer_list<const NDArray*>& readList) {

    for (const auto& p : readList)
        if(p != nullptr)
            p->tickReadHost();

    for (const auto& p : writeList)
        p->tickWriteHost();
}

//////////////////////////////////////////////////////////////////////////
void NDArray::syncShape() const {
    cudaMemcpy(getSpecialShapeInfo(), getShapeInfo(), shape::shapeInfoByteLength(getShapeInfo()), cudaMemcpyHostToDevice);
}

//////////////////////////////////////////////////////////////////////////
void* NDArray::specialBufferWithOffset(Nd4jLong offset) const {
    return getSpecialBuffer() != nullptr ? static_cast<int8_t*>(getSpecialBuffer()) + (offset * sizeOfT()) : nullptr;
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
NDArray NDArray::tile(const std::vector<Nd4jLong>& reps) const {
    int dim = reps.size();
    int product = 1;
    for(const auto& item : reps)
        product *= item;
    if(product == 0)
        throw std::runtime_error("NDArray::tile method: one of the elements in reps array is zero !");

    int rankOld = rankOf();
    int diff = rankOld - dim;
    if(product==1) {        // in this case 2 possibilities are present: just reshape or nothing to do
        NDArray result(*this);
        if(diff < 0) {      // reshape to higher dimension
            std::vector<Nd4jLong> shapeNew = reps;               // need to have unities at first "diff" positions of new shape
            memcpy(&shapeNew[-diff], result.getShapeInfo()+1, rankOld * sizeof(Nd4jLong));   // put old shape numbers at rest of positions
            result.reshapei(ordering(), shapeNew);
        }
        return result;             // nothing to do, if diff >= 0 -> identity tile
    }

    // evaluate shapeInfo for resulting array
    auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, getContext()->getWorkspace());
    // create new buffer, in any case the memory amount new buffer points to is bigger then those for old _buffer
    std::shared_ptr<DataBuffer> newBuff = std::make_shared<DataBuffer>(shape::length(newShapeInfo) * sizeOfT(), dataType(), getContext()->getWorkspace(), true);
    // assign new shape and new buffer to resulting array
    NDArray result(newBuff, ShapeDescriptor(newShapeInfo), getContext());

    // fill newBuff, loop through all elements of newBuff
    // looping through getBuffer() goes automatically by means of getSubArrayIndex applying
    const auto resultLen = result.lengthOf();
    auto xType = this->dataType();
    auto stream = getContext()->getCudaStream();

    prepareSpecialUse({&result}, {this});
    BUILD_SINGLE_SELECTOR(xType, tileKernelH, (this->getSpecialBuffer(), this->getSpecialShapeInfo(), result.getSpecialBuffer(), result.getSpecialShapeInfo(), resultLen, stream), LIBND4J_TYPES);
    registerSpecialUse({&result}, {this});

    return result;
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
void NDArray::tile(const std::vector<Nd4jLong>& reps, NDArray& target) const {

    // evaluate true tile shapeInfo for comparison with target shapeInfo
    auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, getContext()->getWorkspace());
    if(!shape::equalsSoft(newShapeInfo, target.getShapeInfo()))  {
        throw std::runtime_error("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");
    }

    // fill newBuff, loop through all elements of newBuff
    // looping through getBuffer() goes automatically by means of getSubArrayIndex applying
    const int ews = target.ews();
    const int targetLen = target.lengthOf();
    auto stream = getContext()->getCudaStream();

    prepareSpecialUse({&target}, {this});
    BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), tileKernelHH, (getSpecialBuffer(), getSpecialShapeInfo(), target.getSpecialBuffer(), target.getSpecialShapeInfo(), targetLen, ews, stream), LIBND4J_TYPES, LIBND4J_TYPES);
    registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::tile(NDArray& target) const {
    if(rankOf() > target.rankOf())
        throw std::runtime_error("NDArray::tile method - rank of target array must be bigger or equal to the rank of this array !");

    if(!ShapeUtils::areShapesBroadcastable(*this, target))
        throw std::runtime_error("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");

    // fill newBuff, loop through all elements of newBuff
    // looping through getBuffer() goes automatically by means of getSubArrayIndex applying
    const auto ews = target.ews();
    const auto targetLen = target.lengthOf();
    auto stream = getContext()->getCudaStream();

    prepareSpecialUse({&target}, {this});
    BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), tileKernelHH, (getSpecialBuffer(), getSpecialShapeInfo(), target.getSpecialBuffer(), target.getSpecialShapeInfo(), targetLen, ews, stream), LIBND4J_TYPES, LIBND4J_TYPES);
    registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
// create new  array by repeating it the number of times given by reps
NDArray* NDArray::repeat(int dimension, const std::vector<Nd4jLong>& repeats) const {
    auto outShape = ShapeUtils::evalRepeatShape(dimension, repeats, *this);

    // the size of outShape == rank
    int rank = rankOf();            // = outShape.size()

    std::vector<Nd4jLong> newShape(rank);
    for (int i = 0; i < rank; i++)
        newShape[i] = outShape[i];

    auto ret = new NDArray('c', outShape, dataType(),  getContext());

    auto repeatDelta = shape::prodLong(newShape.data(), rank) / this->lengthOf();
    std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rankOf(), {dimension});
    const Nd4jLong numTads = ShapeUtils::getNumOfSubArrs(getShapeInfo(), dimsToExclude); //this->tensorsAlongDimension({dimension});
    std::vector<int> copy({dimension});

    auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);
    auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(ret->getShapeInfo(), copy);

    prepareSpecialUse({ret}, {this});
    auto stream = getContext()->getCudaStream();
    BUILD_SINGLE_SELECTOR(dataType(), repeatKernelH, (getSpecialBuffer(), ret->getSpecialBuffer(), numTads, lengthOf(), ret->lengthOf(), packX.platformShapeInfo(), packX.platformOffsets(), packZ.platformShapeInfo(), packZ.platformOffsets(), *stream), LIBND4J_TYPES);
    registerSpecialUse({ret}, {this});

    return ret;
}

//////////////////////////////////////////////////////////////////////////
// fill array by repeating it the number of times given by reps
void NDArray::repeat(int dimension, NDArray& target) const {

    if(dimension < 0)
        dimension += rankOf();

    if(rankOf() != target.rankOf())
        throw std::invalid_argument("NDArray::repeat(int dimension, NDArray& target) method: wrong rank of target array it must be equal to this array rank!");

    Nd4jLong repeatDelta = target.sizeAt(dimension) / sizeAt(dimension);

    if(repeatDelta == 0)
        throw std::invalid_argument("NDArray::repeat(int dimension, NDArray& target) method: wrong shape of target array!");


    std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(rankOf(), {dimension});
    const Nd4jLong numTads = ShapeUtils::getNumOfSubArrs(getShapeInfo(), dimsToExclude);

    std::vector<int> copy({dimension});
    auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(this->getShapeInfo(), copy);
    auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(target.getShapeInfo(), copy);

    NDArray::prepareSpecialUse({&target}, {this});
    auto stream = getContext()->getCudaStream();
    BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), repeatKernelHH, (getSpecialBuffer(), target.getSpecialBuffer(), numTads, lengthOf(), packX.platformShapeInfo(), packX.platformOffsets(), packZ.platformShapeInfo(), packZ.platformOffsets(), *stream), LIBND4J_TYPES, LIBND4J_TYPES);
    NDArray::registerSpecialUse({&target}, {this});
}

////////////////////////////////////////////////////////////////////////
void* NDArray::specialBuffer() {

    if (_buffer->special() == nullptr)
        return getBuffer();
    // FIXME: this should be fixed once CUDA backend added
    return static_cast<int8_t*>(_buffer->special()) + (_offset * sizeOfT());
}

////////////////////////////////////////////////////////////////////////
void* NDArray::getSpecialBuffer() const {
    if (_buffer->special() == nullptr)
        return getBuffer();
    // FIXME: this should be fixed once CUDA backend added
    return static_cast<int8_t*>(_buffer->special()) + (_offset * sizeOfT());
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray::printCurrentBuffer(const bool host, const char* msg, const int precision) const {

    if(_length == 0)
            { printf("NDArray::printActualBuffer: array length is zero !\n"); return; }

    if(msg)
        printf("%s", msg);

    if(host) {
        if(getBuffer() == nullptr || _length == 0)
            { printf("NDArray::printActualBuffer: host buffer is nullptr !\n"); return; }

        const T* buff = bufferAsT<T>();
        for (uint i = 0; i < _length; i++)
            printf("%.*f, ", precision, (double)buff[getOffset(i)]);
        printf("\n");
    }
    else {
        if(getSpecialBuffer() == nullptr || _length == 0)
            { printf("NDArray::printSpecialBuffer: special buffer is nullptr !\n"); return; }

        void* pHost = operator new(sizeof(T) * _length);

        if (ews() != 1) {
            for (uint i = 0; i < _length; i++)
                cudaMemcpyAsync(reinterpret_cast<T*>(pHost) + i, specialBufferWithOffset(i), sizeof(T), cudaMemcpyDeviceToHost, *(getContext()->getCudaStream()));
        }
        else
            cudaMemcpyAsync(pHost, getSpecialBuffer(), sizeOfT() * _length, cudaMemcpyDeviceToHost, *getContext()->getCudaStream());

        cudaError_t cudaResult = cudaStreamSynchronize(*getContext()->getCudaStream());
        if(cudaResult != 0)
            throw std::runtime_error("NDArray::printSpecialBuffer: cudaStreamSynchronize failed!");

        for (uint i = 0; i < _length; i++)
            printf("%.*f, ", precision, (double)reinterpret_cast<T*>(pHost)[i]);
        printf("\n");

        operator delete(pHost);
    }
}
template void NDArray::printCurrentBuffer<int>(const bool host,const char* msg, const int precision) const;
template void NDArray::printCurrentBuffer<float>(const bool host, const char* msg, const int precision) const;
template void NDArray::printCurrentBuffer<double>(const bool host, const char* msg, const int precision) const;


#if defined(__CUDACC__) && !defined(BUILD_TESTS)

#include <cpu/NDArrayLambda.hpp>

#endif

} // end namespace nd4j
#endif

