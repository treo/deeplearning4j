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
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/image_suppression.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static __global__ void nonMaxSuppressionV2Kernel() {

    }

    template <typename T>
    static __device__ bool needToSuppressWithThreshold(T* boxes, Nd4jLong* boxesShape, int previousIndex, int nextIndex, T threshold) {
        Nd4jLong previous0[] = {previousIndex, 0};
        Nd4jLong previous1[] = {previousIndex, 1};
        Nd4jLong previous2[] = {previousIndex, 2};
        Nd4jLong previous3[] = {previousIndex, 3};
        Nd4jLong next0[] = {nextIndex, 0};
        Nd4jLong next1[] = {nextIndex, 1};
        Nd4jLong next2[] = {nextIndex, 2};
        Nd4jLong next3[] = {nextIndex, 3};

        T minYPrev = nd4j::math::nd4j_min(boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), previous0, 2)], boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), previous2, 2)]);
        T minXPrev = nd4j::math::nd4j_min(boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), previous1, 2)], boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), previous3, 2)]);
        T maxYPrev = nd4j::math::nd4j_max(boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), previous0, 2)], boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), previous2, 2)]);
        T maxXPrev = nd4j::math::nd4j_max(boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), previous1, 2)], boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), previous3, 2)]);
        T minYNext = nd4j::math::nd4j_min(boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), next0, 2)], boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), next2, 2)]);
        T minXNext = nd4j::math::nd4j_min(boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), next1, 2)], boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), next3, 2)]);
        T maxYNext = nd4j::math::nd4j_max(boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), next0, 2)], boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), next2, 2)]);
        T maxXNext = nd4j::math::nd4j_max(boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), next1, 2)], boxes[shape::getOffset(0, shape::shapeOf(boxesShape), shape::stride(boxesShape), next3, 2)]);

        T areaPrev = (maxYPrev - minYPrev) * (maxXPrev - minXPrev);
        T areaNext = (maxYNext - minYNext) * (maxXNext - minXNext);

        if (areaNext <= T(0.f) || areaPrev <= T(0.f)) return false;

        T minIntersectionY = nd4j::math::nd4j_max(minYPrev, minYNext);
        T minIntersectionX = nd4j::math::nd4j_max(minXPrev, minXNext);
        T maxIntersectionY = nd4j::math::nd4j_min(maxYPrev, maxYNext);
        T maxIntersectionX = nd4j::math::nd4j_min(maxXPrev, maxXNext);
        T intersectionArea =
                nd4j::math::nd4j_max(T(maxIntersectionY - minIntersectionY), T(0.0f)) *
                nd4j::math::nd4j_max(T(maxIntersectionX - minIntersectionX), T(0.0f));
        T intersectionValue = intersectionArea / (areaPrev + areaNext - intersectionArea);
        return intersectionValue > threshold;
    };

    template <typename T>
    static __device__ bool needToSelect(T* boxes, Nd4jLong* boxesShape, Nd4jLong* indices, int* selectedIndices, int current, int numSelected, T threshold) {
        bool shouldSelect = true;
        for (int j = numSelected - 1; j >= 0; --j) {
            if (needToSuppressWithThreshold<T>(boxes, boxesShape, indices[current], indices[selectedIndices[j]], threshold)) {
                shouldSelect = false;
                break;
            }
        }
        return shouldSelect;
    }

    template <typename T>
    static __global__ void nonMaxSuppressionKernel(T* boxes, Nd4jLong* boxesShape, Nd4jLong* indices, int* selected, int* selectedIndices, Nd4jLong numBoxes, T* output, Nd4jLong* outputShape, T threshold) {
        __shared__ bool canContinue;
        __shared__ int numSelected;
        __shared__ Nd4jLong outputLen;

        if (threadIdx.x == 0) {
            canContinue = true;
            numSelected = 0;
            outputLen = shape::length(outputShape);
        }
        __syncthreads();

        auto start = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = blockDim.x * gridDim.x;

        for (int i = start; i < numBoxes && canContinue; i += step) {
            //if (selected.size() >= output->lengthOf()) break;
            bool shouldSelect = needToSelect<T>(boxes, boxesShape, indices, selectedIndices, i, numSelected, threshold);
            // Overlapping boxes are likely to have similar scores,
            // therefore we iterate through the selected boxes backwards.

            if (shouldSelect) {
                selected[numSelected] = indices[i];
                output[numSelected] = indices[i];
                selectedIndices[numSelected++] = i;
            }

            if (numSelected == outputLen) {
                canContinue = false;
                break;
            }
        }
    }

    template <typename T>
    static void nonMaxSuppressionV2_(nd4j::LaunchContext* context, NDArray* boxes, NDArray* scales, int maxSize, double threshold, NDArray* output) {
        NDArray indices = NDArrayFactory::create<Nd4jLong>({scales->lengthOf()});
        indices.linspace(0);
        // TO DO: sort indices using scales as value row
        //std::sort(indices.begin(), indices.end(), [scales](int i, int j) {return scales->e<T>(i) > scales->e<T>(j);});

        NDArray selected = NDArrayFactory::create<int>({output->lengthOf()});

        NDArray selectedIndices = NDArrayFactory::create<int>({output->lengthOf()});
        int numSelected = 0;
        int numBoxes = boxes->sizeAt(0);
        auto stream = context->getCudaStream();
        T* boxesBuf = reinterpret_cast<T*>(boxes->specialBuffer());
        Nd4jLong* indicesData = reinterpret_cast<Nd4jLong*>(indices.specialBuffer());
        int* selectedData = reinterpret_cast<int*>(selected.specialBuffer());
        int* selectedIndicesData = reinterpret_cast<int*>(selectedIndices.specialBuffer());
        T* outputBuf = reinterpret_cast<T*>(output->specialBuffer());
        nonMaxSuppressionKernel<T><<<1, 512, 1024, *stream>>>(boxesBuf, boxes->specialShapeInfo(), indicesData, selectedData, selectedIndicesData, numBoxes, outputBuf, output->specialShapeInfo(), T(threshold));
//        for (int i = 0; i < boxes->sizeAt(0); ++i) {
//            if (selected.size() >= output->lengthOf()) break;
//            bool shouldSelect = true;
//            // Overlapping boxes are likely to have similar scores,
//            // therefore we iterate through the selected boxes backwards.
//            for (int j = numSelected - 1; j >= 0; --j) {
//                if (needToSuppressWithThreshold(*boxes, indices[i], indices[selectedIndices[j]], T(threshold)) {
//                    shouldSelect = false;
//                    break;
//                }
//            }
//            if (shouldSelect) {
//                selected.push_back(indices[i]);
//                selectedIndices[numSelected++] = i;
//            }
//        }
//        for (size_t e = 0; e < selected.size(); ++e)
//            output->p<int>(e, selected[e]);
//
    }

    void nonMaxSuppressionV2(nd4j::LaunchContext * context, NDArray* boxes, NDArray* scales, int maxSize, double threshold, NDArray* output) {
        BUILD_SINGLE_SELECTOR(output->dataType(), nonMaxSuppressionV2_, (context, boxes, scales, maxSize, threshold, output), NUMERIC_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void nonMaxSuppressionV2_, (nd4j::LaunchContext * context, NDArray* boxes, NDArray* scales, int maxSize, double threshold, NDArray* output), NUMERIC_TYPES);

}
}
}