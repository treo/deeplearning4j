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
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/legacy_helpers.h>
#include <NDArrayFactory.h>
#include <op_boilerplate.h>

namespace nd4j {
namespace ops {
namespace helpers {
    template <typename T>
    linkage void reluDerivative__(NDArray* theFirst, NDArray* theSecond) {
        auto functor = LAMBDA_TT(x, y){
            return x > (T) 0.f ? y : T(0.f);
        };

        theFirst->applyPairwiseLambda(theSecond, functor, nullptr);
    }
    BUILD_SINGLE_TEMPLATE(template void reluDerivative__, (NDArray* input, NDArray* epsilon), FLOAT_TYPES);

    void reluDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), reluDerivative__, (theFirst, theSecond), FLOAT_TYPES);
    }

    template <typename T>
    linkage void reluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return x > (T)0.f ? y : T(0.f);
        };

        input->applyPairwiseLambda(epsilon, functor, output);
    }
    BUILD_SINGLE_TEMPLATE(template void reluDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void reluDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), reluDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    linkage void relu6Derivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return x > (T)0.f && x < (T)6.f? y : T(0.f);
        };

        input->applyPairwiseLambda(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void relu6Derivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void relu6Derivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), relu6Derivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    linkage void leakyReluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return x >= (T)0.f? T(1.f) : T(0.f);
        };

        input->applyPairwiseLambda(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void leakyReluDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void leakyReluDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), leakyReluDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    linkage void eluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return y * nd4j::math::nd4j_eluderivative<T,T>(x);
        };

        input->applyPairwiseLambda(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void eluDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void eluDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), eluDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    linkage void seluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return y * simdOps::SELUDerivative<T>::op(x, nullptr);
        };

        input->applyPairwiseLambda(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void seluDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void seluDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), seluDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    linkage void cubeDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return y * (3 * x * x);
        };

        input->applyPairwiseLambda(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void cubeDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void cubeDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), cubeDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    //return (x >= X(0.f) ? y: -y);
    template <typename T>
    linkage void reduceNorm1_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return x > T(0.f)? y : -y;
        };

        input->applyPairwiseLambda(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void reduceNorm1_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void reduceNorm1(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), reduceNorm1_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    linkage void sigmCrossEntropy_(NDArray* logits, NDArray* labels, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return nd4j::math::nd4j_max<T>(x, (T)0.f) - x * y + nd4j::math::nd4j_log<T,T>((T)1.f + nd4j::math::nd4j_exp<T,T>(-nd4j::math::nd4j_abs(x)));
        };

        logits->applyPairwiseLambda(labels, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void sigmCrossEntropy_, (NDArray* logits, NDArray* labels, NDArray* output);, FLOAT_TYPES);

    void sigmCrossEntropy(nd4j::LaunchContext * context, NDArray* logits, NDArray* labels, NDArray* output) {
        BUILD_SINGLE_SELECTOR(logits->dataType(), sigmCrossEntropy_, (logits, labels, output), FLOAT_TYPES);
    }

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    linkage void sigmCrossEntropyGrad_(NDArray* logits, NDArray* labels, NDArray* output) {
        // 1 - labels - 1 / (1 + exp(logits))
        auto functor = LAMBDA_TT(x, y) {
            if(x <= 0)
                return static_cast<T>(1.) - y - static_cast<T>(1.) / (static_cast<T>(1.) + nd4j::math::nd4j_exp<T,T>(x));
            auto e = nd4j::math::nd4j_exp<T,T>(-x);
            return static_cast<T>(1.) - y - e / (static_cast<T>(1.) + e);
        };

        logits->applyPairwiseLambda(labels, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void sigmCrossEntropyGrad_, (NDArray* logits, NDArray* labels, NDArray*output);, FLOAT_TYPES);

    void sigmCrossEntropyGrad(nd4j::LaunchContext * context, NDArray* logits, NDArray* labels, NDArray* output) {
        BUILD_SINGLE_SELECTOR(logits->dataType(), sigmCrossEntropyGrad_, (logits, labels, output), FLOAT_TYPES);
    }

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    linkage void tanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            T th = nd4j::math::nd4j_tanh<T,T>(x);
            return y * ((T)1.0f - (th * th));
        };

        input->applyPairwiseLambda(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void tanhDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void tanhDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), tanhDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    // return static_cast<X>(d2) * simdOps::HardTanhDerivative<X>::op(d1, nullptr);
    template <typename T>
    linkage void hardTanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            T th = nd4j::math::nd4j_tanh<T,T>(x);
            return y * simdOps::HardTanhDerivative<T>::op(x, nullptr);
        };

        input->applyPairwiseLambda(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void hardTanhDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void hardTanhDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), hardTanhDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    linkage void rationalTanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return y * simdOps::RationalTanhDerivative<T>::op(x, nullptr);
        };

        input->applyPairwiseLambda(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void rationalTanhDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void rationalTanhDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), rationalTanhDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    linkage void rectifiedTanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return x > (T) 0.0f ? y * (nd4j::math::nd4j_tanhderivative<T,T>(x)) : (T) 0.0f;
        };

        input->applyPairwiseLambda(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void rectifiedTanhDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void rectifiedTanhDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), rectifiedTanhDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    //            X f = (X) 1.0f + nd4j::math::nd4j_abs<X>(d1);
    //            return (X) d2 * ((X) 1.0f / (f * f));

    template <typename T>
    linkage void softSignDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            T ss = (T)1.f + nd4j::math::nd4j_abs<T>(x);
            return y * ((T) 1.0f  / (ss * ss));
        };

        input->applyPairwiseLambda(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void softSignDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void softSignDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), softSignDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    linkage void softPlusDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            T p = nd4j::math::nd4j_pow<T, T, T>(static_cast<T>(M_E), x);
            return y * (p / (p + 1.));
        };

        input->applyPairwiseLambda(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void softPlusDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void softPlusDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), softPlusDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }
///
/// \param theFirst
/// \param theSecond
/// \param theOutput
    template <typename T>
    linkage void sigmoidDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            T s = nd4j::math::nd4j_sigmoid<T,T>(x);
            return y * (s * ((T) 1.0f - s));
        };

        input->applyPairwiseLambda(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void sigmoidDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void sigmoidDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), sigmoidDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    linkage void hardSigmoidDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return y * simdOps::HardSigmoidDerivative<T>::op(x, nullptr);
        };

        input->applyPairwiseLambda(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void hardSigmoidDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void hardSigmoidDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), hardSigmoidDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    linkage void logSumExp_(NDArray* input, NDArray* axis, NDArray* output) {
        // reduce along axis with
        std::unique_ptr<NDArray> tempInput(input->dup());
        input->applyTransform(transform::Exp, tempInput.get());
        std::vector<int> axisVector;
        if (axis != nullptr) {
            axisVector.resize(axis->lengthOf());
            for (size_t i = 0; i < axisVector.size(); ++i)
                axisVector[i] = axis->e<int>(i);
        }
        tempInput->reduceAlongDimension(reduce::Sum, output, axisVector);
        output->applyTransform(transform::Log, nullptr, nullptr);
    }

    template <typename T>
    linkage void logSumExp_(NDArray* input, NDArray* subtrah, NDArray* axis, NDArray* output) {
        // reduce along axis with
        std::unique_ptr<NDArray> tempInput(input->dup());
        input->applyPairwiseTransform(pairwise::Subtract, subtrah, tempInput.get());
        tempInput->applyTransform(transform::Exp, nullptr, nullptr);

        std::vector<int> axisVector;
        if (axis != nullptr) {
            axisVector.resize(axis->lengthOf());
            for (size_t i = 0; i < axisVector.size(); ++i)
                axisVector[i] = axis->e<int>(i);
        }
        tempInput->reduceAlongDimension(reduce::Sum, output, axisVector);
        output->applyTransform(transform::Log, nullptr, nullptr);
    }

    void logSumExp(nd4j::LaunchContext * context, NDArray* input, NDArray* axis, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), logSumExp_, (input, axis, output), FLOAT_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void logSumExp_, (NDArray* input, NDArray* axis, NDArray*output);, FLOAT_TYPES);

    void logSumExp(nd4j::LaunchContext * context, NDArray* input, NDArray* subtrah, NDArray* axis, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), logSumExp_, (input, subtrah, axis, output), FLOAT_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void logSumExp_, (NDArray* input, NDArray* subtrah, NDArray* axis, NDArray*output);, FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void weightedCrossEntropyWithLogitsFunctor_(NDArray const* targets, NDArray const* input, NDArray const* weights, NDArray* output) {

}

void weightedCrossEntropyWithLogitsFunctor(nd4j::LaunchContext * context, NDArray const* targets, NDArray const* input, NDArray const* weights, NDArray* output) {
    BUILD_SINGLE_SELECTOR(targets->dataType(), weightedCrossEntropyWithLogitsFunctor_, (targets, input, weights, output), FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE(template void weightedCrossEntropyWithLogitsFunctor_, (NDArray const* targets, NDArray const* input, NDArray const* weights, NDArray* output), FLOAT_TYPES);

}
}
}