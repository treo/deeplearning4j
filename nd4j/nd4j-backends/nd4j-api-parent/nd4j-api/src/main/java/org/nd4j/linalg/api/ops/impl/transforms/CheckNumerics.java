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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * CheckNumerics op wrapper
 * @author raver119@gmail.com
 */
public class CheckNumerics extends DynamicCustomOp {

    public CheckNumerics(SameDiff sd, SDVariable input, SDVariable message){
        super(sd, new SDVariable[]{input, message});
    }

    public CheckNumerics(){ }

    @Override
    public String opName() {
        return "check_numerics";
    }

    @Override
    public String tensorflowName() {
        return "CheckNumerics";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int numOutputArguments(){
        return 1;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 2, "Expected 2 datatype in, got %s", inputDataTypes);
        Preconditions.checkState(inputDataTypes.get(0).isFPType(), "Input datatype must be a floating point type, got %s", inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
