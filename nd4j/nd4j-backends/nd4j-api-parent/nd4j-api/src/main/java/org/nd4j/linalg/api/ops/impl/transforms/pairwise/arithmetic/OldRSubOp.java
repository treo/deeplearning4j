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

package org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformAnyOp;

import java.util.ArrayList;
import java.util.List;

/**
 * Division operation
 *
 * @author Adam Gibson
 */
public class OldRSubOp extends BaseTransformAnyOp {
    public OldRSubOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public OldRSubOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public OldRSubOp() {}

    public OldRSubOp(INDArray x) {
        super(x);
    }

    public OldRSubOp(INDArray x, INDArray z) {
        super(x, z);
    }

    public OldRSubOp(INDArray x, INDArray y, INDArray z) {
        super(x, y, z);
    }

    @Override
    public int opNum() {
        return 5;
    }

    @Override
    public String opName() {
        return "old_rsub";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable gradWrtX = f().div(i_v.get(0),rarg());
        SDVariable gradWrtY = f().mul(f().neg(gradWrtX),f().div(larg(),rarg()));

        List<SDVariable> ret = new ArrayList<>(2);
        ret.add(gradWrtX);
        ret.add(gradWrtY);
        return ret;
    }


}
