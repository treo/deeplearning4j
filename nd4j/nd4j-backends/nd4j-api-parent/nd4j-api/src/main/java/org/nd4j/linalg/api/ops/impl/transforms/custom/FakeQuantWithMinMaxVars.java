package org.nd4j.linalg.api.ops.impl.transforms.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class FakeQuantWithMinMaxVars extends DynamicCustomOp {

    protected boolean narrowRange;
    protected int numBits;

    public FakeQuantWithMinMaxVars(SameDiff sd, SDVariable input, SDVariable min, SDVariable max, boolean narrowRange, int numBits){
        super(sd, new SDVariable[]{input, min, max});
        this.narrowRange = narrowRange;
        this.numBits = numBits;
        addArgs();
    }

    protected void addArgs(){
        iArguments.clear();
        addIArgument(numBits, narrowRange ? 1 : 0);
    }

    @Override
    public String opName(){
        return "fake_quant_with_min_max_vars";
    }

    @Override
    public String tensorflowName(){
        return "FakeQuantWithMinMaxVars";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        if(attributesForNode.containsKey("narrow_range")){
            this.narrowRange = attributesForNode.get("narrow_range").getB();
        }
        this.numBits = (int)attributesForNode.get("num_bits").getI();
        addArgs();
    }


    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 3, "Expected exactly 3 inputs, got %s", inputDataTypes);
        return Collections.singletonList(DataType.FLOAT);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients){
        return Arrays.asList(sameDiff.zerosLike(arg(0)), sameDiff.zerosLike(arg(1)), sameDiff.zerosLike(arg(2)));
    }
}
