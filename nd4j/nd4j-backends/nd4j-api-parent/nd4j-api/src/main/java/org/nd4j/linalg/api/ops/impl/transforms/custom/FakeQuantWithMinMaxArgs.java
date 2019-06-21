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

public class FakeQuantWithMinMaxArgs extends DynamicCustomOp {

    protected boolean narrowRange;
    protected int numBits;
    protected float min;
    protected float max;

    public FakeQuantWithMinMaxArgs(SameDiff sd, SDVariable input, float min, float max, boolean narrowRange, int numBits){
        super(sd, input);
        this.narrowRange = narrowRange;
        this.numBits = numBits;
        this.min = min;
        this.max = max;
        addArgs();
    }

    public FakeQuantWithMinMaxArgs(){ }

    protected void addArgs(){
        iArguments.clear();
        addIArgument(numBits, narrowRange ? 1 : 0);
        addTArgument(min, max);
    }

    @Override
    public String opName(){
        return "fake_quant_with_min_max_args";
    }

    @Override
    public String tensorflowName(){
        return "FakeQuantWithMinMaxArgs";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        if(attributesForNode.containsKey("narrow_range")){
            this.narrowRange = attributesForNode.get("narrow_range").getB();
        }
        this.numBits = (int)attributesForNode.get("num_bits").getI();
        this.min = attributesForNode.get("min").getF();
        this.max = attributesForNode.get("max").getF();
        addArgs();
    }


    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 1, "Expected exactly 1 input, got %s", inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients){
        return Arrays.asList(sameDiff.zerosLike(arg(0)), sameDiff.zerosLike(arg(1)), sameDiff.zerosLike(arg(2)));
    }
}
