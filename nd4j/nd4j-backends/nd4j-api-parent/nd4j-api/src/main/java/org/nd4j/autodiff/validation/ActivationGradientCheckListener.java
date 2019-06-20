package org.nd4j.autodiff.validation;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * A listener used for debugging and testing purposes - specifically for gradient checking activations internally in
 * {@link GradCheckUtil}. It probably isn't useful for anything outside of this.
 *
 * @author Alex Black
 */
@NoArgsConstructor
public class ActivationGradientCheckListener extends BaseListener {

    @Getter @Setter
    private String variableName;
    @Getter @Setter
    private long[] idx;
    @Getter @Setter
    private double eps;

    @Override
    public void opExecution(SameDiff sd, At at, boolean training, SameDiffOp op, INDArray[] outputs) {
        Preconditions.checkState(variableName != null, "No variable name has been set yet. Variable name must be set before using this listener");
        Preconditions.checkState(eps != 0.0, "Epsilon has not been set");


        List<String> outs = op.getOutputsOfOp();
        int i = 0;
        for(String s : outs){
            if(variableName.equals(s)){
                Preconditions.checkState(idx != null || outputs[i].isScalar(),
                        "No index to modify has been set yet. Index must be set before using this listener");

                double orig = outputs[i].getDouble(idx);
                outputs[i].putScalar(idx, orig + eps);
                return;
            }
            i++;
        }
    }

}
