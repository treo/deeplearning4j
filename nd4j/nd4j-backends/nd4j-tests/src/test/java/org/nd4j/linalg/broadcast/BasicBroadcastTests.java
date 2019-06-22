/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.linalg.broadcast;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.RealDivOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class BasicBroadcastTests extends BaseNd4jTest {
    public BasicBroadcastTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void basicBroadcastTest_1() {
        val x = Nd4j.create(DataType.FLOAT, 3, 5);
        val y = Nd4j.createFromArray(new float[]{1.f, 1.f, 1.f, 1.f, 1.f});
        val e = Nd4j.create(DataType.FLOAT, 3, 5).assign(1.f);

        // inplace setup
        val op = new AddOp(new INDArray[]{x, y}, new INDArray[]{x});

        Nd4j.exec(op);

        assertEquals(e, x);
    }

    @Test
    public void basicBroadcastTest_2() {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2);
        val y = Nd4j.createFromArray(new float[]{1.f, 1.f, 1.f, 1.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.FLOAT, 3, 2, 2).assign(1.f);

        val z = x.add(y);

        assertEquals(e, z);
    }


    @Test
    public void basicBroadcastTest_3() {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(1);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.FLOAT, 3, 2, 2).assign(2.f);

        val z = x.mul(y);

        assertEquals(e, z);
    }

    @Test
    public void basicBroadcastTest_4() {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.FLOAT, 3, 2, 2).assign(2.f);

        val z = x.div(y);

        assertEquals(e, z);
    }

    @Test
    public void basicBroadcastTest_5() {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.FLOAT, 3, 2, 2).assign(2.f);

        val z = x.sub(y);

        assertEquals(e, z);
    }

    @Test
    public void basicBroadcastTest_6() {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.FLOAT, 3, 2, 2).assign(-2.f);

        val z = x.rsub(y);

        assertEquals(e, z);
    }

    @Test
    public void basicBroadcastTest_7() {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.BOOL, 3, 2, 2).assign(false);

        val z = x.lt(y);

        assertEquals(e, z);
    }

    @Test(expected = IllegalArgumentException.class)
    public void basicBroadcastFailureTest_1() {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val z = x.subi(y);
    }

    @Test(expected = IllegalArgumentException.class)
    public void basicBroadcastFailureTest_2() {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val z = x.divi(y);
    }

    @Test(expected = IllegalArgumentException.class)
    public void basicBroadcastFailureTest_3() {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val z = x.muli(y);
    }

    @Test(expected = IllegalArgumentException.class)
    public void basicBroadcastFailureTest_4() {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val z = x.addi(y);
    }

    @Test(expected = IllegalArgumentException.class)
    public void basicBroadcastFailureTest_5() {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val z = x.rsubi(y);
    }

    @Test(expected = IllegalArgumentException.class)
    public void basicBroadcastFailureTest_6() {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val z = x.rdivi(y);
    }

    @Test
    public void basicBroadcastTest_8() {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(4.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.BOOL, 3, 2, 2).assign(true);

        val z = x.gt(y);

        assertEquals(e, z);
    }

    @Test
    public void basicBroadcastTest_9() {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(2.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.BOOL, 3, 2, 2).assign(true);

        val z = x.eq(y);

        assertEquals(e, z);
    }

    @Test
    public void basicBroadcastTest_10() {
        val x = Nd4j.create(DataType.FLOAT, 3, 1, 2).assign(1.f);
        val y = Nd4j.createFromArray(new float[]{2.f, 2.f, 2.f, 2.f}).reshape(2, 2);
        val e = Nd4j.create(DataType.BOOL, 3, 2, 2).assign(false);

        val z = x.eq(y);

        assertEquals(e, z);
    }

    @Test
    public void emptyBroadcastTest_1() {
        val x = Nd4j.create(DataType.FLOAT, 1, 2);
        val y = Nd4j.create(DataType.FLOAT, 0, 2);

        val z = x.add(y);
        assertEquals(y, z);
    }

    @Test(expected = IllegalArgumentException.class)
    public void emptyBroadcastTest_2() {
        val x = Nd4j.create(DataType.FLOAT, 1, 2);
        val y = Nd4j.create(DataType.FLOAT, 0, 2);

        val z = x.addi(y);
        assertEquals(y, z);
    }

    @Test
    public void emptyBroadcastTest_3() {
        val x = Nd4j.create(DataType.FLOAT, 1, 0, 1);
        val y = Nd4j.create(DataType.FLOAT, 1, 0, 2);

        val op = new RealDivOp(new INDArray[]{x, y}, new INDArray[]{});
        val z = Nd4j.exec(op)[0];

        assertEquals(y, z);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
