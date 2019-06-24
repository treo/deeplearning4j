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

package org.deeplearning4j.datasets.iterator;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This iterator virtually splits given MultiDataSetIterator into Train and Test parts.
 * I.e. you have 100000 examples. Your batch size is 32. That means you have 3125 total batches. With split ratio of 0.7 that will give you 2187 training batches, and 938 test batches.
 *
 * PLEASE NOTE: You can't use Test iterator twice in a row. Train iterator should be used before Test iterator use.<br>
 * PLEASE NOTE: You can't use this iterator, if underlying iterator uses randomization/shuffle between epochs.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class DataSetIteratorSplitter {
    protected DataSetIterator backedIterator;
    protected final long totalExamples;
    protected final double ratio;
    protected final double[] ratios;
    protected final long numTrain;
    protected final long numTest;
    protected final long numArbitrarySets;
    protected final int[] splits;


    protected AtomicLong counter = new AtomicLong(0);

    protected AtomicBoolean resetPending = new AtomicBoolean(false);
    protected DataSet firstTrain = null;

    protected int partNumber = 0;

    /**
     * The only constructor
     *
     * @param baseIterator - iterator to be wrapped and split
     * @param totalBatches - total batches in baseIterator
     * @param ratio - train/test split ratio
     */
    public DataSetIteratorSplitter(@NonNull DataSetIterator baseIterator, long totalBatches, double ratio) {
        if (!(ratio > 0.0 && ratio < 1.0))
            throw new ND4JIllegalStateException("Ratio value should be in range of 0.0 > X < 1.0");

        if (totalBatches < 0)
            throw new ND4JIllegalStateException("totalExamples number should be positive value");

        if (!baseIterator.resetSupported())
            throw new ND4JIllegalStateException("Underlying iterator doesn't support reset, so it can't be used for runtime-split");


        this.backedIterator = baseIterator;
        this.totalExamples = totalBatches;
        this.ratio = ratio;
        this.ratios = null;
        this.numTrain = (long) (totalExamples * ratio);
        this.numTest = totalExamples - numTrain;
        this.numArbitrarySets = 2;
        this.splits = null;

        log.warn("IteratorSplitter is used: please ensure you don't use randomization/shuffle in underlying iterator!");
    }

    public DataSetIteratorSplitter(@NonNull DataSetIterator baseIterator, long totalBatches, double[] ratios) {
        for (double ratio : ratios) {
            if (!(ratio > 0.0 && ratio < 1.0))
                throw new ND4JIllegalStateException("Ratio value should be in range of 0.0 > X < 1.0");
        }

        if (totalBatches < 0)
            throw new ND4JIllegalStateException("totalExamples number should be positive value");

        if (!baseIterator.resetSupported())
            throw new ND4JIllegalStateException("Underlying iterator doesn't support reset, so it can't be used for runtime-split");


        this.backedIterator = baseIterator;
        this.totalExamples = totalBatches;
        this.ratio = 0.0;
        this.ratios = ratios;
        this.numTrain = 0; //(long) (totalExamples * ratio);
        this.numTest = 0; //totalExamples - numTrain;
        this.numArbitrarySets = ratios.length;
        this.splits = null;

        log.warn("IteratorSplitter is used: please ensure you don't use randomization/shuffle in underlying iterator!");
    }

    public DataSetIteratorSplitter(@NonNull DataSetIterator baseIterator, long totalBatches, int[] splits) {

        /*if (!(simpleRatio > 0.0 && simpleRatio < 1.0))
           throw new ND4JIllegalStateException("Ratio value should be in range of 0.0 > X < 1.0");*/

        if (totalBatches < 0)
            throw new ND4JIllegalStateException("totalExamples number should be positive value");

        if (!baseIterator.resetSupported())
            throw new ND4JIllegalStateException("Underlying iterator doesn't support reset, so it can't be used for runtime-split");


        this.backedIterator = baseIterator;
        this.totalExamples = totalBatches;
        this.ratio = 0.0;
        this.ratios = null;

        this.numTrain = 0; //(long) (totalExamples * ratio);
        this.numTest = 0; //totalExamples - numTrain;
        this.splits = splits;
        this.numArbitrarySets = splits.length;

        log.warn("IteratorSplitter is used: please ensure you don't use randomization/shuffle in underlying iterator!");
    }

    public static class DataSetIterators {
        private List<DataSetIterator> iterators = new ArrayList<>();
        private int currentIndex = 0;

        public void add(DataSetIterator iterator) {
            iterators.add(iterator);
        }

        public DataSetIterator get(int i) {
            this.currentIndex = i;
            return iterators.get(i);
        }

        public int getCurrentIndex() {
            return currentIndex;
        }

        public List<DataSetIterator> asList() {
            return iterators;
        }
    }

    public static class ScrollableDataSetIterator implements DataSetIterator {
        private int thisPart = 0;
        protected DataSetIterator backedIterator;
        protected AtomicLong counter = new AtomicLong(0);

        protected AtomicBoolean resetPending = new AtomicBoolean(false);
        protected DataSet firstTrain = null;
        private double ratio;
        private long totalExamples;
        private long itemsPerPart;


        public ScrollableDataSetIterator(int num, DataSetIterator backedIterator, AtomicLong counter,
                                         AtomicBoolean resetPending, DataSet firstTrain, double ratio,
                                         int totalExamples) {
            this.thisPart = num;
            this.backedIterator = backedIterator;
            this.counter = counter;
            this.resetPending = resetPending;
            this.firstTrain = firstTrain;
            this.ratio = ratio;
            this.totalExamples = totalExamples;
            this.itemsPerPart = (long)(totalExamples * ratio);
        }

        public ScrollableDataSetIterator(int num, DataSetIterator backedIterator, AtomicLong counter,
                                         AtomicBoolean resetPending, DataSet firstTrain,
                                         int itemsPerPart) {
            this.thisPart = num;
            this.backedIterator = backedIterator;
            this.counter = counter;
            this.resetPending = resetPending;
            this.firstTrain = firstTrain;
            //this.totalExamples = totalExamples;
            this.itemsPerPart = itemsPerPart;
        }

        @Override
        public DataSet next(int i) {

            //throw new UnsupportedOperationException();
            if (i == 0) {
                backedIterator.reset();
                if (backedIterator.hasNext())
                    return backedIterator.next();
            }
            else {
                int cnt = 0;
                DataSet retVal = null;
                int top = (int)((thisPart) * itemsPerPart + i);
                for (; cnt <= top; ++cnt) {
                    counter.incrementAndGet();
                    val state = backedIterator.hasNext();
                    if (state /*&& counter.get() < ratio * totalExamples*/)
                        retVal = backedIterator.next();
                    else
                        break;
                }
                if (cnt == top+1)
                    return retVal;
            }
            return null;
        }

        @Override
        public List<String> getLabels() {
            return backedIterator.getLabels();
        }

        @Override
        public int inputColumns() {
            return backedIterator.inputColumns();
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }

        @Override
        public int totalOutcomes() {
            return backedIterator.totalOutcomes();
        }

        @Override
        public boolean resetSupported() {
            return backedIterator.resetSupported();
        }

        @Override
        public boolean asyncSupported() {
            return backedIterator.asyncSupported();
        }

        @Override
        public void reset() {
            resetPending.set(true);
        }

        @Override
        public int batch() {
            return backedIterator.batch();
        }

        @Override
        public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
            backedIterator.setPreProcessor(dataSetPreProcessor);
        }

        @Override
        public DataSetPreProcessor getPreProcessor() {
            return backedIterator.getPreProcessor();
        }


        @Override
        public boolean hasNext() {
            if (resetPending.get()) {
                if (resetSupported()) {
                    backedIterator.reset();
                    counter.set(0);
                    resetPending.set(false);
                } else
                    throw new UnsupportedOperationException("Reset isn't supported by underlying iterator");
            }

            boolean state = false;
            for (int i = 0; i <= thisPart * itemsPerPart; ++i) {
                state = backedIterator.hasNext();
                if (!state)
                    break;
            }
            if (state && counter.get() < itemsPerPart)
                return true;
            else
                return false;

        }

        @Override
        public DataSet next() {
            counter.incrementAndGet();
            val p = backedIterator.next();

            if (counter.get() == 1 && firstTrain == null) {
                // first epoch ever, we'll save first dataset and will use it to check for equality later
                firstTrain = p.copy();
                firstTrain.detach();
            } else if (counter.get() == 1) {
                // epoch > 1, comparing first dataset to previously stored dataset. they should be equal
                int cnt = 0;
                if (!p.getFeatures().equalsWithEps(firstTrain.getFeatures(), 1e-5))
                    throw new ND4JIllegalStateException("First examples do not match. Randomization was used?");
            }

            return p;
        }
    }

    public DataSetIterators getIterators() {
        DataSetIterators retVal = new DataSetIterators();
        int partN = 0;
        if (ratios == null) {
            for (final int split : splits) {
                ScrollableDataSetIterator partIterator =
                        new ScrollableDataSetIterator(partN++, backedIterator, counter, resetPending, firstTrain, split);
                retVal.add(partIterator);
            }
        }
        else
        for (final double ratio : ratios) {
            ScrollableDataSetIterator partIterator =
                    new ScrollableDataSetIterator(partN++, backedIterator, counter, resetPending, firstTrain, ratio, (int)totalExamples);
            retVal.add(partIterator);
        }
        return retVal;
    }


    /**
     * This method returns train iterator instance
     *
     * @return
     */
    public DataSetIterator getTrainIterator() {
        return new DataSetIterator() {
            @Override
            public DataSet next(int i) {
                throw new UnsupportedOperationException();
            }

            @Override
            public List<String> getLabels() {
                return backedIterator.getLabels();
            }

            @Override
            public int inputColumns() {
                return backedIterator.inputColumns();
            }

            @Override
            public void remove() {
                throw new UnsupportedOperationException();
            }

            @Override
            public int totalOutcomes() {
                return backedIterator.totalOutcomes();
            }

            @Override
            public boolean resetSupported() {
                return backedIterator.resetSupported();
            }

            @Override
            public boolean asyncSupported() {
                return backedIterator.asyncSupported();
            }

            @Override
            public void reset() {
                resetPending.set(true);
            }

            @Override
            public int batch() {
                return backedIterator.batch();
            }

            @Override
            public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
                backedIterator.setPreProcessor(dataSetPreProcessor);
            }

            @Override
            public DataSetPreProcessor getPreProcessor() {
                return backedIterator.getPreProcessor();
            }


            @Override
            public boolean hasNext() {
                if (resetPending.get()) {
                    if (resetSupported()) {
                        backedIterator.reset();
                        counter.set(0);
                        resetPending.set(false);
                    } else
                        throw new UnsupportedOperationException("Reset isn't supported by underlying iterator");
                }

                val state = backedIterator.hasNext();
                if (state && counter.get() < numTrain)
                    return true;
                else
                    return false;
            }

            @Override
            public DataSet next() {
                counter.incrementAndGet();
                val p = backedIterator.next();

                if (counter.get() == 1 && firstTrain == null) {
                    // first epoch ever, we'll save first dataset and will use it to check for equality later
                    firstTrain =  p.copy();
                    firstTrain.detach();
                } else if (counter.get() == 1) {
                    // epoch > 1, comparing first dataset to previously stored dataset. they should be equal
                    int cnt = 0;
                    if (!p.getFeatures().equalsWithEps(firstTrain.getFeatures(), 1e-5))
                        throw new ND4JIllegalStateException("First examples do not match. Randomization was used?");
                }

                return p;
            }
        };
    }

    /**
     * This method returns test iterator instance
     *
     * @return
     */
    public DataSetIterator getTestIterator() {
        return new DataSetIterator() {
            @Override
            public DataSet next(int i) {
                throw new UnsupportedOperationException();
            }

            @Override
            public List<String> getLabels() {
                return backedIterator.getLabels();
            }

            @Override
            public int inputColumns() {
                return backedIterator.inputColumns();
            }

            @Override
            public void remove() {
                throw new UnsupportedOperationException();
            }

            @Override
            public int totalOutcomes() {
                return backedIterator.totalOutcomes();
            }

            @Override
            public boolean resetSupported() {
                return backedIterator.resetSupported();
            }

            @Override
            public boolean asyncSupported() {
                return backedIterator.asyncSupported();
            }

            @Override
            public void reset() {
                resetPending.set(true);
            }

            @Override
            public int batch() {
                return backedIterator.batch();
            }

            @Override
            public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
                backedIterator.setPreProcessor(dataSetPreProcessor);
            }

            @Override
            public DataSetPreProcessor getPreProcessor() {
                return backedIterator.getPreProcessor();
            }


            @Override
            public boolean hasNext() {
                val state = backedIterator.hasNext();
                if (state && counter.get() < numTrain + numTest)
                    return true;
                else
                    return false;
            }

            @Override
            public DataSet next() {
                counter.incrementAndGet();
                return backedIterator.next();
            }
        };
    }
}
