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

package org.nd4j.jita.allocator;

import lombok.extern.slf4j.Slf4j;
import lombok.var;
import org.apache.commons.lang3.RandomUtils;
import org.bytedeco.javacpp.Pointer;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.context.impl.LimitedContextPool;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.impl.MemoryTracker;

import lombok.val;

import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.flow.FlowController;
import org.nd4j.jita.memory.impl.CudaFullCachingProvider;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.executors.ExecutorServiceProvider;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.jita.memory.impl.CudaDirectProvider;
import org.nd4j.jita.memory.impl.CudaCachingZeroProvider;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.linalg.primitives.Pair;

import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import static org.junit.Assert.*;

@Slf4j
@Ignore("AB 2019/05/23 - Getting stuck (tests never finishing) on CI - see issue #7657")
public class AllocatorTest {
    private static final long SAFETY_OFFSET = 1024L;	

    @Test
    public void testCounters() {
        int deviceId = 0;
        MemoryTracker tracker = new MemoryTracker();

        assertTrue(0 == tracker.getAllocatedAmount(deviceId));
        assertTrue(0 == tracker.getCachedAmount(deviceId));
        //assertTrue(0 == tracker.getTotalMemory(deviceId));

        tracker.incrementAllocatedAmount(deviceId, 10);
        assertTrue(10 == tracker.getAllocatedAmount(deviceId));

        tracker.incrementCachedAmount(deviceId, 5);
        assertTrue(5 == tracker.getCachedAmount(deviceId));

        tracker.decrementAllocatedAmount(deviceId, 5);
        assertTrue(5 == tracker.getAllocatedAmount(deviceId));

        tracker.decrementCachedAmount(deviceId, 5);
        assertTrue(0 == tracker.getCachedAmount(deviceId));

        //assertTrue(0 == tracker.getTotalMemory(deviceId));

        for (int e = 0; e < Nd4j.getAffinityManager().getNumberOfDevices(); e++) {
            val ttl = tracker.getTotalMemory(e);
            log.info("Device_{} {} bytes", e, ttl);
            assertNotEquals(0, ttl);
        }
    }

    @Test
    public void testWorkspaceInitSize() {

        long initSize = 1024;
	    MemoryTracker tracker = MemoryTracker.getInstance();

        WorkspaceConfiguration workspaceConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .initialSize(initSize)
                .build();

	    try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, "test121")) {
        	assertEquals(initSize + SAFETY_OFFSET, tracker.getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        }

	    val ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("test121");
	    ws.destroyWorkspace();

	    assertEquals(0, tracker.getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
    }


    @Test
    public void testWorkspaceSpilledSize() {

        long initSize = 0;
        MemoryTracker tracker = MemoryTracker.getInstance();

        WorkspaceConfiguration workspaceConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .initialSize(initSize)
                .build();

        try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, "test99323")) {
            assertEquals(0L, tracker.getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));

            val array = Nd4j.createFromArray(1.f, 2.f, 3.f, 4.f);

            assertEquals(array.length() * array.data().getElementSize(), tracker.getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        }

        val ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("test99323");
        ws.destroyWorkspace();

        assertEquals(0, tracker.getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
    }

    @Test
    public void testWorkspaceSpilledSizeHost() {

        long initSize = 0;
        MemoryTracker tracker = MemoryTracker.getInstance();

        WorkspaceConfiguration workspaceConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .policyMirroring(MirroringPolicy.HOST_ONLY)
                .initialSize(initSize)
                .build();

        try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, "test99323222")) {
            assertEquals(0L, tracker.getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));

            val array = Nd4j.createFromArray(1.f, 2.f, 3.f, 4.f);

            assertEquals(0, tracker.getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        }

        val ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("test99323222");
        ws.destroyWorkspace();

        assertEquals(0, tracker.getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
    }


    @Ignore
    @Test
    public void testWorkspaceAlloc() {

        long initSize = 0;
        long allocSize = 48;

        val workspaceConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .initialSize(initSize)
                .policyMirroring(MirroringPolicy.HOST_ONLY) // Commenting this out makes it so that assert is not triggered (for at least 40 secs or so...)
                .build();

	    try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, "test")) {
        	final INDArray zeros = Nd4j.zeros(allocSize, 'c');
		System.out.println("Alloc1:" + MemoryTracker.getInstance().getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        	assertTrue(allocSize ==
                    MemoryTracker.getInstance().getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
	}
        assertTrue(allocSize == 
                MemoryTracker.getInstance().getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        /*Nd4j.getWorkspaceManager().destroyWorkspace(ws);
        assertTrue(0L ==
                MemoryTracker.getInstance().getWorkspaceAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));*/
    }

    @Test
    public void testDirectProvider() {
        INDArray input = Nd4j.zeros(1024);
        CudaDirectProvider provider = new CudaDirectProvider();
        AllocationShape shape = AllocationUtils.buildAllocationShape(input);
        AllocationPoint point = new AllocationPoint();
        point.setShape(shape);

        val allocBefore = MemoryTracker.getInstance().getAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());
        val cachedBefore = MemoryTracker.getInstance().getCachedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());

	    val pointers = provider.malloc(shape, point, AllocationStatus.DEVICE);
	    point.setPointers(pointers);

        System.out.println(MemoryTracker.getInstance().getAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        System.out.println(MemoryTracker.getInstance().getCachedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));

        val allocMiddle = MemoryTracker.getInstance().getAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());
        val cachedMiddle = MemoryTracker.getInstance().getCachedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());

	    provider.free(point);

        System.out.println(MemoryTracker.getInstance().getAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        System.out.println(MemoryTracker.getInstance().getCachedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));

        val allocAfter = MemoryTracker.getInstance().getAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());
        val cachedAfter = MemoryTracker.getInstance().getCachedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());

        assertTrue(allocBefore < allocMiddle);
        assertEquals(allocBefore, allocAfter);

        assertEquals(cachedBefore, cachedMiddle);
        assertEquals(cachedBefore, cachedAfter);
    }

    @Test
    public void testZeroCachingProvider() {
        INDArray input = Nd4j.zeros(1024);
        CudaCachingZeroProvider provider = new CudaCachingZeroProvider();
        AllocationShape shape = AllocationUtils.buildAllocationShape(input);
        AllocationPoint point = new AllocationPoint();
        point.setShape(shape);

        val allocBefore = MemoryTracker.getInstance().getAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());
        val cachedBefore = MemoryTracker.getInstance().getCachedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());

        val pointers = provider.malloc(shape, point, AllocationStatus.DEVICE);
        point.setPointers(pointers);

        System.out.println(MemoryTracker.getInstance().getAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        System.out.println(MemoryTracker.getInstance().getCachedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));

        val allocMiddle = MemoryTracker.getInstance().getAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());
        val cachedMiddle = MemoryTracker.getInstance().getCachedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());

        provider.free(point);

        System.out.println(MemoryTracker.getInstance().getAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        System.out.println(MemoryTracker.getInstance().getCachedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));

        val allocAfter = MemoryTracker.getInstance().getAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());
        val cachedAfter = MemoryTracker.getInstance().getCachedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());

        assertTrue(allocBefore < allocMiddle);
        assertEquals(allocBefore, allocAfter);

        assertEquals(cachedBefore, cachedMiddle);
        assertEquals(cachedBefore, cachedAfter);
    }

    @Test
    public void testFullCachingProvider() {
        INDArray input = Nd4j.zeros(1024);
        val provider = new CudaFullCachingProvider();
        AllocationShape shape = AllocationUtils.buildAllocationShape(input);
        AllocationPoint point = new AllocationPoint();
        point.setShape(shape);

        val allocBefore = MemoryTracker.getInstance().getAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());
        val cachedBefore = MemoryTracker.getInstance().getCachedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());

        val pointers = provider.malloc(shape, point, AllocationStatus.DEVICE);
        point.setPointers(pointers);

        System.out.println(MemoryTracker.getInstance().getAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        System.out.println(MemoryTracker.getInstance().getCachedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));

        val allocMiddle = MemoryTracker.getInstance().getAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());
        val cachedMiddle = MemoryTracker.getInstance().getCachedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());

        provider.free(point);

        System.out.println(MemoryTracker.getInstance().getAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        System.out.println(MemoryTracker.getInstance().getCachedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));

        val allocAfter = MemoryTracker.getInstance().getAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());
        val cachedAfter = MemoryTracker.getInstance().getCachedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread());

        assertTrue(allocBefore < allocMiddle);
        assertEquals(allocBefore, allocAfter);

        //assertEquals(0, cachedBefore);
        //assertEquals(0, cachedMiddle);
        //assertEquals(shape.getNumberOfBytes(), cachedAfter);

        assertEquals(cachedBefore, cachedMiddle);
        assertTrue(cachedBefore < cachedAfter);
    }

    @Test
    public void testCyclicCreation() throws Exception {
        Nd4j.create(100);

        log.info("Approximate free memory: {}", MemoryTracker.getInstance().getApproximateFreeMemory(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
        log.info("Real free memory: {}", MemoryTracker.getInstance().getPreciseFreeMemory(Nd4j.getAffinityManager().getDeviceForCurrentThread()));

        val timeStart = System.currentTimeMillis();

        while (true) {
            //val array = Nd4j.create(DataType.FLOAT, 1000, 1000);
            val array = Nd4j.create(DataType.FLOAT, RandomUtils.nextInt(100, 1000), RandomUtils.nextInt(100, 1000));

            val timeEnd = System.currentTimeMillis();
            if (timeEnd - timeStart > 5 * 60 * 1000) {
                log.info("Exiting...");
                break;
            }
        }

        while (true) {
            log.info("Cached device memory: {}", MemoryTracker.getInstance().getCachedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
            log.info("Active device memory: {}", MemoryTracker.getInstance().getAllocatedAmount(Nd4j.getAffinityManager().getDeviceForCurrentThread()));
            log.info("Cached host memory: {}", MemoryTracker.getInstance().getCachedHostAmount());
            log.info("Active host memory: {}", MemoryTracker.getInstance().getAllocatedHostAmount());

            System.gc();
            Thread.sleep(30000);
        }
    }

    @Test
    public void testAllocations() {
        INDArray x = Nd4j.create(DataType.FLOAT, 10, 5);
        assertArrayEquals(new long[]{10, 5}, x.shape());

        for (DataType dataType : DataType.values()) {
            for (int i = 0; i < 10; ++i) {

                x = Nd4j.create(DataType.FLOAT, 10 * i + 1, 5 * i + 2);
                assertArrayEquals(new long[]{10 * i + 1, 5 * i + 2}, x.shape());

                val pointX = AtomicAllocator.getInstance().getAllocationPoint(x.shapeInfoDataBuffer());
                assertNotNull(pointX);
                assertTrue(x.shapeInfoDataBuffer().isConstant());

                assertNotNull(pointX.getHostPointer());
                assertNotNull(pointX.getDevicePointer());

                assertEquals(64, pointX.getShape().getNumberOfBytes());
            }
        }
    }

    @Test
    public void testAllocations1() {
        INDArray x = Nd4j.zeros(1,10);

        for (int i = 0; i < 100000; ++i) {
            INDArray toAdd = Nd4j.ones(1,10);
            x.putRow(i+1, toAdd);
        }

        assertTrue(x.shapeInfoDataBuffer().isConstant());

        val pointX = AtomicAllocator.getInstance().getAllocationPoint(x.shapeInfoDataBuffer());
        assertNotNull(pointX);

        assertNotNull(pointX);
        assertTrue(x.shapeInfoDataBuffer().isConstant());

        assertNotNull(pointX.getHostPointer());
        assertNotNull(pointX.getDevicePointer());

        assertEquals(64, pointX.getShape().getNumberOfBytes());
    }

    @Test
    public void testReallocate() {
        INDArray x = Nd4j.create(DataType.FLOAT, 10, 5);
        var pointX = AtomicAllocator.getInstance().getAllocationPoint(x.data());

        assertNotNull(pointX);

        assertEquals(200, pointX.getShape().getNumberOfBytes());

        val hostP = pointX.getHostPointer();
        val deviceP = pointX.getDevicePointer();

        assertEquals(50, x.data().capacity());
        x.data().reallocate(500);

        pointX = AtomicAllocator.getInstance().getAllocationPoint(x.data());

        assertEquals(500, x.data().capacity());
        assertEquals(2000, pointX.getShape().getNumberOfBytes());

        assertNotEquals(hostP, pointX.getHostPointer());
        assertNotEquals(deviceP, pointX.getDevicePointer());
    }

    @Test
    public void testDataMigration() {

        for (boolean p2pEnabled : new boolean[]{true, false}) {

            CudaEnvironment.getInstance().getConfiguration().allowCrossDeviceAccess(p2pEnabled);

            Thread[] threads = new Thread[4];
            List<Pair<INDArray, INDArray>> sumsPerList = new ArrayList<>();
            List<INDArray> lst = new ArrayList<>();

            for (int i = 0; i < 4; ++i) {
                threads[i] = new Thread() {
                    @Override
                    public void run() {
                        INDArray x = Nd4j.rand(1, 10);
                        Pair<INDArray, INDArray> pair = new Pair<>();
                        pair.setFirst(Nd4j.sum(x));
                        pair.setSecond(x);
                        sumsPerList.add(pair);
                        lst.add(x);
                    }
                };
                threads[i].start();
            }

            try {
                for (val thread : threads) {
                    thread.join();
                }
            } catch (InterruptedException e) {
                log.info("Interrupted");
            }

            Collections.shuffle(lst);

            for (int i = 0; i < lst.size(); ++i) {
                INDArray data = lst.get(i);

                for (int j = 0; j < sumsPerList.size(); ++j) {
                    if (sumsPerList.get(j).getFirst().equals(data))
                        assertEquals(sumsPerList.get(j).getSecond(), data);

                }
            }
        }
    }


    @Ignore
    @Test
    public void testHostFallback() {
        // Take device memory
        long bytesFree = MemoryTracker.getInstance().getApproximateFreeMemory(0);
        Pointer p = Nd4j.getMemoryManager().allocate((long)(bytesFree*0.75), MemoryKind.DEVICE, true);

        // Fallback to host
        INDArray x1  = Nd4j.create(1, (long)(bytesFree*0.15));
        val pointX = AtomicAllocator.getInstance().getAllocationPoint(x1.shapeInfoDataBuffer());

        assertNotNull(pointX);
        assertNotNull(pointX.getHostPointer());
        assertNotNull(pointX.getDevicePointer());

        Nd4j.getMemoryManager().release(p, MemoryKind.DEVICE);
    }

    @Test
    public void testAffinityGuarantees() {
        ExecutorService service = ExecutorServiceProvider.getExecutorService();
        final INDArray steady = Nd4j.rand(1,100);
        Map<INDArray, Integer> deviceData = new HashMap<>();

        Future<List<INDArray>>[] results = new Future[10];
        for (int i = 0; i < results.length; ++i) {
            results[i] = service.submit(new Callable<List<INDArray>>() {
                @Override
                public List<INDArray> call() {
                    List<INDArray> retVal = new ArrayList<>();
                    for (int i = 0; i < 100; ++i) {
                        INDArray x = Nd4j.rand(1, 100);
                        System.out.println("Device for x:" + Nd4j.getAffinityManager().getDeviceForArray(x));
                        System.out.println("Device for steady: " + Nd4j.getAffinityManager().getDeviceForArray(steady));
                        deviceData.put(x, Nd4j.getAffinityManager().getDeviceForArray(x));
                        deviceData.put(steady, Nd4j.getAffinityManager().getDeviceForArray(steady));
                        retVal.add(x);
                    }
                    Thread[] innerThreads = new Thread[4];
                    for (int k = 0; k < 4; ++k) {
                        innerThreads[k] = new Thread() {
                            @Override
                            public void run() {
                                for (val res : retVal) {
                                    assertEquals(deviceData.get(res), Nd4j.getAffinityManager().getDeviceForArray(res));
                                    assertEquals(deviceData.get(steady), Nd4j.getAffinityManager().getDeviceForArray(steady));
                                }
                            }
                        };
                        innerThreads[k].start();
                    }
                    try {
                        for (int k = 0; k < 4; ++k) {
                            innerThreads[k].join();
                        }
                    } catch (InterruptedException e) {
                        log.info(e.getMessage());
                    }
                    return retVal;
                }
            });

            try {
                List<INDArray> resArray = results[i].get();
                for (val res : resArray) {
                    assertEquals(deviceData.get(res), Nd4j.getAffinityManager().getDeviceForArray(res));
                    assertEquals(deviceData.get(steady), Nd4j.getAffinityManager().getDeviceForArray(steady));
                }
            } catch (Exception e) {
                log.info(e.getMessage());
            }
        }
    }

    @Test
    public void testEventsRelease() {
        FlowController controller = AtomicAllocator.getInstance().getFlowController();
        long currEventsNumber = controller.getEventsProvider().getEventsNumber();

        INDArray x = Nd4j.rand(1,10);
        controller.prepareAction(x);
        assertEquals(currEventsNumber+1, controller.getEventsProvider().getEventsNumber());

        INDArray arg1 = Nd4j.rand(1,100);
        INDArray arg2 = Nd4j.rand(1,200);
        INDArray arg3 = Nd4j.rand(1,300);
        controller.prepareAction(x, arg1, arg2, arg3);
        assertEquals(currEventsNumber+5, controller.getEventsProvider().getEventsNumber());
    }

    @Test
    public void testReleaseContext() {
        LimitedContextPool pool = (LimitedContextPool) AtomicAllocator.getInstance().getContextPool();
        System.out.println(pool.acquireContextForDevice(0));
        INDArray x = Nd4j.rand(1,10);
        pool.releaseContext(pool.getContextForDevice(0));
        System.out.println(pool.getContextForDevice(0));
    }

    @Test
    public void testDataBuffers() {
        INDArray x = Nd4j.create(DataType.FLOAT, 10, 5);
        val pointX = AtomicAllocator.getInstance().getAllocationPoint(x.shapeInfoDataBuffer());
        assertEquals(50, x.data().capacity());
        x.data().destroy();
        assertNull(x.data());
        assertEquals(64, pointX.getShape().getNumberOfBytes());
        System.out.println(pointX.getHostPointer());
        System.out.println(pointX.getDevicePointer());
    }
}
