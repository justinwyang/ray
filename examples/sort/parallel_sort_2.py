import numpy as np
import ray
import time

ray.init(
    num_cpus=8, # We will be using 8 workers
    include_webui=False,
    plasma_directory='/tmp', # The object store is mounted to local file system
    ignore_reinit_error=True,
    object_store_memory=int(2*1e9),
)

num_input_blocks = 4
num_output_blocks = 4
num_samples_for_cutoffs = 100
array = np.random.randint(0, 256, size =10**8, dtype=np.uint8)


def compute_cutoffs(array, num_samples):
    samples = array[np.random.randint(0, len(array), size=num_samples)]
    samples.sort()
    boundary_indices = np.arange(1, num_output_blocks) * (len(samples) // num_output_blocks)

    # These are the boundaries between the output blocks. We will assume that each
    # boundary value is contained in the upper block.
    cutoffs = samples[boundary_indices]
    return cutoffs


@ray.remote(num_return_vals=num_output_blocks)
def partition_input_block(input_block, cutoffs):
    # By default, numpy arrays passed to remote functions are read-only so that
    # they can be accessed through shared memory without creating a copy.
    # However, we need to mutate this array, so we will create a copy.
    input_block = input_block.copy()
    input_block.sort()
    partition_indices = input_block.searchsorted(cutoffs)
    return np.split(input_block, partition_indices)


@ray.remote
def compute_output_block(*partitions):
    # There are probably faster ways to merge these sorted partitions together
    # than to concatenate them and sort the result, but this seems to work.
    result = np.concatenate(partitions)
    result.sort()
    return result

if __name__ == "__main__":
    parallel_start_time = time.time()

    cutoffs = compute_cutoffs(array, num_samples_for_cutoffs)
    blocks = np.split(array, num_input_blocks)
    block_ids = [ray.put(block) for block in blocks]
    partition_ids = np.empty(shape=(num_input_blocks, num_output_blocks), dtype=object)
    for i in range(num_input_blocks):
        partition_ids[i] = np.array(partition_input_block.remote(block_ids[i], cutoffs))
    output_block_ids = []
    for j in range(num_output_blocks):
        output_block_ids.append(compute_output_block.remote(*partition_ids[:, j]))
    sorted_result = np.concatenate(ray.get(output_block_ids))

    parallel_duration = time.time() - parallel_start_time

    print("Parallel sort duration: {}.".format(parallel_duration))

    array_copy = array.copy()
    serial_start_time = time.time()
    array_copy.sort()
    serial_duration = time.time() - serial_start_time
    print("Serial sort duration: {}.".format(serial_duration))

    # Check work.
    assert parallel_duration < 0.75 * serial_duration
    assert np.all(sorted_result == array_copy)
