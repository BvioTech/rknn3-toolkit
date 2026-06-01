import os
import sys
import time
import argparse
import numpy as np

from rknn3lite.api import RKNN3Lite, dump_tensor_attr
from rknn3lite.api.rknn3_types import (
    RKNN3QueryCmd,
    RKNN3TensorType,
    RKNN3TensorLayout,
    RKNN3TensorQntType,
    rknn3_get_layout_string,
    rknn3_get_type_string,
)

DEFAULT_LOOP_COUNT = 1

# ============================================= Utility Functions =============================================


def cosine_similarity(a, b):
    """Compute cosine similarity between two float32 vectors."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-30 or norm_b < 1e-30:
        return 0.0
    similarity = dot / (norm_a * norm_b)
    return float(np.clip(similarity, -1.0, 1.0))


def get_align_hw(hw):
    """Align HW dimension (same logic as C++ getAlignHW)."""
    if hw == 1:
        return 1
    return (hw + 3) // 4 * 4



def shape_count(shape, n_dims):
    """Calculate total number of elements from shape."""
    elems = 1
    for i in range(n_dims):
        elems *= shape[i]
    return elems


def print_data_values(data, count=10):
    """Print first N values of a numpy array."""
    n = min(count, data.size)
    for i in range(n):
        print(f'  [{i}] {data.flat[i]}')


# ============================================= Main =============================================


def main():
    parser = argparse.ArgumentParser(
        description='RKNN3 Model Test - Load and run RKNN model with optional input/golden comparison.\n'
                    'Example: python rknn3_model_test.py --rknn_path model.rknn --weight_path model.weight '
                    '--input_npy_paths input0.npy#input1.npy --golden_npy_paths golden0.npy#golden1.npy '
                    '--core_mask 0x0f --loop_count 10',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--rknn_path', type=str, required=True, help='RKNN model file path (.rknn)')
    parser.add_argument('--input_npy_paths', type=str, default=None,
                        help='Input npy file paths, separated by "#". If not provided, random input will be generated.')
    parser.add_argument('--golden_npy_paths', type=str, default=None,
                        help='Golden output npy file paths, separated by "#". If not provided, cosine similarity will be skipped.')
    parser.add_argument('--core_mask', type=str, default=None,
                        help='NPU core mask in hex (e.g. 0x0f). If not provided, auto-generated based on core_number.')
    parser.add_argument('--loop_count', type=int, default=DEFAULT_LOOP_COUNT,
                        help=f'Number of inference loops (default: {DEFAULT_LOOP_COUNT})')
    parser.add_argument('--target', type=str, default='rk1820', help='Target device (default: rk1820)')
    args = parser.parse_args()
    
    model_path = args.rknn_path
    weight_path = args.rknn_path.replace('.rknn', '.weight')
    loop_count = args.loop_count
    target = args.target

    # Parse input/golden paths
    use_random_input = False
    skip_golden_comparison = False

    if args.input_npy_paths and args.golden_npy_paths:
        input_files = [f for f in args.input_npy_paths.split('#') if f]
        golden_files = [f for f in args.golden_npy_paths.split('#') if f]
        if not input_files or not golden_files:
            use_random_input = True
            skip_golden_comparison = True
    else:
        use_random_input = True
        skip_golden_comparison = True
        input_files = []
        golden_files = []

    if use_random_input:
        print('Input paths or golden output paths not provided, using random input and skipping golden comparison')

    print('RKNN3 Model Test (Python)')

    # Create RKNN3Lite object
    rknn_lite = RKNN3Lite()

    # Load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(model_path=model_path, weight_path=weight_path)
    if ret != 0:
        print('Load RKNN model failed')
        sys.exit(ret)
    print('done')

    # Get device id
    device_id = rknn_lite.get_devices_id()
    dev_id = device_id[0] if device_id else None

    # Query core number (need init_runtime first to query)
    print('--> Init runtime environment')
    # First init with a temporary core_mask to query core_number
    # We use core_mask=0x01 for initial query, then re-init if needed
    if args.core_mask is not None:
        core_mask = int(args.core_mask, 16)
    else:
        # Init with default core_mask first, will be adjusted after query
        core_mask = 0x01

    ret = rknn_lite.init_runtime(target=target, core_mask=core_mask, device_id=dev_id)
    if ret != 0:
        print('Init runtime environment failed')
        sys.exit(ret)
    print('done')

    # Query device memory info
    dev_mem_info = rknn_lite.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_DEVICE_MEM_INFO)
    if dev_mem_info is not None:
        print(f'Device Memory Info: total={dev_mem_info.sys_total // (1024 * 1024)} MB, '
              f'free={dev_mem_info.sys_free // (1024 * 1024)} MB')
        for i in range(dev_mem_info.node_num):
            print(f'  Node {i}: total={dev_mem_info.node_mem_info[i].total // (1024 * 1024)} MB, '
                  f'free={dev_mem_info.node_mem_info[i].free // (1024 * 1024)} MB')

    # Query core number
    core_num = rknn_lite.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_CORE_NUMBER)
    if core_num is not None:
        print(f'Core number: {core_num}')
    else:
        print('Warning: Failed to query core number, using default')
        core_num = 1

    # If core_mask was not specified, auto-generate based on core_number
    if args.core_mask is None:
        core_mask = 0
        for i in range(core_num):
            core_mask |= (1 << i)
        print(f'Auto-generated core_mask: 0x{core_mask:x} for {core_num} cores')

        # Re-init runtime with the correct core_mask
        rknn_lite.release()
        rknn_lite = RKNN3Lite()
        ret = rknn_lite.load_rknn(model_path=model_path, weight_path=weight_path)
        if ret != 0:
            print('Load RKNN model failed')
            sys.exit(ret)
        ret = rknn_lite.init_runtime(target=target, core_mask=core_mask, device_id=dev_id)
        if ret != 0:
            print('Init runtime environment failed')
            sys.exit(ret)
    else:
        # Validate user-provided core_mask
        user_core_num = bin(core_mask).count('1')
        if user_core_num != core_num:
            print(f'Error: core_mask 0x{core_mask:x} ({user_core_num} cores) does not match core number {core_num}')
            rknn_lite.release()
            sys.exit(-1)

    # Query input/output info
    io_num = rknn_lite.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_IN_OUT_NUM)
    if io_num is None:
        print('Failed to query IO number')
        rknn_lite.release()
        sys.exit(-1)
    n_input = io_num.n_input
    n_output = io_num.n_output
    print(f'Model input num: {n_input}, output num: {n_output}')

    # Query and dump input tensor attributes
    print('Input tensors:')
    input_attrs = []
    for i in range(n_input):
        attr = rknn_lite.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_INPUT_ATTR, index=i)
        if attr is None:
            print(f'Failed to query input attr for index {i}')
            rknn_lite.release()
            sys.exit(-1)
        input_attrs.append(attr)
        dump_tensor_attr(attr, prefix='  input')

    # Query and dump output tensor attributes
    print('Output tensors:')
    output_attrs = []
    for i in range(n_output):
        attr = rknn_lite.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_OUTPUT_ATTR, index=i)
        if attr is None:
            print(f'Failed to query output attr for index {i}')
            rknn_lite.release()
            sys.exit(-1)
        output_attrs.append(attr)
        dump_tensor_attr(attr, prefix='  output')

    # Load input npy files
    input_arrays = []
    if not use_random_input:
        if len(input_files) != n_input:
            print(f'Error: Number of input files ({len(input_files)}) does not match model inputs ({n_input})')
            rknn_lite.release()
            sys.exit(-1)
        for i, fpath in enumerate(input_files):
            arr = np.load(fpath)
            input_arrays.append(arr)
            print(f'Loaded input[{i}]: {fpath}, shape={arr.shape}, dtype={arr.dtype}')

    # Load golden output npy files
    golden_arrays = []
    if not skip_golden_comparison:
        if len(golden_files) != n_output:
            print(f'Error: Number of golden files ({len(golden_files)}) does not match model outputs ({n_output})')
            rknn_lite.release()
            sys.exit(-1)
        for i, fpath in enumerate(golden_files):
            arr = np.load(fpath)
            golden_arrays.append(arr)
            print(f'Loaded golden[{i}]: {fpath}, shape={arr.shape}, dtype={arr.dtype}')

    # Prepare input data
    inputs = []
    for in_idx in range(n_input):
        input_attr = input_attrs[in_idx]
        dst_elems = shape_count(input_attr.shape, input_attr.n_dims)

        if use_random_input:
            # Generate random input data
            name = input_attr.name.decode('utf-8', errors='ignore')
            print(f'Generating random input data for input {in_idx} (name: {name})')

            # Check special tensors (e.g. LLM Th/Tc/Ts/Tsr)
            is_special_tensor = False
            special_value = 0.0
            if name == 'Th':
                is_special_tensor = True
                special_value = 0.0
                print("Special tensor 'Th' detected, setting all values to 0")
            elif name == 'Tc':
                is_special_tensor = True
                special_value = 1.0
                print("Special tensor 'Tc' detected, setting all values to 1")
            elif name in ('Ts', 'Tsr'):
                is_special_tensor = True
                special_value = 0.0
                print(f"Special tensor '{name}' detected, setting all values to 0")

            # Generate shape for numpy array
            shape = [input_attr.shape[j] for j in range(input_attr.n_dims)]

            dtype = input_attr.dtype
            if dtype == RKNN3TensorType.RKNN3_TENSOR_FLOAT16:
                if is_special_tensor:
                    data = np.full(shape, special_value, dtype=np.float16)
                else:
                    data = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float16)
            elif dtype == RKNN3TensorType.RKNN3_TENSOR_FLOAT32:
                if is_special_tensor:
                    data = np.full(shape, special_value, dtype=np.float32)
                else:
                    data = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            elif dtype == RKNN3TensorType.RKNN3_TENSOR_INT32:
                if is_special_tensor:
                    data = np.full(shape, int(special_value), dtype=np.int32)
                else:
                    data = np.random.randint(-1000, 1001, size=shape, dtype=np.int32)
            elif dtype == RKNN3TensorType.RKNN3_TENSOR_INT8:
                if is_special_tensor:
                    data = np.full(shape, int(special_value), dtype=np.int8)
                else:
                    data = np.random.randint(-128, 128, size=shape, dtype=np.int8)
            elif dtype == RKNN3TensorType.RKNN3_TENSOR_UINT8:
                if is_special_tensor:
                    data = np.full(shape, int(special_value), dtype=np.uint8)
                else:
                    data = np.random.randint(0, 256, size=shape, dtype=np.uint8)
            else:
                print(f'Unsupported tensor type for random generation: {rknn3_get_type_string(dtype)}')
                rknn_lite.release()
                sys.exit(-1)

            inputs.append(data)
        else:
            # Use loaded npy data
            input_data = input_arrays[in_idx]
            inputs.append(input_data)

            print(f'Input {in_idx} prepared: {dst_elems} elements')

            # Print first 10 values
            print(f'Input[{in_idx}] first values:')
            print_data_values(inputs[in_idx], count=10)

    # Run inference
    print(f'\nRunning model {loop_count} times...')
    sync_in_total = 0.0
    run_total = 0.0
    sync_out_total = 0.0
    all_outputs = None
    run_failed = False

    for loop in range(loop_count):
        t_start = time.perf_counter()
        outputs = rknn_lite.inference(inputs=inputs)
        t_end = time.perf_counter()

        if outputs is None:
            print(f'rknn3 inference failed at loop {loop + 1}!')
            run_failed = True
            break

        loop_ms = (t_end - t_start) * 1000.0
        run_total += loop_ms
        print(f'loop: {loop + 1}, inference cost {loop_ms:.3f} ms')
        all_outputs = outputs

    if run_failed:
        rknn_lite.release()
        sys.exit(-1)

    print(f'\nAll {loop_count} loops completed successfully')
    print(f'Average total time per loop: {run_total / loop_count:.3f} ms')

    # Verify outputs
    if not skip_golden_comparison and all_outputs is not None:
        for i in range(n_output):
            output_attr = output_attrs[i]
            output_data = all_outputs[i].flatten().astype(np.float32)
            golden_data = golden_arrays[i].flatten().astype(np.float32)

            dst_elems = golden_data.size

            # Save output to npy file
            output_shape = [output_attr.shape[j] for j in range(output_attr.n_dims)]
            np.save(f'output_data_{i}.npy', output_data[:dst_elems].reshape(golden_arrays[i].shape))

            # Print first 10 values comparison
            print(f'\nComparing first 10 values of Output {i}:')
            n_print = min(10, dst_elems)
            for j in range(n_print):
                print(f'  Index[{j}]: Output={output_data[j]:.5f}, Golden={golden_data[j]:.5f}')

            # Print last 10 values comparison
            print(f'\nComparing last 10 values of Output {i}:')
            for j in range(max(0, dst_elems - 10), dst_elems):
                print(f'  Index[{j}]: Output={output_data[j]:.5f}, Golden={golden_data[j]:.5f}')

            src_elems = shape_count(output_attr.shape, output_attr.n_dims)
            print(f'output native elems: {src_elems}, golden elems: {dst_elems}')

            # Compute cosine similarity
            n_compare = min(output_data.size, golden_data.size)
            similarity = cosine_similarity(output_data[:n_compare], golden_data[:n_compare])
            print(f'Output {i} cosine similarity: {similarity:.5f}')


    # Release
    rknn_lite.release()
    print('\ndone')


if __name__ == '__main__':
    main()
