import os
import cv2
import time
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
from transformers import AutoTokenizer
import ctypes
from rknn3lite.api import (RKNN3Lite, RKLLMCallback, LLMResultCallback, LLMTokenizerCallback,
                            LLMGetEmbedCallback, LLMOutputCallback, RKNN3Image, Float16,
                            dump_tensor_attr, RKNN3QueryCmd, RKNN3MemAllocFlags,
                            rknn3_get_layout_string, RKNN3Tensor, RKNN3TensorAttr, RKNN3TensorMemory,
                            LLMOutputCallbackState)


RKNN_LLM_MODEL = '/userdata/rknn_Qwen2_5_VL_demo/model/GmeQwen2VL-llm_quant.rknn'
RKNN_VISION_MODEL = '/userdata/rknn_Qwen2_5_VL_demo/model/GmeQwen2VL-vision_quant.rknn'
EMBED_PATH = '/userdata/rknn_Qwen2_5_VL_demo/model/GmeQwen2VL-llm_quant.embed.bin'
TOKENIZER_PATH = 'gme-Qwen2-VL-2B'

VOCAB_SIZE = 151936
MAX_CONTEXT_LEN = 1024
MAX_NEW_TOKENS = 1

ARGS = [{"max_new_tokens": MAX_NEW_TOKENS,
         "top_k": 1,
         "top_p": 0.001,
         "temperature": 0.1,
         "repeat_penalty": 1.05,
         "vocab_size": VOCAB_SIZE,
         "special_bos_id": 151643,
         "special_eos_id": 151645,
         "max_context_len": MAX_CONTEXT_LEN,
         "keep_history": 0}]

# Vision VIT parameters (from huggingface model config.json)
merge_size = 2
patch_size = 14

tokenizer = None
embeds_data = None
first_token = None

model_outputs = []


# ===================== Callback Functions =====================

def result_callback(userdata, result_ptr, state):
    global tokenizer, first_token

    if not hasattr(result_callback, "accumulated_tokens"):
        result_callback.accumulated_tokens = []
        result_callback.last_output_text = ""

    def decode_safe(tokens):
        text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text.split('\ufffd', 1)[0] if '\ufffd' in text else text

    # ERROR
    if state == 5:
        print("\n\nError occurred during inference")
        return 0

    # FINISH / STOP / MAX_TOKEN
    if state in (2, 3, 4):
        if result_callback.accumulated_tokens:
            try:
                safe_text = decode_safe(result_callback.accumulated_tokens)
                new_part = safe_text[len(result_callback.last_output_text):]
                if new_part:
                    print(new_part, end="", flush=True)
            except Exception as e:
                print(f"\n[Decode error: {e}]", flush=True)
        result_callback.accumulated_tokens.clear()
        result_callback.last_output_text = ""
        msg = {2: "Finished", 3: "Stop", 4: "Max new token reached"}.get(state, "Unknown")
        print(f"\n\n--------------------{msg}--------------------")
        return 0

    # WAITING
    if state == 1:
        print("\n\nWaiting for UTF-8 encoded character")
        return 0

    # NORMAL
    if state == 0:
        n = result_ptr.contents.num_tokens
        new_tokens = [result_ptr.contents.token_ids[i] for i in range(n)]
        result_callback.accumulated_tokens.extend(new_tokens)
        if first_token is None:
            first_token = time.perf_counter()
        try:
            safe_text = decode_safe(result_callback.accumulated_tokens)
            new_part = safe_text[len(result_callback.last_output_text):]
            if new_part:
                print(new_part, end="", flush=True)
                result_callback.last_output_text += new_part
        except Exception as e:
            print(f"\n[Temp decode error: {e}], waiting for more tokens", flush=True)
            return 0

    return 0


def tokenizer_callback(userdata, text_ptr, text_len, tokens_ptr, n_tokens_max):
    text = text_ptr.decode('utf-8')
    inputs = tokenizer(text, return_tensors='np', truncation=True)
    
    tokens = inputs['input_ids'][0][:n_tokens_max]

    n_tokens = len(tokens)
    
    if n_tokens <= 0:
        print(f"Tokenizer failed for {text}")
        return n_tokens
    
    for i in range(n_tokens):
        tokens_ptr[i] = tokens[i]
    
    return n_tokens


def embed_callback(userdata, tokens_ptr, num_tokens, embed, length):
    global embeds_data
    embedding_dim = embeds_data.shape[1]

    expected_len = num_tokens * embedding_dim * np.dtype(np.float16).itemsize
    if length != expected_len:
        print("invalid embed buffer")
        return -1

    dst = np.ctypeslib.as_array(
        ctypes.cast(embed, ctypes.POINTER(ctypes.c_uint16)),
        shape=(num_tokens * embedding_dim,)
    ).view(np.float16)

    tokens = [tokens_ptr[i] for i in range(num_tokens)]

    dst[:] = embeds_data[tokens].ravel()

    return 0



def output_callback(userdata, output_tensors_ptr, n_output_tensors, state):
    global model_outputs, first_token
    print(f"\noutput_callback: state = {state}")

    if state != LLMOutputCallbackState.RKLLM_OUTPUT_CALLBACK_PREFILL_FINISHED:
        return 0

    if first_token is None:
        first_token = time.perf_counter()

    if n_output_tensors == 0 or not output_tensors_ptr:
        return 0

    try:
        for i in range(n_output_tensors):
            tensor = output_tensors_ptr[i]
            if not tensor.mem or not tensor.attr:
                continue

            mem = tensor.mem.contents
            attr = tensor.attr.contents

            print(f"output_callback: output[{i}]->attr->index = {attr.index}")
            print(f"output_callback: output[{i}]->attr->name = {attr.name.decode('utf-8', errors='ignore')}")
            print(f"output_callback: output[{i}]->mem->size = {mem.size}")

            if mem.virt_addr and attr.n_elems > 0:
                n_elems = attr.n_elems

                try:
                    data = np.ctypeslib.as_array(
                            ctypes.cast(mem.virt_addr, ctypes.POINTER(ctypes.c_uint16)),
                            shape=(n_elems,)
                        ).view(np.float16).astype(np.float32)
                    # L2 normalize
                    norm = np.linalg.norm(data)
                    normalized = data / norm if norm > 0 else data
                    model_outputs.append(normalized)

                    # Print first 10 values (matching C++ debug output)
                    for j in range(min(10, n_elems)):
                        print(f"output_callback: output[{i}][{j}] = {normalized[j]:.6f}")

                except Exception as e:
                    print(f"  Error reading tensor {i}: {e}")
    except Exception as e:
        print(f"Error in output_callback: {e}")
        import traceback
        traceback.print_exc()

    return 0


# ===================== Utility Functions =====================

def printf_perf(first_token_time, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time, vision_latency):
    """Print performance metrics matching the C++ printf_perf format."""
    print("\n--------------------------------------------------------------------------------------")
    print(" %-12s  %-15s  %-8s  %-23s  %-23s" %
          ("Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"))
    print("--------------------------------------------------------------------------------------")

    prefill_ms = (first_token_time - llm_start_time) * 1000.0
    if n_prefill_tokens == 0:
        prefill_tpt, prefill_tps = 0.0, 0.0
    else:
        prefill_tpt = prefill_ms / n_prefill_tokens
        prefill_tps = n_prefill_tokens * 1000.0 / prefill_ms

    print(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f" %
          ("Prefill", prefill_ms, n_prefill_tokens, prefill_tpt, prefill_tps))

    decode_ms = (llm_end_time - first_token_time) * 1000.0
    if n_decode_tokens == 0:
        decode_tpt, decode_tps = 0.0, 0.0
    else:
        decode_tpt = decode_ms / n_decode_tokens
        decode_tps = n_decode_tokens * 1000.0 / decode_ms

    print(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f" %
          ("Generate", decode_ms, n_decode_tokens, decode_tpt, decode_tps))

    print("--------------------------------------------------------------------------------------")
    vision_latency_ms = vision_latency * 1000.0
    fps = 1.0 / vision_latency if vision_latency > 0 else 0.0
    print(f" Vision latency = {vision_latency_ms:.2f} ms, FPS = {fps:.2f}")


def dump_all_tensor_attr(rknn_vision, rknn_llm):
    """Print tensor attributes for both vision and LLM models."""
    for input_attr in rknn_vision.get_inputs_tensor_attr():
        dump_tensor_attr(input_attr, prefix="rknn_vision input")
    for output_attr in rknn_vision.get_outputs_tensor_attr():
        dump_tensor_attr(output_attr, prefix="rknn_vision output")

    for input_attr in rknn_llm.get_inputs_tensor_attr():
        dump_tensor_attr(input_attr, prefix="rknn_llm input")
    for output_attr in rknn_llm.get_outputs_tensor_attr():
        dump_tensor_attr(output_attr, prefix="rknn_llm output")


def prune_model_img_process(img):
    img = np.float32(img)
    img[0,2,...] = (img[0,2,...] - 104.09)/70.3 
    img[0,1,...] = (img[0,1,...] - 116.74)/66.6 
    img[0,0,...] = (img[0,0,...] - 122.70)/68.5 
    patches = np.concatenate([img, img], axis=1)
    h = img.shape[2]
    w = img.shape[3]
    patches = patches.reshape(1, 2, 3, h // 2 // 14, 2, 14, w // 2 // 14, 2, 14)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    feature = patches.reshape(1 * h // 14 * w // 14, 3 * 2 * 14 * 14)
    return feature

# ===================== Main =====================

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Inference GME-Qwen2-VL model with RKNN")
    parser.add_argument("--rknn_llm_path", type=str, required=False, default=RKNN_LLM_MODEL)
    parser.add_argument("--rknn_vision_path", type=str, required=False, default=RKNN_VISION_MODEL)
    parser.add_argument("--tokenizer_path", type=str, required=False, default=TOKENIZER_PATH)
    parser.add_argument("--embed_path", type=str, required=False, default=EMBED_PATH)
    parser.add_argument("--image_path", type=str, required=False, default="./demo.jpg")
    parser.add_argument("--prompt", type=str, required=False, default="There are astronaut in the image.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    embeds_data = np.fromfile(args.embed_path, dtype=np.float16)
    embeds_data = embeds_data.reshape(VOCAB_SIZE, -1)

    VISION_CORE_MASK = 0xff
    LLM_CORE_MASK = 0xff

    rknn_vision = RKNN3Lite()
    rknn_llm = RKNN3Lite(llm_mode=True)

    print('--> Loading vision model')
    ret = rknn_vision.load_rknn(args.rknn_vision_path, args.rknn_vision_path.replace(".rknn", ".weight"))
    if ret != 0:
        print('Load vision model failed!')
        exit(ret)
    print('done')

    print('--> Loading LLM model')
    ret = rknn_llm.load_rknn(args.rknn_llm_path, args.rknn_llm_path.replace(".rknn", ".weight"))
    if ret != 0:
        print('Load LLM model failed!')
        exit(ret)
    print('done')

    print('--> Init vision runtime')
    ret = rknn_vision.init_runtime(target='rk1820', core_mask=VISION_CORE_MASK)
    if ret != 0:
        print('Init vision runtime failed!')
        exit(ret)
    print('done')

    print('--> Init LLM runtime (model only, no session yet)')
    ret = rknn_llm.init_runtime(target='rk1820', core_mask=LLM_CORE_MASK)
    if ret != 0:
        print('Init LLM runtime failed!')
        exit(ret)
    print('done')

    print('\n--> Query model info')

    # Query LLM IO and LLM config
    llm_io = rknn_llm.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_IN_OUT_NUM)
    print(f"LLM model: n_input={llm_io.n_input}, n_output={llm_io.n_output}")

    llm_config = rknn_llm.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_LLM_CONFIG)
    if llm_config is not None:
        print(f"LLM config: max_ctx_len={llm_config.max_ctx_len}, vocab_size={llm_config.vocab_size}, "
              f"embedding_dim={llm_config.embedding_dim}")
        # Update ARGS with queried max_context_len (like C++)
        ARGS[0]['max_context_len'] = llm_config.max_ctx_len

    n_output_tensors = llm_io.n_output

    OutputTensorArray = RKNN3Tensor * n_output_tensors
    output_tensors = OutputTensorArray()

    for i in range(n_output_tensors):
        attr = rknn_llm.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_OUTPUT_ATTR, index=i)
        if attr is None:
            print(f"rknn3_query output attr failed for index {i}")
            exit(-1)

        if rknn_llm.create_mem(attr, output_tensors[i]) is None:
            print(f"Failed to create memory for output tensor {i}")
            exit(-1)

        print(f"output tensor[{i}]: {attr.name.decode('utf-8', errors='ignore')}")

    callback = RKLLMCallback()
    callback.result_callback = LLMResultCallback(result_callback)
    callback.result_userdata = None

    callback.tokenizer_callback = LLMTokenizerCallback(tokenizer_callback)
    _tokenizer_ud = ctypes.py_object(tokenizer)
    callback.tokenizer_userdata = ctypes.cast(ctypes.pointer(_tokenizer_ud), ctypes.c_void_p)

    callback.embed_callback = LLMGetEmbedCallback(embed_callback)
    _embed_ud = ctypes.py_object(embeds_data)
    callback.embed_userdata = ctypes.cast(ctypes.pointer(_embed_ud), ctypes.c_void_p)

    callback.output_callback = LLMOutputCallback(output_callback)
    _output_ud = ctypes.py_object(embeds_data)
    callback.output_userdata = ctypes.cast(ctypes.pointer(_output_ud), ctypes.c_void_p)

    callback.output_tensors = ctypes.cast(output_tensors, ctypes.POINTER(RKNN3Tensor))
    callback.n_output_tensors = n_output_tensors

    ret = rknn_llm.init_llm_session(llm_args=ARGS, llm_callback=callback)
    if ret != 0:
        print('Init LLM session failed!')
        exit(ret)
    print('done')

    # Dump all tensor attributes
    dump_all_tensor_attr(rknn_vision, rknn_llm)

    instruction = "You are a helpful assistant."
    prompt_user = args.prompt

    prompts = ["<|im_start|>system\n" + instruction +"<|im_end|>\n<|im_start|>user\n" + "<image>" +
                        prompt_user + "<|im_end|>\n<|im_start|>assistant\n<|endoftext|>"]
    model_outputs = []
    for prompt in prompts:
        ori_img = cv2.imread(args.image_path)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (392, 392))
        if rknn_vision.get_inputs_tensor_attr()[0].n_dims == 4: #完整版
            feature = np.float16(img.reshape(1,392, 392,3))
        else: #裁剪版
            feature = prune_model_img_process(img.transpose(2,0,1).reshape(1,3,392, 392))
            feature = np.float16(feature)

        # 运行视觉模型进行推理
        print('--> Running vision model')
        vision_start=time.perf_counter()
        outputs = rknn_vision.inference(inputs=[feature])[0]
        vision_latency=time.perf_counter() - vision_start
        outputs = np.float16(np.expand_dims(outputs, 0)) # 注意有的模型输出结果是2维，需要补一个batch维度

        inputs = []
        llm_input = RKNN3Image()
        llm_input.image_embed = outputs.ctypes.data_as(ctypes.POINTER(Float16))
        llm_input.n_image_tokens = outputs.shape[1]
        llm_input.n_image = outputs.shape[0]
        llm_input.image_width = 392
        llm_input.image_height = 392
        llm_input.image_start = "<|vision_start|>".encode('utf-8')
        llm_input.image_end = "<|vision_end|>".encode('utf-8')
        llm_input.image_content = "<|image_pad|>".encode('utf-8')
        inputs.append(llm_input)

        # 运行LLM推理
        ret, [n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time] = rknn_llm.session_run(inputs=inputs, prompt=prompt)
        if ret != 0:
            print('RKNN LLM inference failed!')
            exit(ret)
        printf_perf(first_token, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time, vision_latency)
        first_token = None
        np.save('model_outputs.npy', model_outputs[0])

    print('done')
    for tensor in output_tensors:
        rknn_llm.destroy_mem(tensor.mem)
    rknn_vision.release()
    rknn_llm.release()



