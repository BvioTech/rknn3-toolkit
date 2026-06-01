import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
from transformers import AutoTokenizer
from transformers.utils.versions import require_version
require_version(
    "transformers>=5.5.0",
    "The remote code requires transformers>=5.5.0, please upgrade: pip install -U transformers"
)
import ctypes
import numpy as np
import time
import mmap
import json
import struct

from rknn3lite import api as rknn_api
from rknn3lite.api import RKNN3Lite, RKLLMCallback, LLMResultCallback, LLMGetEmbedCallback, LLMTokenizerCallback, LLMInputCallback, LLMOutputCallback
from rknn3lite.api.rknn3_types import RKNN3QueryCmd, RKNN3KVCacheClearPolicy


RKNN_MODEL = 'gemma-4-e2b-it.rknn'
WEIGHT_MODEL = 'gemma-4-e2b-it.weight'
EMBED_PATH = 'gemma-4-e2b-it.embed.bin'
PER_LAYER_EMBED_PATH = 'gemma-4-e2b-it_per_layer_inputs.embed.bin'
SAFETENSORS_PATH = 'rope_caches.safetensors'

TOKENIZER_PATH = 'gemma-4-e2b-it'

system_prompt = ""
prompt_prefix = "<bos><|turn>user\n"
prompt_postfix = "<turn|>\n<|turn>model\n"

ROPE_CACHE_NAMES = [
    "rope_cos_cache_0", "rope_sin_cache_0",
    "rope_cos_cache_1", "rope_sin_cache_1"
]

tokenizer = None
embeds_data = None
per_layer_embeds_data = None
first_token = None
rope_mmap = None
rope_file = None
rope_mmap_addr = 0
rope_mmap_base_obj = None
rope_caches = {}
llm_ext_input_indices = None
callback_refs = []



DTYPE_ELEM_SIZE = {
    0: 4,   # FLOAT32
    1: 2,   # FLOAT16
    2: 1,   # INT8
    3: 1,   # UINT8
    4: 2,   # INT16
    5: 2,   # UINT16
    6: 4,   # INT32
    7: 4,   # UINT32
    8: 8,   # INT64
    9: 8,   # UINT64
    10: 1,  # BOOL
    11: 1,  # INT4
    12: 1,  # FLOAT8E4M3FN
    13: 2,  # BFLOAT16
    14: 1,  # FLOAT8E8M0
    15: 1,  # FLOAT4E2M1
}

def get_dtype_elem_size(dtype: int) -> int:
    return DTYPE_ELEM_SIZE.get(int(dtype), 1)

import time

def result_callback(userdata, result_ptr, state):
    global tokenizer, first_token
    if not hasattr(result_callback, "token_buffer"):
        result_callback.token_buffer = []

    if state == 5:
        print("\n\nError occurred during inference", flush=True)
        result_callback.token_buffer.clear()
        return 0

    # State 2, 3, 4: FINISH / STOP / MAX_TOKEN
    if state in (2, 3, 4):
        # 打印残留的 buffer
        if result_callback.token_buffer:
            try:
                final_text = tokenizer.decode(result_callback.token_buffer, skip_special_tokens=True)
                print(final_text.replace('\ufffd', ''), end="", flush=True)
            except Exception as e:
                print(f"\n[Decode error: {e}]", flush=True)
                
        result_callback.token_buffer.clear()
        
        import sys
        sys.stdout.flush()
        return 0

    if state == 1:
        return 0

    if state == 0:
        if first_token is None:
            first_token = time.perf_counter()

        # 获取新的 token
        n = result_ptr.contents.num_tokens
        new_tokens = [int(result_ptr.contents.token_ids[i]) for i in range(n)]
        
        # 将新 token 放入缓冲
        result_callback.token_buffer.extend(new_tokens)

        try:
            text = tokenizer.decode(result_callback.token_buffer, skip_special_tokens=True) # 尝试解码当前缓冲区的 token
            if '\ufffd' not in text: ## 如果没有乱码字符（\ufffd），说明这是一个完整的 UTF-8 字符/词组
                print(text, end="", flush=True)
                result_callback.token_buffer.clear()
            else:
                # 如果包含 \ufffd，说明汉字字节被截断了，保留在 buffer 中，等下一个 token 来了再一起解
                pass
                
        except Exception as e:
            print(f"\n[Temp decode error: {e}], waiting for more tokens", flush=True)

    return 0

def tokenizer_callback(userdata, text_ptr, text_len, tokens_ptr, n_tokens_max):
    if isinstance(text_ptr, (bytes, bytearray)):
        text = text_ptr[:text_len].decode('utf-8', errors='ignore')
    else:
        text = ctypes.string_at(text_ptr, text_len).decode('utf-8', errors='ignore')

    inputs = tokenizer(text, return_tensors='np', truncation=True, add_special_tokens=False)
    tokens = inputs['input_ids'][0][:n_tokens_max]
    n_tokens = len(tokens)

    if n_tokens <= 0:
        print(f"Tokenizer failed for {text}")
        return n_tokens

    for i in range(n_tokens):
        tokens_ptr[i] = int(tokens[i])

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

    tokens = [int(tokens_ptr[i]) for i in range(num_tokens)]
    for token_id in tokens:
        if token_id < 0 or token_id >= embeds_data.shape[0]:
            print(f"invalid token id: {token_id}")
            return -1
    dst[:] = embeds_data[tokens].ravel()

    return 0

def input_callback(userdata, input_tensors, n_input_tensors, param):
    global per_layer_embeds_data, rope_mmap_addr, rope_caches

    p = param.contents if hasattr(param, "contents") else param
    num_tokens = int(p.num_tokens)
    pos = int(p.pos)
    tokens = [int(p.tokens[i]) for i in range(num_tokens)]
    embedding_dim = per_layer_embeds_data.shape[1]

    for i in range(n_input_tensors):
        tensor = input_tensors[i]
        attr = tensor.attr.contents if hasattr(tensor.attr, "contents") else tensor.attr
        mem = tensor.mem.contents if hasattr(tensor.mem, "contents") else tensor.mem

        name = attr.name
        if isinstance(name, bytes):
            name = name.split(b'\0', 1)[0].decode('utf-8', errors='ignore')
        elif isinstance(name, str):
            name = name.split('\0', 1)[0]
        else:
            try:
                name = ctypes.string_at(name).split(b'\0', 1)[0].decode('utf-8', errors='ignore')
            except Exception:
                name = bytes(name).split(b'\0', 1)[0].decode('utf-8', errors='ignore')

        addr = mem.virt_addr
        if hasattr(addr, "value"):
            addr = addr.value

        if name in rope_caches:
            cache = rope_caches[name]

            elem_sz = get_dtype_elem_size(cache["dtype"])
            c1 = int(cache["shape"][1])
            c2_bytes = int(cache["shape"][4]) * elem_sz
            src_stride = int(cache["shape"][3]) * c2_bytes
            dst_stride = int(attr.shape[3]) * c2_bytes
            src_base = int(cache["offset"]) + pos * c2_bytes

            # 直接用 mmap 的虚拟地址做地址偏移，避免 rope_mmap[...] 切片产生 bytes 临时拷贝。
            for c in range(c1):
                src_addr = rope_mmap_addr + src_base + c * src_stride
                dst_addr = addr + c * dst_stride
                ctypes.memmove(dst_addr, src_addr, dst_stride)

            continue

        if name != "per_layer_inputs":
            continue

        dst = np.ctypeslib.as_array(
            ctypes.cast(ctypes.c_void_p(addr), ctypes.POINTER(ctypes.c_uint16)),
            shape=(num_tokens * embedding_dim,)
        ).view(np.float16)

        for t, token_id in enumerate(tokens):
            begin = t * embedding_dim
            end = begin + embedding_dim

            if 0 <= token_id < per_layer_embeds_data.shape[0]:
                dst[begin:end] = per_layer_embeds_data[token_id]
            else:
                dst[begin:end] = 0

    return 0


def output_callback(userdata, output_tensors, n_output_tensors, state):
    return 0


def printf_perf(first_token, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time):
    print("\n--------------------------------------------------------------------------------------")
    print(" %-12s  %-15s  %-8s  %-23s  %-23s" % 
          ("Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"))
    print("--------------------------------------------------------------------------------------")

    # Prefill 阶段：从 llm_start_time 到 first_token
    prefill_time_sec = first_token - llm_start_time
    prefill_ms = prefill_time_sec * 1000.0
    prefill_n_tokens = n_prefill_tokens

    if prefill_n_tokens == 0:
        prefill_tpt = 0.0
        prefill_tps = 0.0
    else:
        prefill_tpt = prefill_ms / prefill_n_tokens
        prefill_tps = (prefill_n_tokens * 1000.0) / prefill_ms  # tokens per second

    print(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f" %
          ("Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps))

    # Decode/Generate 阶段：从 first_token 到 llm_end_time
    decode_time_sec = llm_end_time - first_token
    decode_ms = decode_time_sec * 1000.0
    decode_n_tokens = n_decode_tokens

    if decode_n_tokens == 0:
        decode_tpt = 0.0
        decode_tps = 0.0
    else:
        decode_tpt = decode_ms / decode_n_tokens
        decode_tps = (decode_n_tokens * 1000.0) / decode_ms

    print(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f" %
          ("Generate", decode_ms, decode_n_tokens, decode_tpt, decode_tps))

    print("--------------------------------------------------------------------------------------")


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Inference Gemma4 llm model of RKNN")
    parser.add_argument("--rknn_path", type=str, help="rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument("--tokenizer_path", type=str, help="huggingface tokenizer path or tokenizer.json", required=False, default=TOKENIZER_PATH)
    parser.add_argument("--embed_path", type=str, help="token embedding path", required=False, default=EMBED_PATH)
    parser.add_argument("--per_layer_embed_path", type=str, help="per_layer_inputs embedding path", required=False, default=PER_LAYER_EMBED_PATH)
    parser.add_argument("--safetensors_path", type=str, help="rope_caches.safetensors path", required=False, default=SAFETENSORS_PATH)
    parser.add_argument("--max_context_len", type=int, help="max context len, only used for check", required=False, default=1024)
    parser.add_argument("--max_new_tokens", type=int, help="max new tokens", required=False, default=1024)
    parser.add_argument("--core_mask", type=lambda x: int(x, 0), help="npu core mask, e.g. 0xff", required=False, default=0xff)
    args = parser.parse_args()

    if LLMInputCallback is None:
        print("Current rknn3lite.api has no LLMInputCallback, please use the RKNN3Lite package that supports Gemma4 input callback")
        exit(-1)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Create RKNN object
    rknn = RKNN3Lite(llm_mode=True, verbose=True)

    # Load model
    print('--> Loading model')
    ret = rknn.load_rknn(args.rknn_path, args.rknn_path.replace(".rknn",".weight"))
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Init runtime first, query model info, then build LLM_ARGS and init LLM session.
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk1820', core_mask=args.core_mask)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    print('--> Query model info')
    llm_config = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_LLM_CONFIG)
    print("llm_config",llm_config)
    if llm_config is None:
        print('Query RKNN3_QUERY_LLM_CONFIG failed!')
        exit(-1)

    io_num = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_IN_OUT_NUM)
    if io_num is None:
        print('Query RKNN3_QUERY_IN_OUT_NUM failed!')
        exit(-1)

    print("len(tokenizer)=",len(tokenizer))
    vocab_size = int(getattr(llm_config, 'vocab_size', 0))
    if vocab_size <= 0:
        vocab_size = len(tokenizer)
    if len(tokenizer) != vocab_size:
        print(f"Warning: tokenizer vocab size ({len(tokenizer)}) != llm_config.vocab_size ({vocab_size}), use llm_config.vocab_size for embeddings")

    embeds_data = np.memmap(args.embed_path, dtype=np.float16, mode='r')
    embedding_dim = embeds_data.size // vocab_size
    embeds_data = embeds_data.reshape(vocab_size, embedding_dim)

    per_layer_embeds_data = np.memmap(args.per_layer_embed_path, dtype=np.float16, mode='r')
    per_layer_embedding_dim = per_layer_embeds_data.size // vocab_size
    per_layer_embeds_data = per_layer_embeds_data.reshape(vocab_size, per_layer_embedding_dim)

    print(f"vocab_size={vocab_size}, embedding_dim={embedding_dim}, per_layer_embedding_dim={per_layer_embedding_dim}")
    print(f"max_ctx_len={llm_config.max_ctx_len}, max_position_embeddings={llm_config.max_position_embeddings}")

    need_rope_cache = getattr(llm_config, 'rope_cache_host_storage', 0) != 0
    ext_indices = []
    for i in range(io_num.n_input):
        attr = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_INPUT_ATTR, index=i)
        if attr is None:
            print(f'Query RKNN3_QUERY_INPUT_ATTR failed! index={i}')
            exit(-1)

        name = attr.name
        if isinstance(name, bytes):
            name = name.split(b'\0', 1)[0].decode('utf-8', errors='ignore')
        elif isinstance(name, str):
            name = name.split('\0', 1)[0]
        else:
            try:
                name = ctypes.string_at(name).split(b'\0', 1)[0].decode('utf-8', errors='ignore')
            except Exception:
                name = bytes(name).split(b'\0', 1)[0].decode('utf-8', errors='ignore')

        if name == "per_layer_inputs":
            ext_indices.append(i)
        elif need_rope_cache and ("rope_cos_cache" in name or "rope_sin_cache" in name):
            ext_indices.append(i)

    if len(ext_indices) == 0:
        print('no ext input tensors found: per_layer_inputs or rope cache')
        exit(-1)

    if need_rope_cache:
        if not os.path.exists(args.safetensors_path):
            print('model requires rope_caches.safetensors, but safetensors_path is invalid')
            exit(-1)

        rope_file = open(args.safetensors_path, 'rb')
        # ACCESS_COPY 便于 ctypes.from_buffer 拿到底层地址；不写入文件，也不会把整文件提前复制到 Python 堆。
        rope_mmap = mmap.mmap(rope_file.fileno(), 0, access=mmap.ACCESS_COPY)
        rope_mmap_base_obj = ctypes.c_char.from_buffer(rope_mmap)
        rope_mmap_addr = ctypes.addressof(rope_mmap_base_obj)

        header_size = struct.unpack('<Q', rope_mmap[:8])[0]
        header = json.loads(rope_mmap[8:8 + header_size].decode('utf-8'))
        meta_index = json.loads(header['__metadata__']['index'])
        data_base = 8 + header_size

        for name in ROPE_CACHE_NAMES:
            meta_t = meta_index[name]
            t = header[name]
            shape = t['shape']
            offsets = t['data_offsets']
            if len(shape) != 5:
                print(f"Tensor {name}: expected 5-D NC1HWC2")
                exit(-1)
            rope_caches[name] = {
                "dtype": int(meta_t['dtype']),
                "layout": int(meta_t['layout']),
                "shape": shape,
                "offset": data_base + int(offsets[0])
            }
            print("Loaded %-24s  dtype=%-2d  shape=%s" % (name, rope_caches[name]["dtype"], shape))

    if args.max_context_len != getattr(llm_config, 'max_ctx_len', args.max_context_len):
        if args.max_context_len < llm_config.max_ctx_len:
            print(f"Warning: max_context_len ({args.max_context_len}) is less than llm_config.max_ctx_len ({llm_config.max_ctx_len}).")
            print(f"It's recommended to set --max_context_len to {llm_config.max_ctx_len}.")
        else:
            print(f"Error: max_context_len ({args.max_context_len}) is greater than llm_config.max_ctx_len ({llm_config.max_ctx_len}).")
            print(f"Please set --max_context_len to {llm_config.max_ctx_len}.")
            exit(-1)


    special_eos_id = [1, 50, 106] # Gemma4 拥有多个特殊 EOS 可从 tokenizer_config.json 查询到
    special_bos_id = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 2]

    LLM_ARGS = [{"max_new_tokens": args.max_new_tokens,
                 "top_k": 1, "top_p": 0.9,
                 "temperature": 1.0,
                 "repeat_penalty": 1.0,
                 "frequency_penalty": 0.0,
                 "presence_penalty": 0.0,
                 "vocab_size": vocab_size,
                 "special_eos_id": special_eos_id,
                 "special_bos_id": special_bos_id,
                 "max_context_len": llm_config.max_ctx_len,
                 "keep_history": 0,
                 "logits_name": b"logits_gathered"}
                ]

    print("\n=============================================================")
    print("%-32s: %-8d" % ("Max Context Length", getattr(llm_config, 'max_ctx_len', 0)))
    print("%-32s: %-8d" % ("Max Position Embeddings", getattr(llm_config, 'max_position_embeddings', 0)))
    print("%-32s: %s" % ("Model Type", getattr(llm_config, 'model_type', b'')))
    print("%-32s: %-8d" % ("Max New Tokens", args.max_new_tokens))
    print("=============================================================\n")

    # Callback
    callback = RKLLMCallback()

    callback.result_callback = LLMResultCallback(result_callback)
    callback.result_userdata = None

    callback.tokenizer_callback = LLMTokenizerCallback(tokenizer_callback)
    userdata = ctypes.py_object(tokenizer)
    callback.tokenizer_userdata = ctypes.cast(ctypes.pointer(userdata), ctypes.c_void_p)

    callback.embed_callback = LLMGetEmbedCallback(embed_callback)
    userdata = ctypes.py_object(embeds_data)
    callback.embed_userdata = ctypes.cast(ctypes.pointer(userdata), ctypes.c_void_p)

    callback.input_callback = LLMInputCallback(input_callback)
    userdata = ctypes.py_object(per_layer_embeds_data)
    callback.input_userdata = ctypes.cast(ctypes.pointer(userdata), ctypes.c_void_p)

    callback.output_callback = LLMOutputCallback(output_callback)
    userdata = ctypes.py_object(embeds_data)
    callback.output_userdata = ctypes.cast(ctypes.pointer(userdata), ctypes.c_void_p)

    llm_ext_input_indices = (ctypes.c_int * len(ext_indices))(*ext_indices)
    callback.input_tensors_index = llm_ext_input_indices
    callback.n_input_tensors = len(ext_indices)

    print('--> Init LLM session')
    ret = rknn.init_llm_session(llm_args=LLM_ARGS, llm_callback=callback)
    if ret != 0:
        print('Init llm session failed!')
        exit(ret)
    print('done')

    ret = rknn.set_chat_template(system_prompt, prompt_prefix, prompt_postfix)
    if ret != 0:
        print('Set chat template failed!')
        exit(ret)

    # LLM Inference
    print('--> inference gemma4 llm model')
    prompts = ["请解释一下相对论的基本概念？", "你是谁？", "介绍一下LLM模型的工作原理。"]
    for prompt in prompts:
        first_token = None
        ret, [n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time] = rknn.session_run(
            prompt=prompt,
            keep_history=0,
            max_new_tokens=args.max_new_tokens
        )
        if ret != 0:
            print('RKNN gemma4 llm inference failed!')
            exit(ret)

        ret = rknn.clear_kvcache(RKNN3KVCacheClearPolicy.RKNN3_KVCACHE_CLEAR_ALL)
        if ret != 0:
            print(f'Clear kvcache failed! ret={ret}')
        printf_perf(first_token, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time)

    print('done')

    rknn.release()
    rope_mmap_base_obj = None
    if rope_mmap is not None:
        rope_mmap.close()
    if rope_file is not None:
        rope_file.close()