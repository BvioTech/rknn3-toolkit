import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

import ctypes
import threading
import time
from argparse import ArgumentParser

import numpy as np
from transformers import AutoTokenizer

from rknn3lite.api import (
    RKNN3Lite,
    RKLLMCallback,
    LLMResultCallback,
    LLMGetEmbedCallback,
    LLMTokenizerCallback,
)
from rknn3lite.api.rknn3_types import (
    LLMCallState,
    RKNN3KVCacheClearPolicy,
    RKNN3KVCachePolicy,
    RKNN3QueryCmd,
    dump_tensor_attr,
)

# ============================================= Default Config =============================================

RKNN_MODEL     = "Qwen2.5-0.5B-Instruct.rknn"
WEIGHT_MODEL   = "Qwen2.5-0.5B-Instruct.weight"
EMBED_PATH     = "Qwen2.5-0.5B-Instruct.embed.bin"
TOKENIZER_PATH = "Qwen/Qwen2.5-0.5B-Instruct"

system_prompt  = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
prompt_prefix  = "<|im_start|>user\n"
prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n"

tokenizer = None
embeds_data = None
first_token_time = None

decode_started_event = threading.Event()

# Keep ctypes callback/userdata objects alive.
_callback_refs = []


# ============================================= Callbacks =============================================

def reset_decode_status():
    global first_token_time
    first_token_time = None
    decode_started_event.clear()
    result_callback.accumulated_tokens = []
    result_callback.last_output_text = ""


def result_callback(userdata, result_ptr, state):
    """LLM result callback. Print tokens and record first decode token time."""
    global tokenizer, first_token_time

    state = int(state)

    if not hasattr(result_callback, "accumulated_tokens"):
        result_callback.accumulated_tokens = []
        result_callback.last_output_text = ""

    def decode_safe(tokens):
        text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text.split("\ufffd", 1)[0] if "\ufffd" in text else text

    if state == LLMCallState.RKLLM_RUN_ERROR:
        print("\n\nError occurred during inference", flush=True)
        return 0

    if state == LLMCallState.RKLLM_RUN_PAUSE:
        print("\n\n-----------------------Pause---------------------", flush=True)
        return 0

    if state == LLMCallState.RKLLM_RUN_RESUME:
        print("\n\n-----------------------Resume---------------------", flush=True)
        return 0

    if state in (
        LLMCallState.RKLLM_RUN_FINISH,
        LLMCallState.RKLLM_RUN_STOP,
        LLMCallState.RKLLM_RUN_MAX_NEW_TOKEN_REACHED,
    ):
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

        msg = {
            LLMCallState.RKLLM_RUN_FINISH: "Finished",
            LLMCallState.RKLLM_RUN_STOP: "Stop",
            LLMCallState.RKLLM_RUN_MAX_NEW_TOKEN_REACHED: "Max new token reached",
        }.get(LLMCallState(state), "Unknown")
        print(f"\n\n--------------------{msg}--------------------", flush=True)
        return 0

    if state == LLMCallState.RKLLM_RUN_WAITING:
        print("\n\nWaiting for UTF-8 encoded character", flush=True)
        return 0

    if state == LLMCallState.RKLLM_RUN_NORMAL:
        if first_token_time is None:
            first_token_time = time.perf_counter()
            decode_started_event.set()

        n_tokens = result_ptr.contents.num_tokens
        new_tokens = [result_ptr.contents.token_ids[i] for i in range(n_tokens)]
        result_callback.accumulated_tokens.extend(new_tokens)

        try:
            safe_text = decode_safe(result_callback.accumulated_tokens)
            new_part = safe_text[len(result_callback.last_output_text):]
            if new_part:
                print(new_part, end="", flush=True)
                result_callback.last_output_text += new_part
        except Exception as e:
            print(f"\n[Temp decode error: {e}], waiting for more tokens", flush=True)

    return 0


def tokenizer_callback(userdata, text_ptr, text_len, tokens_ptr, n_tokens_max):
    """Convert prompt text to token ids."""
    global tokenizer

    text = ctypes.string_at(text_ptr, text_len).decode("utf-8", errors="ignore")
    inputs = tokenizer(text, return_tensors="np", truncation=True)
    tokens = inputs["input_ids"][0][:n_tokens_max]

    n_tokens = len(tokens)
    if n_tokens <= 0:
        print(f"Tokenizer failed for {text}")
        return n_tokens

    for i, token in enumerate(tokens):
        tokens_ptr[i] = int(token)

    return n_tokens


def embed_callback(userdata, tokens_ptr, num_tokens, embed, length):
    """Copy token embeddings from embedding.bin into RKNN3 embed buffer."""
    global embeds_data

    embedding_dim = embeds_data.shape[1]
    expected_len = num_tokens * embedding_dim * np.dtype(np.float16).itemsize
    if length != expected_len:
        print(f"invalid embed buffer, expected {expected_len}, got {length}")
        return -1

    tokens = [int(tokens_ptr[i]) for i in range(num_tokens)]
    if tokens and (min(tokens) < 0 or max(tokens) >= embeds_data.shape[0]):
        print("invalid token id in embed callback")
        return -1

    dst = np.ctypeslib.as_array(
        ctypes.cast(embed, ctypes.POINTER(ctypes.c_uint16)),
        shape=(num_tokens * embedding_dim,),
    ).view(np.float16)
    dst[:] = embeds_data[tokens].ravel()
    return 0


# ============================================= Utilities =============================================

def parse_args():
    parser = ArgumentParser(description="RKNN3 Session Test - pause/resume Python version")
    parser.add_argument("--rknn_path",      type=str, default=RKNN_MODEL,     help="rknn model path")
    parser.add_argument("--weight_path",    type=str, default=None,           help="rknn weight path")
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH, help="HuggingFace tokenizer directory/path")
    parser.add_argument("--embed_path",     type=str, default=EMBED_PATH,     help="embedding bin path")
    parser.add_argument("--max_context_len", type=int, default=1024,          help="max context length")
    parser.add_argument("--max_new_token",  type=int, default=256,            help="max new tokens")
    parser.add_argument("--core_mask",      type=str, default="0xff",         help="NPU core mask in hex")
    parser.add_argument("--pause_seconds",  type=float, default=5.0,          help="pause duration for each pause/resume stage")
    parser.add_argument("--target",         type=str, default="rk1820",       help="target platform")

    args = parser.parse_args()
    if args.weight_path is None:
        args.weight_path = args.rknn_path.replace(".rknn", ".weight")

    return args


def load_tokenizer_and_embedding(tokenizer_path, embed_path, vocab_size):
    """Load tokenizer and reshape embedding table."""
    global tokenizer, embeds_data

    if tokenizer_path.endswith(".gguf"):
        print(
            "Warning: this Python demo uses transformers.AutoTokenizer. "
            "Please pass the HuggingFace tokenizer directory instead of tokenizer.gguf if loading fails."
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    data = np.fromfile(embed_path, dtype=np.float16)
    if data.size == 0:
        raise RuntimeError(f"empty embedding file: {embed_path}")
    if data.size % vocab_size != 0:
        raise RuntimeError(
            f"embedding size mismatch: data.size={data.size}, vocab_size={vocab_size}"
        )

    embedding_dim = data.size // vocab_size
    embeds_data = np.ascontiguousarray(data.reshape(vocab_size, embedding_dim))
    return embedding_dim


def build_callback():
    """Build RKLLMCallback and keep all ctypes references alive."""
    callback = RKLLMCallback()

    result_cb = LLMResultCallback(result_callback)
    tokenizer_cb = LLMTokenizerCallback(tokenizer_callback)
    embed_cb = LLMGetEmbedCallback(embed_callback)

    callback.result_callback = result_cb
    callback.result_userdata = None

    callback.tokenizer_callback = tokenizer_cb
    tokenizer_ud = ctypes.py_object(tokenizer)
    callback.tokenizer_userdata = ctypes.cast(ctypes.pointer(tokenizer_ud), ctypes.c_void_p)

    callback.embed_callback = embed_cb
    embed_ud = ctypes.py_object(embeds_data)
    callback.embed_userdata = ctypes.cast(ctypes.pointer(embed_ud), ctypes.c_void_p)

    _callback_refs.extend([callback, result_cb, tokenizer_cb, embed_cb, tokenizer_ud, embed_ud])
    return callback


def print_perf(start_time, first_time, end_time, state):
    if first_time is None:
        print("\nPerformance Statistics: first token was not received, skip perf printing")
        return

    prefill_n_tokens = int(state.n_prefill_tokens)
    decode_n_tokens = int(state.n_decode_tokens)

    prefill_ms = (first_time - start_time) * 1000.0
    decode_ms = (end_time - first_time) * 1000.0

    prefill_tpt = 0.0 if prefill_n_tokens == 0 else prefill_ms / prefill_n_tokens
    decode_tpt = 0.0 if decode_n_tokens == 0 else decode_ms / decode_n_tokens

    prefill_tps = 0.0 if prefill_ms <= 0 or prefill_n_tokens == 0 else prefill_n_tokens * 1000.0 / prefill_ms
    decode_tps = 0.0 if decode_ms <= 0 or decode_n_tokens == 0 else decode_n_tokens * 1000.0 / decode_ms

    print("\nPerformance Statistics: ")
    print("-----------------------------------------------------------------------------------------")
    print(" %-10s | %-16s | %-8s | %-20s | %-20s " % (
        "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"))
    print("-----------------------------------------------------------------------------------------")
    print(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f " % (
        "Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps))
    print(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f " % (
        "Generate", decode_ms, decode_n_tokens, decode_tpt, decode_tps))
    print("-----------------------------------------------------------------------------------------")


def make_long_prompt():
    prompt = "You are a helpful assistant. Please summarize the following repeated paragraph into key points: "
    for _ in range(60):
        prompt += "Relativity describes how space, time, and gravity are related in modern physics. "
    return prompt


# ============================================= Thread Functions =============================================

def infer_thread_func(rknn, prompt, max_new_token, result_holder, infer_running_event, infer_done_event):
    print("[Infer Thread] Start inference")
    infer_running_event.set()

    ret, perf = rknn.session_run(
        prompt=prompt,
        keep_history=True,
        max_new_tokens=max_new_token,
        enable_thinking=False,
    )

    result_holder["ret"] = ret
    result_holder["perf"] = perf

    infer_running_event.clear()
    infer_done_event.set()
    print("\n[Infer Thread] Inference finished")


def control_thread_func(rknn, flags, infer_running_event, infer_done_event, pause_seconds):
    # Wait until inference really starts.
    while not infer_running_event.is_set() and not infer_done_event.is_set():
        time.sleep(0.001)

    if infer_done_event.is_set():
        print("[Control Thread] inference finished before control stage")
        return

    # Stage 1: pause/resume during prefill.
    time.sleep(0.1)

    print("\n[Control Thread] ======= PREFILL PAUSE SESSION =======")
    ret = rknn.session_pause()
    if ret != 0:
        print(f"[Control Thread] prefill pause failed, ret={ret}")
        flags["prefill_pause_ok"] = False
    else:
        print("[Control Thread] session paused (prefill)")
        flags["prefill_pause_ok"] = True

    print(f"[Control Thread] sleep {pause_seconds:g} second (prefill paused)...")
    time.sleep(pause_seconds)

    print("[Control Thread] ======= PREFILL RESUME SESSION =======")
    ret = rknn.session_resume()
    if ret != 0:
        print(f"[Control Thread] prefill resume failed, ret={ret}")
        flags["prefill_resume_ok"] = False
    else:
        print("[Control Thread] session resumed (prefill)")
        flags["prefill_resume_ok"] = True

    # Stage 2: wait until first decode token, then pause/resume again.
    decode_started = decode_started_event.wait(timeout=10.0)
    if infer_done_event.is_set():
        print("[Control Thread] inference finished before decode pause stage")
        return
    if not decode_started:
        print("[Control Thread] decode did not start within timeout, skip decode pause")
        return

    time.sleep(0.1)

    print("\n[Control Thread] ======= DECODE PAUSE SESSION =======")
    ret = rknn.session_pause()
    if ret != 0:
        print(f"[Control Thread] decode pause failed, ret={ret}")
        flags["decode_pause_ok"] = False
    else:
        print("[Control Thread] session paused (decode)")
        flags["decode_pause_ok"] = True

    print(f"[Control Thread] sleep {pause_seconds:g} second (decode paused)...")
    time.sleep(pause_seconds)

    print("[Control Thread] ======= DECODE RESUME SESSION =======")
    ret = rknn.session_resume()
    if ret != 0:
        print(f"[Control Thread] decode resume failed, ret={ret}")
        flags["decode_resume_ok"] = False
    else:
        print("[Control Thread] session resumed (decode)")
        flags["decode_resume_ok"] = True


# ============================================= Main =============================================

def main():
    args = parse_args()
    core_mask = int(args.core_mask, 16)

    print("*******************************NEW TEST**********************************")

    rknn = RKNN3Lite(llm_mode=True, verbose=True)

    try:
        print("--> Loading model")
        ret = rknn.load_rknn(args.rknn_path, args.weight_path)
        if ret != 0:
            print("Load model failed!")
            return ret
        print("done")

        print("--> Init runtime environment")
        ret = rknn.init_runtime(target=args.target, core_mask=core_mask)
        if ret != 0:
            print("Init runtime environment failed!")
            return ret
        print("done")

        print("--> Query model info")
        io_num = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_IN_OUT_NUM)
        if io_num is None:
            print("Query IO number failed!")
            return -1
        print(f"model input num: {io_num.n_input}, output num: {io_num.n_output}")

        for i in range(io_num.n_output):
            output_attr = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_OUTPUT_ATTR, index=i)
            if output_attr is not None:
                dump_tensor_attr(output_attr, prefix="output")

        llm_config = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_LLM_CONFIG)
        if llm_config is None:
            print("Query LLM config failed!")
            return -1

        if args.max_context_len != llm_config.max_ctx_len:
            print(
                f"max_context_len != llm_config.max_ctx_len, "
                f"max_context_len {args.max_context_len}, llm_config.max_ctx_len {llm_config.max_ctx_len}"
            )
            print(f"please set <max_context_len> to {llm_config.max_ctx_len}")
            return -1

        print("--> Loading tokenizer and embedding")
        embedding_dim = load_tokenizer_and_embedding(args.tokenizer_path, args.embed_path, llm_config.vocab_size)
        print(f"vocab_size={llm_config.vocab_size}, embedding_dim={embedding_dim}")
        print(f"max_ctx_len={llm_config.max_ctx_len}, max_position_embeddings={llm_config.max_position_embeddings}")
        print("done")

        llm_args = [{
            "max_new_tokens":    args.max_new_token,
            "top_k":             1,
            "top_p":             0.9,
            "temperature":       1.0,
            "repeat_penalty":    1.1,
            "frequency_penalty": 0.0,
            "presence_penalty":  0.0,
            "vocab_size":        int(llm_config.vocab_size),
            "special_eos_id":    tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1,
            "max_context_len":   int(llm_config.max_ctx_len),
            "keep_history":      1,
            "logits_name":       b"output",
        }]

        print("--> Init LLM session")
        callback = build_callback()
        ret = rknn.init_llm_session(llm_args=llm_args, llm_callback=callback)
        if ret != 0:
            print("Init LLM session failed!")
            return ret
        print("done")

        # ret = rknn.set_chat_template(system_prompt, prompt_prefix, prompt_postfix)
        # if ret != 0:
        #     print("Set chat template failed!")
        #     return ret

        ret = rknn.set_kvcache_policy(RKNN3KVCachePolicy.RKNN3_KVCACHE_POLICY_NORMAL)
        if ret != 0:
            print("Set kvcache policy failed!")
            return ret

        print("\n=============================================================")
        print(f"{'Max Context Length':<32}: {llm_config.max_ctx_len:<8}")
        print(f"{'Max Position Embeddings':<32}: {llm_config.max_position_embeddings:<8}")
        print("=============================================================\n")

        prompt = make_long_prompt()
        print("\n--------------------Input[0]--------------------")
        print(prompt)
        print("\n--------------------Output----------------------")

        reset_decode_status()
        flags = {
            "prefill_pause_ok": False,
            "prefill_resume_ok": False,
            "decode_pause_ok": False,
            "decode_resume_ok": False,
        }
        infer_result = {"ret": -1, "perf": []}
        infer_running_event = threading.Event()
        infer_done_event = threading.Event()

        start_time = time.perf_counter()

        control_thread = threading.Thread(
            target=control_thread_func,
            args=(rknn, flags, infer_running_event, infer_done_event, args.pause_seconds),
            daemon=True,
        )
        infer_thread = threading.Thread(
            target=infer_thread_func,
            args=(rknn, prompt, args.max_new_token, infer_result, infer_running_event, infer_done_event),
            daemon=True,
        )

        control_thread.start()
        infer_thread.start()

        control_thread.join()
        infer_thread.join()

        end_time = time.perf_counter()

        if infer_result["ret"] != 0:
            print(f"RKNN llm inference failed, ret={infer_result['ret']}")
            return infer_result["ret"]

        state = rknn.session_query_state()
        if state is None:
            print("rknn session query state failed!")
            return -1

        if state.n_total_tokens >= (state.n_max_tokens - args.max_new_token):
            ret = rknn.clear_kvcache(RKNN3KVCacheClearPolicy.RKNN3_KVCACHE_CLEAR_ALL)
            if ret != 0:
                print(f"clear kvcache failed, ret={ret}")
                return ret

        print_perf(start_time, first_token_time, end_time, state)

        prefill_pause_resume_ok = flags["prefill_pause_ok"] and flags["prefill_resume_ok"]
        decode_pause_resume_ok = flags["decode_pause_ok"] and flags["decode_resume_ok"]
        test_passed = prefill_pause_resume_ok and decode_pause_resume_ok

        print("\nPause/Resume Test Status:")
        print(f"  prefill pause/resume: {'SUCCESS' if prefill_pause_resume_ok else 'FAILED'}")
        print(f"  decode  pause/resume: {'SUCCESS' if decode_pause_resume_ok else 'FAILED'}")
        print(f"  overall test         : {'SUCCESS' if test_passed else 'FAILED'}")

        return 0

    finally:
        rknn.release()
        print("*******************************END TEST**********************************")


if __name__ == "__main__":
    raise SystemExit(main())
