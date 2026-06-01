import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com/")

import ctypes
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
    RKNN3QueryCmd,
    RKNN3KVCachePolicy,
    RKNN3KVCacheClearPolicy,
    dump_tensor_attr,
)


# ============================================= Global Runtime Data =============================================

tokenizer = None
embeds_data = None
rknn = None

first_token_time = None
stop_requested = False
stop_word = "概念"

# Keep ctypes userdata alive. Do not make them local-only.
_tokenizer_userdata = None
_embed_userdata = None


# ============================================= Callbacks =============================================

def _decode_safe(tokens):
    """Decode tokens and avoid printing broken UTF-8 replacement characters."""
    text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text.split("\ufffd", 1)[0] if "\ufffd" in text else text


def result_callback(userdata, result_ptr, state):
    """LLM result callback. Stop session when stop_word appears in generated text."""
    global first_token_time, stop_requested, rknn, stop_word

    if not hasattr(result_callback, "accumulated_tokens"):
        result_callback.accumulated_tokens = []
        result_callback.last_output_text = ""

    try:
        # 5: RKLLM_RUN_ERROR
        if state == 5:
            print("\n\nError occurred during inference", flush=True)
            return 0

        # 2/3/4: FINISH / STOP / MAX_NEW_TOKEN_REACHED
        if state in (2, 3, 4):
            if result_callback.accumulated_tokens:
                safe_text = _decode_safe(result_callback.accumulated_tokens)
                new_part = safe_text[len(result_callback.last_output_text):]
                if new_part:
                    print(new_part, end="", flush=True)

            result_callback.accumulated_tokens.clear()
            result_callback.last_output_text = ""

            msg = {
                2: "Finished",
                3: "Stop",
                4: "Max new token reached",
            }.get(state, "Unknown")
            print(f"\n\n--------------------{msg}--------------------", flush=True)
            return 0

        # 1: RKLLM_RUN_WAITING
        if state == 1:
            print("\n\nWaiting for UTF-8 encoded character", flush=True)
            return 0

        # 0: RKLLM_RUN_NORMAL
        if state == 0:
            n_tokens = result_ptr.contents.num_tokens
            new_tokens = [result_ptr.contents.token_ids[i] for i in range(n_tokens)]
            result_callback.accumulated_tokens.extend(new_tokens)

            if first_token_time is None:
                first_token_time = time.perf_counter()

            safe_text = _decode_safe(result_callback.accumulated_tokens)
            new_part = safe_text[len(result_callback.last_output_text):]
            if new_part:
                print(new_part, end="", flush=True)
                result_callback.last_output_text += new_part

                # C++ demo calls rknn3_session_stop(session) when piece == "概念".
                # Python decoding may join several tokens, so use substring matching.
                if (not stop_requested) and stop_word and (stop_word in result_callback.last_output_text):
                    stop_requested = True
                    if rknn is not None:
                        rknn.session_stop()

    except Exception as e:
        print(f"\n[Callback error: {e}]", flush=True)

    return 0


def tokenizer_callback(userdata, text_ptr, text_len, tokens_ptr, n_tokens_max):
    """Text -> token ids callback."""
    global tokenizer

    try:
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
    except Exception as e:
        print(f"Tokenizer callback failed: {e}")
        return -1


def embed_callback(userdata, tokens_ptr, num_tokens, embed, length):
    """Token ids -> embedding callback."""
    global embeds_data

    try:
        embedding_dim = embeds_data.shape[1]
        expected_len = int(num_tokens) * embedding_dim * np.dtype(np.float16).itemsize
        if int(length) != expected_len:
            print(f"invalid embed buffer, length={length}, expected={expected_len}")
            return -1

        dst = np.ctypeslib.as_array(
            ctypes.cast(embed, ctypes.POINTER(ctypes.c_uint16)),
            shape=(int(num_tokens) * embedding_dim,),
        ).view(np.float16)

        tokens = [int(tokens_ptr[i]) for i in range(int(num_tokens))]
        for token in tokens:
            if token < 0 or token >= embeds_data.shape[0]:
                print(f"invalid token id: {token}, vocab_size={embeds_data.shape[0]}")
                return -1

        dst[:] = embeds_data[tokens].reshape(-1)
        return 0
    except Exception as e:
        print(f"Embed callback failed: {e}")
        return -1


# ============================================= Helpers =============================================

def load_tokenizer_and_embedding(tokenizer_path, embed_path, vocab_size):
    """Load AutoTokenizer and reshape embedding.bin to [vocab_size, embedding_dim]."""
    global tokenizer, embeds_data

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    embeds_data = np.fromfile(embed_path, dtype=np.float16)
    if embeds_data.size == 0:
        raise RuntimeError(f"embedding file is empty: {embed_path}")
    if embeds_data.size % vocab_size != 0:
        raise RuntimeError(
            f"embedding size mismatch: {embeds_data.size} fp16 values cannot be divided by vocab_size={vocab_size}"
        )

    embedding_dim = embeds_data.size // vocab_size
    embeds_data = embeds_data.reshape(vocab_size, embedding_dim)
    return embedding_dim


def build_callback():
    """Build RKLLMCallback and keep userdata objects alive."""
    global _tokenizer_userdata, _embed_userdata

    callback = RKLLMCallback()

    callback.result_callback = LLMResultCallback(result_callback)
    callback.result_userdata = None

    callback.tokenizer_callback = LLMTokenizerCallback(tokenizer_callback)
    _tokenizer_userdata = ctypes.py_object(tokenizer)
    callback.tokenizer_userdata = ctypes.cast(ctypes.pointer(_tokenizer_userdata), ctypes.c_void_p)

    callback.embed_callback = LLMGetEmbedCallback(embed_callback)
    _embed_userdata = ctypes.py_object(embeds_data)
    callback.embed_userdata = ctypes.cast(ctypes.pointer(_embed_userdata), ctypes.c_void_p)

    return callback


def print_perf(first_token, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time):
    """Print performance statistics in the same style as the C++ demo."""
    if first_token is None:
        print("\nNo first token time recorded, skip performance statistics.")
        return

    print("\nPerformance Statistics:")
    print("-----------------------------------------------------------------------------------------")
    print(" %-10s | %-16s | %-8s | %-20s | %-20s " % (
        "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"))
    print("-----------------------------------------------------------------------------------------")

    prefill_ms = max(0.0, (first_token - llm_start_time) * 1000.0)
    if n_prefill_tokens == 0 or prefill_ms <= 0:
        prefill_tpt, prefill_tps = 0.0, 0.0
    else:
        prefill_tpt = prefill_ms / n_prefill_tokens
        prefill_tps = (n_prefill_tokens * 1000.0) / prefill_ms
    print(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f " % (
        "Prefill", prefill_ms, n_prefill_tokens, prefill_tpt, prefill_tps))

    decode_ms = max(0.0, (llm_end_time - first_token) * 1000.0)
    if n_decode_tokens == 0 or decode_ms <= 0:
        decode_tpt, decode_tps = 0.0, 0.0
    else:
        decode_tpt = decode_ms / n_decode_tokens
        decode_tps = (n_decode_tokens * 1000.0) / decode_ms
    print(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f " % (
        "Generate", decode_ms, n_decode_tokens, decode_tpt, decode_tps))

    print("-----------------------------------------------------------------------------------------")


def parse_args():
    parser = ArgumentParser(description="RKNN3 Session Stop Test - Python version")
    parser.add_argument("--rknn_path", type=str, help="rknn model path")
    parser.add_argument("--weight_path", type=str, help="rknn weight path")
    parser.add_argument("--tokenizer_path", type=str, help="HuggingFace tokenizer directory/name")
    parser.add_argument("--embed_path", type=str, help="embedding bin path")
    parser.add_argument("--max_context_len", type=int, help="max context length, should match model llm_config.max_ctx_len")
    parser.add_argument("--max_new_token", type=int, help="max new tokens")
    parser.add_argument("--core_mask", type=str, help="NPU core mask in hex, e.g. 0xff")
    parser.add_argument("--target", type=str, default="rk1820", help="target platform, default: rk1820")
    parser.add_argument("--prompt", type=str, default="请解释一下相对论的基本概念。", help="test prompt")
    parser.add_argument("--stop_word", type=str, default="概念", help="generated text containing this word will trigger session_stop")
    parser.add_argument("--keep_history", type=int, default=1, choices=[0, 1], help="whether to keep history")
    parser.add_argument("--no_verbose", action="store_true", help="disable RKNN verbose log")
    return parser.parse_args()


# ============================================= Main =============================================

def main():
    global rknn, first_token_time, stop_requested, stop_word

    args = parse_args()
    core_mask = int(args.core_mask, 16)
    stop_word = args.stop_word

    print("*******************************NEW TEST**********************************")

    rknn = RKNN3Lite(llm_mode=True, verbose=(not args.no_verbose))

    # Step 1: Load model
    print("--> Loading model")
    ret = rknn.load_rknn(args.rknn_path, args.weight_path)
    if ret != 0:
        print("Load model failed!")
        return ret
    print("done")

    # Step 2: Init runtime, but do not init LLM session yet.
    print("--> Init runtime environment")
    ret = rknn.init_runtime(target=args.target, core_mask=core_mask)
    if ret != 0:
        print("Init runtime environment failed!")
        return ret
    print("done")

    # Step 3: Query model information before init_llm_session.
    print("--> Query model info")
    io_num = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_IN_OUT_NUM)
    if io_num is None:
        print("Query model IO number failed!")
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
            f"\033[33mmax_context_len != llm_config.max_ctx_len, "
            f"max_context_len {args.max_context_len}, llm_config.max_ctx_len {llm_config.max_ctx_len}\033[0m"
        )
        print(f"\033[33mplease set <max_context_len> to {llm_config.max_ctx_len}\033[0m")
        return -1

    # Step 4: Load tokenizer and embedding after vocab_size is known.
    print("--> Loading tokenizer and embedding")
    embedding_dim = load_tokenizer_and_embedding(args.tokenizer_path, args.embed_path, llm_config.vocab_size)
    print(f"vocab_size={llm_config.vocab_size}, embedding_dim={embedding_dim}")
    print("done")

    print("\n=============================================================")
    print(f"{'Max Context Length':<32}: {llm_config.max_ctx_len:<8}")
    print(f"{'Max Position Embeddings':<32}: {llm_config.max_position_embeddings:<8}")
    print("=============================================================\n")

    # Step 5: Build LLM args.
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
        "keep_history":      int(args.keep_history),
        "logits_name":       b"output",
    }]

    # Step 6: Init LLM session.
    print("--> Init LLM session")
    callback = build_callback()
    ret = rknn.init_llm_session(llm_args=llm_args, llm_callback=callback)
    if ret != 0:
        print("Init LLM session failed!")
        return ret
    print("done")

    # Step 7: Set KV cache policy.
    ret = rknn.set_kvcache_policy(RKNN3KVCachePolicy.RKNN3_KVCACHE_POLICY_NORMAL)
    if ret != 0:
        print("Set kvcache policy failed!")
        return ret

    # Step 8: Run inference. session_stop will be triggered inside result_callback.
    print("\n--------------------Input[0]--------------------")
    print(args.prompt)
    print("\n--------------------Output----------------------")

    first_token_time = None
    stop_requested = False
    ret, perf = rknn.session_run(
        prompt=args.prompt,
        keep_history=bool(args.keep_history),
        max_new_tokens=args.max_new_token,
    )
    if ret != 0:
        print("RKNN llm inference failed!")
        return ret

    # Query state after run, same as C++ demo.
    state = rknn.session_query_state()
    if state is not None:
        if state.n_total_tokens >= (state.n_max_tokens - args.max_new_token):
            ret = rknn.clear_kvcache(RKNN3KVCacheClearPolicy.RKNN3_KVCACHE_CLEAR_ALL)
            if ret != 0:
                print("Clear kvcache failed!")
                return ret

    if len(perf) == 4:
        n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time = perf
        print_perf(first_token_time, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time)

    print("done")
    return 0


if __name__ == "__main__":
    ret_code = 0
    try:
        ret_code = main()
    finally:
        if rknn is not None:
            rknn.release()
        print("*******************************END TEST**********************************")
    raise SystemExit(ret_code)
