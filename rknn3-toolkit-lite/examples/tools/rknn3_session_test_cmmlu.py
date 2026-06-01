import os
import sys
import json
import ctypes
import numpy as np
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
from transformers import AutoTokenizer
from rknn3lite.api import RKNN3Lite, RKLLMCallback, LLMResultCallback, LLMGetEmbedCallback, LLMTokenizerCallback
from rknn3lite.api.rknn3_types import RKNN3QueryCmd, RKNN3KVCachePolicy, RKNN3KVCacheClearPolicy, RKNN3LLMTaskType, dump_tensor_attr

# ============================================= Default Config =============================================

RKNN_MODEL     = 'Qwen2.5-0.5B-Instruct.rknn'
WEIGHT_MODEL   = 'Qwen2.5-0.5B-Instruct.weight'
EMBED_PATH     = 'Qwen2.5-0.5B-Instruct.embed.bin'
TOKENIZER_PATH = 'Qwen/Qwen2.5-0.5B-Instruct'
DATASET_PATH   = './llm_dataset/cmmlu_5shot.json'

tokenizer   = None
embeds_data = None
first_token = None

model_answer = ""

# ============================================= Callbacks =============================================

def result_callback(userdata, result_ptr, state):
    """Result callback - captures the model answer token text."""
    global tokenizer, first_token, model_answer

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
                    model_answer = safe_text
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
                model_answer = safe_text
        except Exception as e:
            print(f"\n[Temp decode error: {e}], waiting for more tokens", flush=True)
            return 0

    return 0


def tokenizer_callback(userdata, text_ptr, text_len, tokens_ptr, n_tokens_max):
    global tokenizer
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

# ============================================= Performance =============================================

def printf_perf(first_token_time, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time):
    prefill_ms = (first_token_time - llm_start_time) * 1000.0
    if n_prefill_tokens == 0:
        prefill_tpt, prefill_tps = 0.0, 0.0
    else:
        prefill_tpt = prefill_ms / n_prefill_tokens
        prefill_tps = (n_prefill_tokens * 1000.0) / prefill_ms

    decode_ms = (llm_end_time - first_token_time) * 1000.0
    if n_decode_tokens == 0:
        decode_tpt, decode_tps = 0.0, 0.0
    else:
        decode_tpt = decode_ms / n_decode_tokens
        decode_tps = (n_decode_tokens * 1000.0) / decode_ms

    print(f"\nTTFT: {prefill_ms / 1000.0:.3f} s ({prefill_ms:.3f} ms)")
    total_ms = (llm_end_time - llm_start_time) * 1000.0
    print(f"Total time: {total_ms / 1000.0:.2f} s")

# ============================================= Main =============================================

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="RKNN3 Session CMMLU Test - Evaluates model accuracy on the CMMLU 5-shot dataset.\n"
                    "Example: python rknn3_session_test_cmmlu.py "
                    "--rknn_path Qwen2.5-0.5B.rknn --tokenizer_path Qwen/Qwen2.5-0.5B-Instruct "
                    "--embed_path Qwen2.5-0.5B.embed.bin --max_new_tokens 1 --core_mask 0xff")
    parser.add_argument("--rknn_path",       type=str, default=RKNN_MODEL,     help="rknn model path")
    parser.add_argument("--tokenizer_path",  type=str, default=TOKENIZER_PATH, help="huggingface tokenizer path")
    parser.add_argument("--embed_path",      type=str, default=EMBED_PATH,     help="embedding bin path")
    parser.add_argument("--dataset_path",    type=str, default=DATASET_PATH,   help="cmmlu 5-shot json dataset path")
    parser.add_argument("--max_new_tokens",  type=int, default=256,            help="max new tokens")
    parser.add_argument("--max_context_len", type=int, default=0,              help="max context length (0=use model default)")
    parser.add_argument("--core_mask",       type=str, default="0xff",         help="NPU core mask in hex")
    args = parser.parse_args()

    core_mask = int(args.core_mask, 16)
    weight_path = args.rknn_path.replace('.rknn', '.weight')

    # Load CMMLU dataset
    print(f'--> Loading CMMLU dataset from {args.dataset_path}')
    if not os.path.exists(args.dataset_path):
        print(f'Failed to open {args.dataset_path}')
        exit(-1)
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'Loaded {len(data)} questions')

    # Load tokenizer & embedding
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    embeds_data = np.fromfile(args.embed_path, dtype=np.float16)

    print("*******************************NEW TEST**********************************")

    # Create RKNN object
    rknn = RKNN3Lite(llm_mode=True, verbose=True)

    # Step 1: Load model
    print('--> Loading model')
    ret = rknn.load_rknn(args.rknn_path, weight_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Step 2: Init runtime (without llm_args, separated flow)
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk1820', core_mask=core_mask)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Step 3: Query model info (before init_llm_session)
    print('--> Query model info')
    io_num = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_IN_OUT_NUM)
    print(f"model input num: {io_num.n_input}, output num: {io_num.n_output}")

    for i in range(io_num.n_output):
        output_attr = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_OUTPUT_ATTR, index=i)
        dump_tensor_attr(output_attr, prefix="output")

    llm_config = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_LLM_CONFIG)
    vocab_size    = llm_config.vocab_size
    embedding_dim = embeds_data.size // vocab_size
    embeds_data   = embeds_data.reshape(vocab_size, embedding_dim)
    print(f"vocab_size={vocab_size}, embedding_dim={embedding_dim}")
    print(f"max_ctx_len={llm_config.max_ctx_len}, max_position_embeddings={llm_config.max_position_embeddings}")

    # Validate max_context_len
    max_context_len = args.max_context_len if args.max_context_len > 0 else llm_config.max_ctx_len
    if max_context_len != llm_config.max_ctx_len:
        if max_context_len < llm_config.max_ctx_len:
            print(f"\033[33mWarning: max_context_len ({max_context_len}) is less than "
                  f"llm_config.max_ctx_len ({llm_config.max_ctx_len}).\033[0m")
            print(f"\033[33mIt's recommended to set <max_context_len> to {llm_config.max_ctx_len}.\033[0m")
        elif max_context_len > llm_config.max_ctx_len:
            print(f"\033[33mError: max_context_len ({max_context_len}) is greater than "
                  f"llm_config.max_ctx_len ({llm_config.max_ctx_len}).\033[0m")
            print(f"\033[33mPlease set <max_context_len> to {llm_config.max_ctx_len}.\033[0m")
            rknn.release()
            exit(-1)

    # Step 4: Build LLM args (using queried info)
    # For CMMLU dataset test: repeat_penalty=1.0, no chat template
    LLM_ARGS = [{
        "max_new_tokens":    args.max_new_tokens,
        "top_k":             1,
        "top_p":             0.9,
        "temperature":       1.0,
        "repeat_penalty":    1.0,
        "frequency_penalty": 0.0,
        "presence_penalty":  0.0,
        "vocab_size":        vocab_size,
        "special_eos_id":    tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1,
        "max_context_len":   llm_config.max_ctx_len,
        "keep_history":      0,
        "logits_name":       b"output",
    }]

    # Step 5: Build callback
    callback = RKLLMCallback()
    callback.result_callback = LLMResultCallback(result_callback)
    callback.result_userdata = None

    callback.tokenizer_callback = LLMTokenizerCallback(tokenizer_callback)
    _tok_ud = ctypes.py_object(tokenizer)
    callback.tokenizer_userdata = ctypes.cast(ctypes.pointer(_tok_ud), ctypes.c_void_p)

    callback.embed_callback = LLMGetEmbedCallback(embed_callback)
    _emb_ud = ctypes.py_object(embeds_data)
    callback.embed_userdata = ctypes.cast(ctypes.pointer(_emb_ud), ctypes.c_void_p)

    # Step 6: Init LLM session (separated from init_runtime)
    print('--> Init LLM session')
    ret = rknn.init_llm_session(llm_args=LLM_ARGS, llm_callback=callback)
    if ret != 0:
        print('Init LLM session failed!')
        exit(ret)
    print('done')

    # Step 7: Set chat template (empty for CMMLU dataset test, same as C++ version)
    ret = rknn.set_chat_template("", "", "")
    if ret != 0:
        print('Set chat template failed!')
        exit(ret)

    # Print model config
    task_type_str = ("RKNN3_LLM_TASK_GENERATE"
                     if llm_config.task_type == RKNN3LLMTaskType.RKNN3_LLM_TASK_GENERATE
                     else "RKNN3_LLM_TASK_EMBEDDING")
    model_type_str = (llm_config.model_type.decode('utf-8', errors='ignore')
                      if llm_config.model_type else "Unknown")

    print()
    print("=============================================================")
    print("%*s" % (38, "Model Config"))
    print("=============================================================")
    print("%-32s: %-8d" % ("Max Context Length",       llm_config.max_ctx_len))
    print("%-32s: %-8d" % ("Max Position Embeddings",  llm_config.max_position_embeddings))
    print("%-32s: %s"   % ("Model Type",               model_type_str))
    print("%-32s: %s"   % ("Task Type",                task_type_str))
    print("%-32s: %-8d" % ("Max New Tokens",           args.max_new_tokens))
    print("=============================================================")
    print()

    # Step 8: Run CMMLU evaluation
    test_num = 0
    correct_num = 0

    for item in data:
        cur_prompt = item["question"]
        ref_answer = item["answer"]

        print(f"\n--------------------Input[{test_num}]--------------------")
        print(cur_prompt)
        print("\n--------------------Output----------------------")

        # Reset state for each question
        first_token = None
        model_answer = ""
        result_callback.accumulated_tokens = []
        result_callback.last_output_text = ""

        # Run inference (keep_history=False for each independent question)
        ret, [n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time] = rknn.session_run(
            prompt=cur_prompt,
            keep_history=False,
            max_new_tokens=args.max_new_tokens,
        )
        if ret != 0:
            print(f'rknn3_session_run failed, ret={ret}')
            break

        # Print performance for this question
        if first_token is not None:
            printf_perf(first_token, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time)

        print("\n-------------------------------------------------\n")

        test_num += 1

        # Compare model answer with reference answer
        # Strip whitespace for robust comparison (model may output "A" or " A" etc.)
        answer_stripped = model_answer.strip()
        if answer_stripped == ref_answer.strip():
            correct_num += 1

        # Print progress statistics
        accuracy = float(correct_num * 100.0 / test_num)
        print(f"Progress: {test_num}/{len(data)} | Correct: {correct_num} | Accuracy: {accuracy:.2f}%")
        sys.stdout.flush()

    # Print final results
    print("\n=============================================================")
    print("                     CMMLU Test Results")
    print("=============================================================")
    print(f"Total Questions : {test_num}")
    print(f"Correct Answers : {correct_num}")
    if test_num > 0:
        print(f"Accuracy        : {float(correct_num * 100.0 / test_num):.2f}%")
    print("=============================================================")

    # Step 9: Release
    rknn.release()

    print("\n*******************************END TEST**********************************")
    sys.stdout.flush()
