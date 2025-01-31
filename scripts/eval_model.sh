OUTDIR="evalres"
RUNNAME="trialrun"
# MODEL_PATH="model_cache/qwen-1.5B_gloabl_step_50"
MODEL_PATH="model_cache/qwen1.5B/qwen1.5B_global_step_50"
MATH_PATH="HuggingFaceH4/MATH-500"
PROMPT_FILE="prompt_templates/qwen_jxhe.py"

export TOKENIZERS_PARALLELISM=false
python verl/evaluation/evalmath_main.py \
    --outdir $OUTDIR \
    --run_name $RUNNAME \
    --model_name $MODEL_PATH \
    --data_path $MATH_PATH \
    --input_key problem \
    --temperature 1.0 \
    --top_p 1.0 \
    --n 16 \
    --max_tokens 3000 \
    --prompt_file $PROMPT_FILE \
    --eval_fn deepseek_math \
    --max_sample 4 \