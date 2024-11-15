export HF_ENDPOINT=https://hf-mirror.com

# huggingface-cli download --token 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 --resume-download meta-llama/Meta-Llama-3-8B
# huggingface-cli download --token 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 --resume-download meta-llama/Meta-Llama-3-8B-Instruct
# huggingface-cli download --token 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 --resume-download meta-llama/Meta-Llama-3-70B
# huggingface-cli download --token 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 --resume-download meta-llama/Meta-Llama-3-70B-Instruct --local-dir Llama-3-70B-Instruct --local-dir-use-symlinks False
# huggingface-cli download --token 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 --resume-download meta-llama/Meta-Llama-3.1-8B-Instruct





# huggingface-cli download --resume-download Qwen/Qwen2-7B-Instruct
# huggingface-cli download --resume-download NaughtyDog97/DFE-GPS-9B
# huggingface-cli download --resume-download 01-ai/Yi-1.5-34B-Chat
# huggingface-cli download --resume-download 01-ai/Yi-1.5-9B-Chat-16K
# huggingface-cli download --resume-download showlab/magvitv2
# huggingface-cli download --resume-download microsoft/phi-1_5
# huggingface-cli download --resume-download showlab/show-o-w-clip-vit
# huggingface-cli download --resume-download showlab/show-o-512x512
huggingface-cli download --resume-download openai/clip-vit-large-patch14-336 --local-dir pretrained_ckpt/clip-vit-large-patch14-336 --local-dir-use-symlinks False
# huggingface-cli download --repo-type dataset --resume-download MMInstruction/VLFeedback --local-dir data_backup/vl_dpo_data --local-dir-use-symlinks False