mkdir -p ./cache/Qwen2.5-0.5B-Instruct
./script/hfd.sh Qwen/Qwen2.5-0.5B-Instruct --local-dir ./cache/Qwen2.5-0.5B-Instruct --include "*.safetensors" "*.json"