# minimind

minimind in lumen！



| Model Name        | params | len_vocab | rope_theta | n_layers | d_model | kv_heads | q_heads | share+route |
|-------------------|--------|-----------|------------|----------|---------|----------|---------|-------------|
| MiniMind2-Small   | 26M    | 6400      | 1e6        | 8        | 512     | 2        | 8       | -           |
| MiniMind2-MoE     | 145M   | 6400      | 1e6        | 8        | 640     | 2        | 8       | 1+4         |
| MiniMind2         | 104M   | 6400      | 1e6        | 16       | 768     | 2        | 8       | -           |
| minimind-v1-small | 26M    | 6400      | 1e4        | 8        | 512     | 8        | 16      | -           |
| minimind-v1-moe   | 4×26M  | 6400      | 1e4        | 8        | 512     | 8        | 16      | 1+4         |
| minimind-v1       | 108M   | 6400      | 1e4        | 16       | 768     | 8        | 16      | -           |



## References

- https://github.com/jingyaogong/minimind
- https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files