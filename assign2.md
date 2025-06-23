# 作业二参考资料（整理中）


<img src="https://github.com/user-attachments/assets/c8370d37-f06e-40a3-a2c0-a034f3f6887d" alt="matmuls" width="400"/>


## benchmarking

### nsys profile
1. 安装 [文档](https://developer.nvidia.com/nsight-systems)
2. 命令行 [参考文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
3. 运行 nsys profile
``` sh
uv run nsys profile  -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=none --cudabacktrace=true -x true -o result python benchmark.py
```
