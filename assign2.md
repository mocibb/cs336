# 作业二参考资料（整理中）


<img src="https://github.com/user-attachments/assets/c8370d37-f06e-40a3-a2c0-a034f3f6887d" alt="matmuls" width="400"/>


## benchmarking

###  
small尺寸下的时间(batch: 4, context_length: 256)

cuda情况 forward时间: 0.034s, backward时间: 0.11s

cpu情况 forward时间: 0.79s, backward时间: 2.43s

### nsys profile
1. 安装 [文档](https://developer.nvidia.com/nsight-systems)
2. 命令行 [参考文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
3. 加入nvtx的annotation可以帮助调试，作业指导上给了一个例子。
<img src="https://github.com/user-attachments/assets/ceecfd1c-ef78-43b1-871d-6931c78c1afa" alt="matmuls" width="400"/>

4. 运行 nsys profile
``` sh
uv run nsys profile  -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=none --cudabacktrace=true -x true -o result python benchmark.py
```
