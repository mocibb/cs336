# 作业二参考资料（整理中）


<img src="https://github.com/user-attachments/assets/c8370d37-f06e-40a3-a2c0-a034f3f6887d" alt="matmuls" width="400"/>


## benchmarking

###  
small尺寸下的时间(batch: 4, context_length: 256)

cuda情况 forward时间: 0.034s(0.00036s), backward时间: 0.11s(0.00029s)

cpu情况 forward时间: 0.79s, backward时间: 2.43s

关于warmup的次数，做不做看起来有很大区别，只要做了warmup次数区别并不大。

### nsys profile
1. 安装 [文档](https://developer.nvidia.com/nsight-systems)
2. 命令行 [参考文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
3. 加入nvtx的annotation可以帮助调试，作业指导上给了一个例子。
<img src="https://github.com/user-attachments/assets/ceecfd1c-ef78-43b1-871d-6931c78c1afa" alt="matmuls" width="400"/>

4. 运行 nsys profile
``` sh
uv run nsys profile  -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=none -python-backtrace=cuda --cudabacktrace=true -x true -o result python benchmark.py
```
这里选择trace cuda, nvtx和osrt(OS runtime library)
--cudabacktrace=true需要有管理员权限。

<img src="https://github.com/user-attachments/assets/f244eedc-4429-498d-92c9-0309e4736d1c" alt="matmuls" width="2000"/>
