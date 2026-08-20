# VERL 环境复现

配置 [VERL](https://verl.readthedocs.io/en/latest/index.html) 开发框架：

```bash
# 拉取基础开发环境
docker pull verlai/verl:sgl0512.dev4

# 挂载数据并启动容器
docker run -d \
  --name verl-dev \
  --gpus '"device=4,5,6,7"' \
  --network host \
  --ipc=host \
  --shm-size=32g \
  -v /path/to/models:/models \
  -v /workspace:/workspace \
  -w /workspace
  verlai/verl:sgl0512.dev4 \
  sleep infinity

# 进入开发容器
docker exec -it verl-dev bash

# 安装 VERL 源码
git clone https://github.com/verl-project/verl && cd verl
# Optional: 根据实际项目回退到指定 VERL 版本
# git reset <commit_id>
pip3 install --no-deps -e .

# 在自己的工作区基于 VERL 框架编写 pipeline
# ...

# 退出开发容器
exit
```
