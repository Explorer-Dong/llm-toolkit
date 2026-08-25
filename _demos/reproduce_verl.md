# VERL reproduction manual

We use [VERL](https://github.com/verl-project/verl) to **easily** train models with reinforcement learning algorithms.

## Preparation

```bash
# pull VERL's basic developing environment
docker pull verlai/verl:sgl0512.dev4

# bind your workspace and start docker container
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

# enter the container
docker exec -it verl-dev bash

# start your work
# 1. setup VERL package
git clone https://github.com/verl-project/verl && cd verl
# (Optional) reset to the target VERL version depend on your project
# git reset <commit_id>
# 2. install VERL
pip3 install --no-deps -e .
# 3. build your algorithm framework based on VERL
# from verl import ...

# exit the dev container
exit
```
