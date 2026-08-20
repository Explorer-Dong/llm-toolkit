# Local images

Download or build docker images locally, for environment migration or reproduction.

non-compression:

```bash
# save image
docker save verlai/verl:sgl0512.dev4 -o ./_images/verl-sgl0512.dev4.tar

# load image
docker load -i ./_images/verl-sgl0512.dev4.tar
```

compression:

```bash
# save image
docker save verlai/verl:sgl0512.dev4 | zstd -T0 -o ./_images/verl-sgl0512.dev4.tar.zst

# load image
zstd -dc ./_images/verl-sgl0512.dev4.tar.zst | docker load
```
