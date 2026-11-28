since: rhel8-b5404

**Images: `deepseek.sh` can load the image encoder of the model.**

- DeepSeek-V4.1-Flash reads images, with an encoder that is a file of its own in the repository of the model: `mmproj-F16.gguf` (926 MiB). The script never gave it to the server, which refused every image (`image input is not supported`).
- With `MMPROJ=mmproj-F16.gguf` in `~/.deepseek.conf`, and that file in the folder of the model files (as it is, no repacking), the script copies it to the RAM disk with the weights and starts the server with `--mmproj`. Without the setting nothing changes.
- Nothing changes in the binaries: the encoder, the routing of image tokens and the draft with images are code of upstream ik_llama.cpp.
- Not tried on the 2-socket server: that an image is read right, how long it takes, the draft and the saved conversations with images.
- A conversation saved in `CACHE_DIR` without the encoder is not taken back once it is loaded, nor the other way round: those files stay in the folder and take room there until they are removed.
