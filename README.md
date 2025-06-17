# GRADIO GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control

This the Gradio variation of https://github.com/nv-tlabs/GEN3C to run on https://modal.com/. Currently Running on a Single A100-80GB GPU 
with inference time for each clip around ~15 mins. Modal offers 30 dollars for free for GPU computing which is more than enough to run this
model for couple of inferences!

Make sure you have Modal module installed

```
python3 -m pip install Modal
```

and also set up correctly

```
python3 -m modal setup
```

you would also need a huggingface Token set a Secret with the name HF_TOKEN on modal dashboard then simply deploy the model

```
python3 -m modal deploy modal_cli.py
```

The first time runnning the Gradio interface, you would need to dowload the models on Modal's storage by clicking Download Checkpoints
before running inference, it's 70+ GB of data and then after you wouldn't need to download it anymore as it gets stored on the given volume

![image](https://github.com/user-attachments/assets/9f49a44d-3d6b-4c5f-aadb-3f543223c43b)
