This the Gradio variation of https://github.com/nv-tlabs/GEN3C to run on modal.com.
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
