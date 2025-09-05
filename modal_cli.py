from modal import Image, method
import modal
import uuid
import os
from gradio.routes import mount_gradio_app
import gradio as gr
from fastapi import FastAPI
import shutil

base_image =  Image.from_registry("nvcr.io/nvidia/pytorch:24.10-py3")
volume_1 = modal.Volume.from_name("gen3c-data", create_if_missing=True)
volume_2 = modal.Volume.from_name("gen3c-results", create_if_missing=True)

def login_huggingface():
    from huggingface_hub import login

    login(os.environ["HF_TOKEN"])

def download_moge_model():
    from huggingface_hub import snapshot_download

    snapshot_download('Ruicheng/moge-vitl')

def download_checkpoints(checkpoint_output_directory):

    # Ensure the directory exists
    os.makedirs(checkpoint_output_directory, exist_ok=True)

    # If there's anything in there already, skip downloading
    if any(os.scandir(checkpoint_output_directory)):
        print(f"Checkpoints found in {checkpoint_output_directory}, skipping download.")
        return
    
    from scripts.download_gen3c_checkpoints import main as dw_chkpoints

    class _Args():
        pass
    
    args = _Args()
    args.checkpoint_dir = checkpoint_output_directory
    
    dw_chkpoints(args)
    download_moge_model()


full_image = (base_image.run_commands("pip uninstall -y grpclib grpcio").pip_install("grpclib==0.4.7").
            pip_install(
                "huggingface_hub", 
                "diffusers", 
                "transformers", 
                "megatron-core",
                "attrs==25.1.0", 
                "better-profanity==0.7.0", 
                "boto3==1.35.99", 
                "decord==0.6.0", 
                "diffusers==0.32.2", 
                "einops==0.8.1", 
                "huggingface-hub==0.29.2", 
                "hydra-core==1.3.2", 
                "imageio[pyav,ffmpeg]==2.37.0", 
                "iopath==0.1.10", "ipdb==0.13.13", 
                "loguru==0.7.2", "mediapy==1.2.2", 
                "megatron-core==0.10.0", "nltk==3.9.1",
                "numpy==1.26.4", 
                "nvidia-ml-py==12.535.133", 
                "omegaconf==2.3.0", 
                "opencv-python==4.8.0.74", 
                "pandas==2.2.3", "peft==0.14.0", 
                "pillow==11.1.0", 
                "protobuf==4.25.3", 
                "pynvml==12.0.0", 
                "pyyaml==6.0.2", 
                "retinaface-py==0.0.2", 
                "safetensors==0.5.3", 
                "scikit-image==0.25.2", 
                "sentencepiece==0.2.0", 
                "setuptools==76.0.0", 
                "termcolor==2.5.0", 
                "tqdm==4.66.5", 
                "transformers==4.49.0", 
                "warp-lang==1.7.2")
            .run_commands("pip install git+https://github.com/microsoft/MoGe.git")
            .run_commands("pip install git+https://github.com/OutofAi/GEN3C.git")
            )

app = modal.App(
    "GEN3C")

NUM_GPUS = 1

@app.cls(
    image=full_image,
    gpu=f"A100-80GB:{NUM_GPUS}",
    timeout=3000,
    cpu=2 * NUM_GPUS,
    volumes={"/root/data": volume_1, "/root/output": volume_2},
    # volumes={"/root/data": volume_1},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
class COSMOS:
    checkpoint_dir: str = modal.parameter()
    pipeline = None
    moge_model = None
    device = None

    @modal.enter()
    def setup(self):
        from cosmos_predict1.diffusion.inference.gen3c_single_image import load_models

        if self.pipeline == None:
            print("loading Models")
            self.pipeline, self.moge_model, self.device = load_models(self.checkpoint_dir, 1.0)

    @method()
    def run(self, input_image, trajectory, movement_distance, camera_rotation, guidance, session_id):

        file_id = uuid.uuid4().hex

        session_dir = f"/root/output/{session_id}"

        os.makedirs(session_dir, exist_ok=True)

        input_path = os.path.join(session_dir, f"{file_id}.png")
        output_path = os.path.join(session_dir, f"{file_id}.mp4")
        input_image.save(input_path)

        volume_2.commit()

        from cosmos_predict1.diffusion.inference.gen3c_single_image import run_full_demo

        result = run_full_demo(self.pipeline, self.moge_model, self.device, input_path, output_path, trajectory, movement_distance, camera_rotation, guidance)

        volume_2.commit()

        return result
    
cosmos = COSMOS(
    checkpoint_dir="/root/data/checkpoints",
)

@app.function(
    image=full_image,
    timeout=3000,
    cpu=2 * NUM_GPUS,
    volumes={"/root/data": volume_1, "/root/output": volume_2},
    # volumes={"/root/data": volume_1},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
@modal.concurrent(max_inputs=1000)  # Gradio can handle many async inputs
@modal.asgi_app()
def ui():

    def start_session(request: gr.Request):
        return request.session_hash
    
    def cleanup(request: gr.Request):
        sid = request.session_hash
        if sid:
            d1 = os.path.join(cosmos.checkpoint_dir, sid)
            shutil.rmtree(d1, ignore_errors=True)

    def run_and_fetch(inp, traj, dist, rot, guid, session_id):

        mp4_path = cosmos.run.remote(inp, traj, dist, rot, guid, session_id)
        volume_2.reload()
        return mp4_path
    
    def download_data():
        login_huggingface()
        download_checkpoints(cosmos.checkpoint_dir)
        return gr.update(interactive=False), gr.update(interactive=True)

    css = """
    #col-container {
        margin: 0 auto;
        max-width: 1024px;
    }
    """

    with gr.Blocks(css=css) as demo:

        session_state = gr.State()
        demo.load(start_session, outputs=[session_state])

        # ensure the dir exists in *this* container before scanning it
        os.makedirs(cosmos.checkpoint_dir, exist_ok=True)
        try:
            has_checkpoints = any(os.scandir(cosmos.checkpoint_dir))
        except FileNotFoundError:
            has_checkpoints = False
        
        gr.HTML(
            """
            <div style="text-align: center;">
                <p style="font-size:16px; display: inline; margin: 0;">
                    <strong>GEN3C:</strong> 3D-Informed World-Consistent Video Generation with Precise Camera Control
                </p>
                <a href="https://github.com/nv-tlabs/GEN3C" style="display: inline-block; vertical-align: middle; margin-left: 0.5em;">
                    <img src="https://img.shields.io/badge/GitHub-Repo-blue" alt="GitHub Repo">
                </a>
            </div>
            """
        )
        with gr.Column(elem_id="col-container"):
            with gr.Row():
                with gr.Column():
                    inp_image = gr.Image(label="Input Image(Preferred Resolution 1280x720)", type="pil") 

                    trajectory = gr.Dropdown(
                        choices=["left", "right", "up", "down", "zoom_in", "zoom_out", "clockwise", "counterclockwise", "none"],
                        value="left",
                        label="Trajectory"
                    )
                    download_btn = gr.Button("Download Checkpoints", interactive=not has_checkpoints)
                    run_btn = gr.Button("Move Camera", variant="primary", interactive=has_checkpoints)

                    move_dist = gr.Slider(0.0, 1.0, value=0.3, label="Movement Distance")
                    cam_rot = gr.Radio(
                        choices=["center_facing", "no_rotation", "trajectory_aligned"],
                        value="center_facing",
                        label="Camera Rotation"
                    )
                    guidance = gr.Slider(0.0, 10.0, value=1.0, label="Guidance Scale")

                with gr.Column():
                        output_video = gr.Video(label="Generated Video")

        
        download_btn.click(
            fn=download_data,
            inputs=[],
            outputs=[download_btn, run_btn]          
        )
        run_btn.click(
            fn=run_and_fetch,
            inputs=[inp_image, trajectory, move_dist, cam_rot, guidance, session_state],
            outputs=output_video
        )

        demo.unload(cleanup)
    fastapi_app = FastAPI(title="GEN3C")
    return mount_gradio_app(fastapi_app, blocks=demo, path="/")

