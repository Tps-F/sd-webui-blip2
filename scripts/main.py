import json
import os

import cv2
import gradio as gr
import numpy as np
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

from modules import script_callbacks
import modules.scripts as scripts
import modules.ui


model_namelist, model_typelist, model_history = [], [], ""
model, vis_processors = "", {}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_data(file_name: str = f"{scripts.basedir()}/model.json") -> dict:
    print("Loading data...")
    data = {}

    with open(file_name, "r") as f:
        data = json.load(f)

    return data


def get_model_type() -> list:
    data = load_data()

    for model in data["model"]:
        model_typelist.append(model["type"])
    return model_typelist


def load_model(model_type):
    global model_history
    data = load_data()
    name, modeltype = (
        data["model"][model_type]["name"],
        data["model"][model_type]["type"],
    )
    if modeltype != model_history:
        global model, vis_processors

        print(f"loading model {modeltype}...It takes time.")
        model, vis_processors, _ = load_model_and_preprocess(
            name=name, model_type=modeltype, is_eval=True, device=get_device()
        )
        print(f"Finish loading model {modeltype}!")

        model_history = modeltype

    return model, vis_processors


def gen_caption(image, caption_type, length_penalty, repetition_penalty, temperature):
    print("generating...")
    device = get_device()
    image = vis_processors["eval"](image).unsqueeze(0).to(device)
    if caption_type == "Beam Search":
        caption = model.generate(
            {"image": image},
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )
    else:
        caption = model.generate(
            {"image": image},
            use_nucleus_sampling=True,
            num_captions=3,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )

    caption = "\n".join(caption)
    print(f"Finish ! caption:{caption}")

    return caption


class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "BLIP2"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return ()


def on_ui_tabs():
    get_model_type()
    with gr.Blocks(analytics_enabled=False) as blip2:
        with gr.Column():
            model_type = gr.Dropdown(
                model_typelist, label="model list", interactive=True, type="index"
            )
            with gr.Row():
                with gr.Column():
                    caption_type = gr.Radio(
                        choices=["Beam Search", "Nucleus Sampling"],
                        label="Type",
                        value="Beam Search",
                        interactive=True,
                    )
                    with gr.Tab("Beam Search"):
                        length_penalty = gr.Slider(
                            label="Length Penalty",
                            minimum=-1,
                            maximum=2,
                            value=1,
                            interactive=True,
                        )
                        repetition_penalty = gr.Slider(
                            label="Repeat Penalty",
                            minimum=1,
                            maximum=5,
                            value=1.5,
                            interactive=True,
                        )
                    with gr.Tab("Nucleus Sampling"):
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.5,
                            maximum=1,
                            value=1,
                            interactive=True,
                        )
                        repetition_penalty = gr.Slider(
                            label="Repeat Penalty",
                            minimum=1,
                            maximum=5,
                            value=1.5,
                            interactive=True,
                        )
                    btn_caption = gr.Button("Generate Caption", variant="primary")
                input_image = gr.Image(label="Image", type="pil")
            output_text = gr.Textbox(label="Answer", lines=5, interactive=False)
            with gr.Row():
                send_to_buttons = (
                    modules.generation_parameters_copypaste.create_buttons(
                        ["txt2img", "img2img", "extras"]
                    )
                )

        modules.generation_parameters_copypaste.bind_buttons(
            send_to_buttons, "", output_text
        )

        model_type.change(load_model, inputs=[model_type])

        btn_caption.click(
            gen_caption,
            inputs=[
                input_image,
                caption_type,
                length_penalty,
                repetition_penalty,
                temperature,
            ],
            outputs=[output_text],
        )

        return [(blip2, "BLIP2", "blip2")]


script_callbacks.on_ui_tabs(on_ui_tabs)
