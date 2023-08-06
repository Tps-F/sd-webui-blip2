import csv
import glob
import json
import os
from pathlib import Path

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
    else:
        return "cpu"
    # Now blip2 is not support mps. I'll make PR


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

    else:
        pass

    return model, vis_processors


def unload_model():
    print("unloading model")
    global model, vis_processors
    del model, vis_processors
    device =  get_device() 
    if device == 'mps':
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()
    model, vis_processors = "", {}
    print("Finish!")


def save_csv_f(caption, output_dir, image_filename):
    type = 'a' if os.path.exists(f'{output_dir}/blip2_caption.csv') else 'x'
    with open(f'{output_dir}/blip2_caption.csv', type, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        csvlist = [image_filename[0]]
        csvlist.extend(caption.splitlines())
        writer.writerow(csvlist)


def save_txt_f(caption, output_dir, image_filename):
    if os.path.exists(f'{output_dir}/{image_filename[0]}.txt'):
        f = open(f'{output_dir}/{image_filename[0]}.txt', 'w', encoding='utf-8')
    else:
        f = open(f'{output_dir}/{image_filename[0]}.txt', 'x', encoding='utf-8')
    f.write(f'{caption}\n')
    f.close()
        
        
def prepare(image, process_type, input_dir, output_dir, extension, save_csv, save_txt, caption_type, length_penalty, repetition_penalty, temperature):
    if process_type == "Single image":
        caption = gen_caption(image, process_type, caption_type, length_penalty, repetition_penalty, temperature)
        return caption
    else:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        # images = glob.glob(f'{input_dir}/*.png')
        extension = extension.split(", ")
        images = [i for i in Path(input_dir).glob('**/*.*') if i.suffix in extension]

        for image in images:
            image_filename = os.path.splitext(os.path.basename(image))
            image_dir = os.path.dirname(image)
            raw = Image.open(image).convert('RGB')
            caption = gen_caption(raw, process_type, caption_type, length_penalty, repetition_penalty, temperature)
            if not save_csv and not save_txt:
                save_csv_f(caption, output_dir, image_filename)
            else:
                if save_csv:
                    save_csv_f(caption, output_dir, image_filename)
                if save_txt:
                    save_txt_f(caption, image_dir, image_filename)
               
                
        return "Finish!"
    
                
def gen_caption(image, process_type, caption_type, length_penalty, repetition_penalty, temperature):

    device = get_device()
    try:
        image = vis_processors["eval"](image).unsqueeze(0).to(device)
    except KeyError:
        print("Please select models!")
    else:
        print("generating...")
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
        print(f"Finish! caption:{caption}")

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
                load_model_btn = gr.Button("Load model")
                unload_model_btn = gr.Button("Unload model")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        process_type = gr.Radio(
                            choices=["Single image", "Batch process"],
                            label="Process type",
                            value="Single image",
                            interactive=True
                        )
                        caption_type = gr.Radio(
                            choices=["Beam Search", "Nucleus Sampling"],
                            label="Caption type",
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
                with gr.Tab("single image"):
                    input_image = gr.Image(label="Image", type="pil")
                with gr.Tab("batch process"):
                    input_dir = gr.Textbox(label="Input Directory", interactive=True)
                    output_dir = gr.Textbox(label="Output Directory", interactive=True)
                    extension = gr.Textbox(label="File extensions", value=".png, .jpg", interactive=True)
                    with gr.Row():
                        save_csv = gr.Checkbox(label="Save as csv(default)", value=True, interactive=True)
                        save_txt = gr.Checkbox(label="Save as txt", interactive=True)
                    
                    gr.Markdown("#### If you do not know the path, try opening the folder in Explorer and copying the path")
                    

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

        # model_type.change(load_model, inputs=[model_type])
        load_model_btn.click(load_model, inputs=[model_type])
        unload_model_btn.click(unload_model)

        btn_caption.click(
            prepare,
            inputs=[
                input_image,
                process_type,
                input_dir,
                output_dir,
                extension,
                save_csv,
                save_txt,
                caption_type,
                length_penalty,
                repetition_penalty,
                temperature,
            ],
            outputs=[output_text],
        )

        return [(blip2, "BLIP2", "blip2")]


script_callbacks.on_ui_tabs(on_ui_tabs)
