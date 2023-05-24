import csv
import glob
import json
import os
from pathlib import Path

import gradio as gr
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

from modules import script_callbacks
import modules.scripts as scripts
import modules.ui
import modules.shared as shared


model_namelist, model_typelist, model_history = [], [], ""
model, vis_processors, txt_processors = "", {}, {}


def get_device():
    if shared.opts.data.get("blip2ForceCPU", "forceCPU") == "Yes":
        return "cpu"
    if torch.cuda.is_available():
        print("get_device CUDA ================")
        return "cuda"
    else:
        print("get_device CPU *****************")
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
    global model
    if model != "":
        unload_model()
    
    global model_history
    data = load_data()
    name, modeltype = (
        data["model"][model_type]["name"],
        data["model"][model_type]["type"],
    )

    if modeltype != model_history:
        global vis_processors, txt_processors

        print(f"loading model {modeltype}...It takes time.")

        model, vis_processors, txt_processors = load_model_and_preprocess(
            name=name, model_type=modeltype, is_eval=True, device=get_device()
        )
        print(f"Finish loading model {modeltype}!")

        model_history = modeltype

    else:
        pass

    return model_type


def unload_model():
    print("unloading model")
    global model, vis_processors, txt_processors
    del model, vis_processors, txt_processors
    device =  get_device() 
    if device == 'mps':
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()
    model, vis_processors, txt_processors = "", {}, {}
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
        
def respond(message, chat_history, input_image,input_dir,output_dir,extension,save_csv,save_txt,nucleus_sampling,length_penalty,repetition_penalty,temperature,num_beams,max_length,min_length,top_p,num_captions,num_captions_VQA,num_patches,top_k):
    prompt = chat_history_to_prompt(chat_history) + "Question: " + message + " Answer:"
    response = prepare(input_image,False,input_dir,output_dir,extension,save_csv,save_txt,nucleus_sampling,length_penalty,repetition_penalty,temperature,num_beams,max_length,min_length,top_p,num_captions,prompt,False,num_captions_VQA,num_patches,top_k)
    chat_history.append((message, response))
    return "", chat_history

def prepare(image, batch, input_dir, output_dir, extension, save_csv, save_txt, nucleus_sampling, length_penalty, repetition_penalty, temperature, num_beams, max_length, min_length, top_p, num_captions, prompt, prepend_prompt,num_captions_VQA,num_patches,top_k):
    if not batch:
        caption = gen_caption(image, nucleus_sampling, length_penalty, repetition_penalty, temperature, num_beams, max_length, min_length, top_p, num_captions, prompt,num_captions_VQA,num_patches,top_k)
        if prompt is not None and prepend_prompt :
            caption = prompt + " " + caption
        return caption
    else:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        extension = extension.split(", ")
        images = [i for i in Path(input_dir).glob('**/*.*') if i.suffix in extension]

        for image in images:
            image_filename = os.path.splitext(os.path.basename(image))
            raw = Image.open(image).convert('RGB')
            caption = gen_caption(raw, nucleus_sampling, length_penalty, repetition_penalty, temperature, num_beams, max_length, min_length, top_p, num_captions, prompt,num_captions_VQA,num_patches,top_k)
            if prompt is not None and prepend_prompt:
                caption = prompt + " " + caption
            if not save_csv and not save_txt:
                save_csv_f(caption, output_dir, image_filename)
            else:
                if save_csv:
                    save_csv_f(caption, output_dir, image_filename)
                if save_txt:
                    save_txt_f(caption, output_dir, image_filename)

        return "Finish!"
    
                
def gen_caption(image, nucleus_sampling, length_penalty, repetition_penalty, temperature, num_beams, max_length, min_length, top_p, num_captions, prompt, num_captions_VQA, num_patches, top_k):

    device = get_device()
    try:
        image = vis_processors["eval"](image).unsqueeze(0).to(device)
    except KeyError:
        print("Please select models!")
    else:
        print("generating...", prompt)
        imgPrompt = {"image": image}
        if prompt != "":
            print("prompt", prompt)
            imgPrompt["prompt"] = prompt

        if hasattr(model, 'generate'):
            caption = model.generate(
                imgPrompt,
                use_nucleus_sampling=nucleus_sampling,
                num_beams=int(num_beams),
                max_length=max_length,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_captions=int(1 if nucleus_sampling else num_captions if (num_captions <= num_beams and num_beams > 1) else num_beams - 1),
                temperature=temperature,
            )
        elif hasattr(model, 'predict_answers'):
            print("predict_answers")
            question = txt_processors["eval"](prompt)
            caption = model.predict_answers(
                samples={"image": image, "text_input": question}, 
                inference_method="generate", 
                num_captions=int(num_captions_VQA), 
                num_patches=int(num_patches), 
                cap_max_length=int(max_length), 
                cap_min_length=int(min_length),
                top_k=top_k
            ) 
        else:
            print("ERR(blip2-gen_caption): Model has neither a generate nor a predict_answers method.")

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
        with gr.Row():
            model_type = gr.Dropdown(
                model_typelist, label="model list", interactive=True, type="index"
            )
            with gr.Column():
                load_model_btn = gr.Button("Load model")
                unload_model_btn = gr.Button("Unload model")
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    nucleus_sampling = gr.Checkbox(
                        label="Nucleus Sampling",
                        value=False,
                        interactive=True,
                    )
                    num_captions = gr.Number(
                        label="Number of Captions",
                        minimum=1,
                        step=1,
                        value=1,
                        interactive=True,
                    )
                    repetition_penalty = gr.Slider(
                        label="Repeat Penalty",
                        minimum=0.0,
                        maximum=4.99,
                        value=1,
                        interactive=True,
                    )
                    max_length = gr.Slider(
                        label="Max Length",
                        minimum=10,
                        maximum=50,
                        step=1,
                        value=30,
                        interactive=True,
                    )
                    min_length = gr.Slider(
                        label="Min Length",
                        minimum=1,
                        maximum=40,
                        step=1,
                        value=5,
                        interactive=True,
                    )                    
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.01,
                        maximum=1.0,
                        value=1.0,
                        interactive=True,
                    )
                    gr.Markdown("##### Beam")
                    num_beams = gr.Slider(
                        label="Number of Beams",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=1,
                        interactive=True,
                    )
                    length_penalty = gr.Slider(
                        label="Length Penalty",
                        minimum=-1.0,
                        maximum=2.0,
                        value=1.0,
                        interactive=True,
                    )
                    gr.Markdown("##### Nucleus & VQA")
                    top_p = gr.Slider(
                        label="Top P",
                        minimum=0.01,
                        maximum=1.0,
                        value=0.9,
                        interactive=True,
                    )
                    gr.Markdown(
                    """
                    #### VQA specific model settings
                    """)
                    num_captions_VQA = gr.Number(
                        label="Number of Captions VQA",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=50,
                        interactive=True,
                        precision=0,
                    )
                    num_patches = gr.Number(
                        label="Number of Patches",
                        minimum=1,
                        step=1,
                        value=50,
                        interactive=True,
                        precision=0,
                    )
                    top_k = gr.Number(
                        label="Top K",
                        minimum=0.0,
                        value=50,
                        interactive=True,
                    )
                    with gr.Row():
                        gr.Markdown(
                            """ - use_nucleus_sampling:     Whether to use nucleus sampling. If False, use top-k sampling.
                                - num_captions:             Number of captions to be generated for each image.
                                    - if using nucleus sampling it will be 1
                                    - has to be smaller than `num_beams` if `num_beams` > 1

                                - repetition_penalty:       The parameter for repetition penalty. 1.0 means no penalty.
                                    - https://arxiv.org/pdf/1909.05858.pdf
                                - max_length:               The maximum length of the sequence to be generated.
                                - min_length:               The minimum length of the sequence to be generated.
                                - temperature:              The value used to module the next token probabilities.
                                ##### Nucleus
                                - top_p:                    The cumulative probability for nucleus sampling.
                                ##### Beam
                                - num_beams:                Number of beams for beam search. 1 means no beam search.
                                - length_penalty:           penalty to the length that is used with beam generation. 
                                    - `length_penalty` > 0.0 promotes longer sequences,
                                    - `length_penalty` < 0.0 encourages shorter sequences.

                                #### VQA
                                - top_k (float): The number of the highest probability tokens for top-k sampling.
                                - num_captions_VQA (int): Number of captions generated for each image.
                                - num_patches (int): Number of patches sampled for each image.
                            """
                        )
                with gr.Column():
                    with gr.Row():
                        with gr.Tab("single image"):
                            input_image = gr.Image(label="Image", type="pil")
                            
                            gr.Markdown("### Q&A")
                            chatbot = gr.Chatbot(label="Q&A")
                            msg = gr.Textbox(label="ChatBox (hit enter to submit)")
                            clear = gr.Button("Clear")

                            with gr.Row():
                                gr.Markdown("""### Caption""")
                                prepend_single = gr.Checkbox(label="Prepend prompt to answer", interactive=True)
                            input_text_single = gr.Textbox(label="Prompt (optional)", lines=2, interactive=True)
                            output_text_single = gr.Textbox(label="Answer", lines=5, interactive=False)
                            btn_caption_single = gr.Button("Generate Caption", variant="primary")

                        with gr.Tab("batch process"):
                            input_dir = gr.Textbox(label="Input Directory", interactive=True)
                            output_dir = gr.Textbox(label="Output Directory", interactive=True)
                            extension = gr.Textbox(label="File extensions", value=".png, .jpg", interactive=True)
                            with gr.Row():
                                save_csv = gr.Checkbox(label="Save as csv(default)", value=True, interactive=True)
                                save_txt = gr.Checkbox(label="Save as txt", interactive=True)
                                prepend_batch = gr.Checkbox(label="Prepend prompt to results", interactive=True)

                            input_text_batch = gr.Textbox(label="Prompt (optional)", lines=2, interactive=True)
                            btn_caption_batch = gr.Button("Generate Captions", variant="primary")
                            output_text_batch = gr.Textbox(label="Response", lines=1, interactive=False)
                
                                
            with gr.Row():
                send_to_buttons = (
                    modules.generation_parameters_copypaste.create_buttons(
                        ["txt2img", "img2img", "extras"]
                    )
                )

            # utterly stupid, insufficient, not right, and unreasonable
            grfalse = gr.Checkbox(value=False, visible=False)
            grtrue = gr.Checkbox(value=True, visible=False)

        modules.generation_parameters_copypaste.bind_buttons(
            send_to_buttons, "", output_text_single
        )

        load_model_btn.click(load_model, inputs=[model_type], outputs=[model_type])
        unload_model_btn.click(unload_model)

        load_model_btn.style(size="sm")
        unload_model_btn.style(size="sm")

        msg.submit(respond, [msg, chatbot, input_image,input_dir,output_dir,extension,save_csv,save_txt,nucleus_sampling,length_penalty,repetition_penalty,temperature,num_beams,max_length,min_length,top_p,num_captions,num_captions_VQA,num_patches,top_k,],
                    [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

        btn_caption_single.click(
            prepare,
            inputs=[
                input_image,
                grfalse,
                input_dir,
                output_dir,
                extension,
                save_csv,
                save_txt,
                nucleus_sampling,
                length_penalty,
                repetition_penalty,
                temperature,
                num_beams,
                max_length,
                min_length,
                top_p,
                num_captions,
                input_text_single,
                prepend_single,
                num_captions_VQA,
                num_patches,
                top_k,
            ],
            outputs=[output_text_single],
        )
        btn_caption_batch.click(
            prepare,
            inputs=[
                input_image,
                grtrue,
                input_dir,
                output_dir,
                extension,
                save_csv,
                save_txt,
                nucleus_sampling,
                length_penalty,
                repetition_penalty,
                temperature,
                num_beams,
                max_length,
                min_length,
                top_p,
                num_captions,
                input_text_batch,
                prepend_batch,
                num_captions_VQA,
                num_patches,
                top_k,
            ],
            outputs=[output_text_batch],
        )
        
        return [(blip2, "BLIP2", "blip2")]



def on_ui_settings():
    section = ("blip2", "BLIP2")

    shared.opts.add_option(
        "blip2ForceCPU",
        shared.OptionInfo(
            "No",
            "Force blip models to use CPU",
            gr.Dropdown,
            lambda: {"choices": ["Yes", "No"]},
            section=section,
        ),
    )

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)

def chat_history_to_prompt(chat_history):
    return "".join(list(map(lambda qa : "Question: {} Answer: {}. ".format(qa[0], qa[1]), chat_history)))
