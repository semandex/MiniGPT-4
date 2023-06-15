"""
Use MiniGPT-4 to generate captions for a provided list of URLs.
"""


import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--urls-path", type=str, default=None, help="specify the path to the URLs to generate captions for")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your image first',
                                                                    interactive=False), gr.update(
        value="Upload & Start Chat", interactive=True), chat_state, img_list


def upload_img(gr_img):
    # if gr_img is None:
    #     return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return chat_state, img_list


def gradio_ask(user_message, chatbot, chat_state):
    # if len(user_message) == 0:
    #     return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    # chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    return llm_message
    # chatbot[-1][1] = llm_message
    # return chatbot, chat_state, img_list


########################
num_beams = 1
temperature = 1
# chat_state = gr.State()
# img_list = gr.State()
chatbot = gr.Chatbot(label='MiniGPT-4')
# text_input = gr.Textbox(label='User', placeholder='Please upload your image first', interactive=False, value='Caption this image')

PROMPT = 'Is this a blank image? Describe it succinctly.'


def get_caption(image):
    chat_state, img_list = upload_img(image)
    # gradio_ask('Describe this image in detail.', chatbot, chat_state)
    gradio_ask(PROMPT, chatbot, chat_state)
    llm_message = gradio_answer(chatbot, chat_state, img_list, num_beams, temperature)
    chat_state.messages = []
    return llm_message


# chatbot, image, text_input, upload_button, chat_state, img_list = gradio_reset(chat_state, img_list)
# print(llm_message)
import csv
import json
import os
from tqdm import tqdm
import multiprocessing

os.makedirs('results', exist_ok=True)


def load_urls(path):
    with open(path, 'r') as f:
        if path.endswith('.txt') or path.endswith('.csv'):
            return [line.strip() for line in f.readlines()]
        elif path.endswith('.json'):
            results = json.load(f)['results']
            return [result['image_url'] for result in results]
        else:
            raise Exception('URL filetype not supported')


lock = multiprocessing.Lock()


def generate_captions(urls_path, pbar):
    urls = load_urls(urls_path)
    if len(urls) == 0:
        return

    out_name = os.path.basename(urls_path)
    if os.path.exists(os.path.join('results', f'{out_name}.csv')):
        with open(os.path.join('results', f'{out_name}.csv'), 'r') as f:
            reader = csv.reader(f)
            next(reader)
            results = dict()
            for row in reader:
                results[row[1]] = row[2]
    else:
        results = dict()

    for url in urls:
        if 'https://' in url:
            # if not url.startswith('http'):
            url = url.split(',', 1)[1]

        if url in results:
            with lock:
                pbar.update()
            continue

        try:
            caption = get_caption(url)
            results[url] = caption
        except Exception as e:
            print(e)
            print(f'Failed for {url}')

        if len(results) % 20 == 0:
            with open(os.path.join('results', f'{out_name}.csv'), 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'URL', 'caption'])
                for i, (key, value) in enumerate(results.items()):
                    writer.writerow([i, key, value])

        with lock:
            pbar.update()

    with open(os.path.join('results', f'{out_name}.csv'), 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'URL', 'caption'])
        for i, (key, value) in enumerate(results.items()):
            writer.writerow([i, key, value])
    print(f'Done with {urls_filename}')


if __name__ == '__main__':
    urls_path = args.urls_path

    # generate captions for a specific set of urls or all URL files in 'url_files'
    if urls_path is None:
        total = 0
        for urls_filename in os.listdir('url_files'):
            urls = load_urls(os.path.join('url_files', urls_filename))
            total += len(urls)
        pbar = tqdm(desc='Generating captions', total=total)
    else:
        urls = load_urls(os.path.join('url_files', urls_path))
        total = len(urls)
        pbar = tqdm(desc=f'Generating captions for {urls_path}', total=total)

    if urls_path is None:
        # threads = list()  # TODO test multiprocessing

        for urls_filename in os.listdir('url_files'):
            urls_path = os.path.join('url_files', urls_filename)
            generate_captions(urls_path, pbar)

        #     thread = multiprocessing.Process(target=generate_captions, args=(urls_filename, pbar))
        #     threads.append(thread)
        #     thread.start()
        #
        # for thread in threads:
        #     thread.join()

    else:
        generate_captions(urls_path, pbar)

    pbar.close()
