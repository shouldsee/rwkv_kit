import gradio as gr
import os, gc, copy, torch, re
from datetime import datetime
from huggingface_hub import hf_hub_download
from pynvml import *
nvmlInit()
gpu_h = nvmlDeviceGetHandleByIndex(0)
ctx_limit = 1024
gen_limit = 500
gen_limit_long = 800

# title = "RWKV-x060-World-7B-v2.1-20240507-ctx4096"

### [shouldsee]: smaller model for testing
title = "RWKV-x060-World-1B6-v2.1-20240328-ctx4096"

os.environ["RWKV_JIT_ON"] = '1'

# from rwkv.model import RWKV
from rwkv_model import RWKV

model_path = hf_hub_download(repo_id="BlinkDL/rwkv-6-world", filename=f"{title}.pth")


# os.environ["RWKV_CUDA_ON"] = '1' # if '1' then use CUDA kernel for seq mode (much faster)
# model = RWKV(model=model_path, strategy='cuda fp16i8 *8 -> cuda fp16')

os.environ["RWKV_CUDA_ON"] = '0' # if '1' then use CUDA kernel for seq mode (much faster)
model = RWKV(model=model_path, strategy='cpu fp32')

# model_path = '/mnt/e/RWKV-Runner/models/rwkv-final-v6-2.1-7b' # conda activate torch2; cd /mnt/program/_RWKV_/_ref_/_gradio_/RWKV-Gradio-2; python app_tab.py
# model = RWKV(model=model_path, strategy='cuda fp16i8 *8 -> cuda fp16')

from rwkv.utils import PIPELINE, PIPELINE_ARGS
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

args = model.args
eng_name = 'rwkv-x060-eng_single_round_qa-1B6-20240516-ctx2048'
# eng_name = 'rwkv-x060-eng_single_round_qa-7B-20240516-ctx2048'
eng_file = hf_hub_download(repo_id="BlinkDL/temp-latest-training-models", filename=f"{eng_name}.pth")
state_eng_raw = torch.load(eng_file)
state_eng = [None] * args.n_layer * 3

chn_name = 'rwkv-x060-chn_single_round_qa-1B6-20240516-ctx2048'
# chn_name = 'rwkv-x060-chn_single_round_qa-7B-20240516-ctx2048'
chn_file = hf_hub_download(repo_id="BlinkDL/temp-latest-training-models", filename=f"{chn_name}.pth")
state_chn_raw = torch.load(chn_file)
state_chn = [None] * args.n_layer * 3


### not available
wyw_name = 'rwkv-x060-eng_single_round_qa-1B6-20240516-ctx2048'
# wyw_name = 'rwkv-x060-chn_文言文和古典名著_single_round_qa-7B-20240601-ctx2048'
wyw_file = hf_hub_download(repo_id="BlinkDL/temp-latest-training-models", filename=f"{wyw_name}.pth")
state_wyw_raw = torch.load(wyw_file)
state_wyw = [None] * args.n_layer * 3

for i in range(args.n_layer):
    dd = model.strategy[i]
    dev = dd.device
    atype = dd.atype    
    state_eng[i*3+0] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
    state_eng[i*3+1] = state_eng_raw[f'blocks.{i}.att.time_state'].transpose(1,2).to(dtype=torch.float, device=dev).requires_grad_(False).contiguous()
    state_eng[i*3+2] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()

    state_chn[i*3+0] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
    state_chn[i*3+1] = state_chn_raw[f'blocks.{i}.att.time_state'].transpose(1,2).to(dtype=torch.float, device=dev).requires_grad_(False).contiguous()
    state_chn[i*3+2] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()

    state_wyw[i*3+0] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
    state_wyw[i*3+1] = state_wyw_raw[f'blocks.{i}.att.time_state'].transpose(1,2).to(dtype=torch.float, device=dev).requires_grad_(False).contiguous()
    state_wyw[i*3+2] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()

def generate_prompt(instruction, input=""):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    input = input.strip().replace('\r\n','\n').replace('\n\n','\n')
    if input:
        return f"""Instruction: {instruction}\n\nInput: {input}\n\nResponse:"""
    else:
        return f"""User: hi\n\nAssistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.\n\nUser: {instruction}\n\nAssistant:"""

def qa_prompt(instruction):
    instruction = instruction.strip().replace('\r\n','\n')
    instruction = re.sub(r'\n+', '\n', instruction)
    return f"User: {instruction}\n\nAssistant:"""

penalty_decay = 0.996

def infer_ctx(
    ctx,
    state=None,
    token_count=gen_limit,
    temperature=1.0,
    top_p=0.3,
    presencePenalty = 0.3,
    countPenalty = 0.3,
):
    state = copy.deepcopy(state)
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0]) # stop generation whenever you see any token here
    ctx = ctx.strip()
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    for i in range(int(token_count)):
        out, state = model.forward(pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token], state)
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        all_tokens += [token]
        for xxx in occurrence:
            occurrence[xxx] *= penalty_decay
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        
        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            out_str += tmp
            yield out_str.strip()
            out_last = i + 1

    gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'{timestamp} - vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')  
    del out
    del state
    gc.collect()
    torch.cuda.empty_cache()
    yield out_str.strip()


import plotly.graph_objects as go

def plot_target_logits(
    ctx,
    token_count=gen_limit,
    temperature=1.0,
    top_p=0.3,
    presencePenalty = 0.3,
    countPenalty = 0.3,
    state=None,
):  
    '''
    输出不同位置的logits
    '''

    state = copy.deepcopy(state)
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0]) # stop generation whenever you see any token here
    ctx = ctx.strip()
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    if 1:
        enc_ctx = pipeline.encode(ctx)[-ctx_limit:]
        out, state, outputs = model.forward_with_internal( enc_ctx, state)
        logits = model.to_logits(outputs)

        targ_tok = enc_ctx[-1]
        xtext = dec_ctx = [ pipeline.decode([xx]) for xx in enc_ctx]
        print(ctx)
        print(dec_ctx)
        fig = go.Figure(data=go.Heatmap(
                            z = logits[:,:,targ_tok].cpu().numpy(),
                            # z = logits[:,:,targ_tok].cpu().numpy().T,
                            # x = dec_ctx,
                            # x = [ pipeline.decode([xx]) for xx in enc_ctx],


                            zmax = 0,
                            zmin = -30,
                            colorbar=dict(title='logits'),
                            colorscale='greens',
                            ),
                            layout=go.Layout(
                                title=f"Target Token Logit Plot: 目标token {[pipeline.decode([targ_tok])]}",
                                xaxis=dict(
                                    title="Input Token",
                                    tickmode='array',
                                    ticktext=xtext,
                                    tickvals=list(range(len(xtext))),
                                
                                ),
                                yaxis=dict(title="Hidden Layer Index"),
                            )
                        )


        return ['dbg', fig]



def evaluate(
    ctx,
    token_count=gen_limit,
    temperature=1.0,
    top_p=0.3,
    presencePenalty=0.3,
    countPenalty=0.3,
):
    yield from infer_ctx(ctx, state=None, 
        token_count=token_count, 
        temperature=temperature,
        top_p = top_p,
        presencePenalty=presencePenalty,
        countPenalty=countPenalty)


def evaluate_eng(
    ctx,
    token_count=gen_limit,
    temperature=1.0,
    top_p=0.3,
    presencePenalty=0.3,
    countPenalty=0.3,
):
    yield from infer_ctx(ctx, state_eng, 
        token_count=token_count, 
        temperature=temperature,
        top_p = top_p,
        presencePenalty=presencePenalty,
        countPenalty=countPenalty)


def evaluate_chn(
    ctx,
    token_count=gen_limit,
    temperature=1.0,
    top_p=0.3,
    presencePenalty=0.3,
    countPenalty=0.3,
):
    yield from infer_ctx(ctx, state_chn, 
        token_count=token_count, 
        temperature=temperature,
        top_p = top_p,
        presencePenalty=presencePenalty,
        countPenalty=countPenalty)


def evaluate_wyw(
    ctx,
    token_count=gen_limit,
    temperature=1.0,
    top_p=0.3,
    presencePenalty=0.3,
    countPenalty=0.3,
):
    yield from infer_ctx(ctx, state_wyw, 
        token_count=token_count, 
        temperature=temperature,
        top_p = top_p,
        presencePenalty=presencePenalty,
        countPenalty=countPenalty)        

examples = [
    ["Assistant: How can we craft an engaging story featuring vampires on Mars? Let's think step by step and provide an expert response.", gen_limit, 1, 0.3, 0.5, 0.5],
    ["Assistant: How can we persuade Elon Musk to follow you on Twitter? Let's think step by step and provide an expert response.", gen_limit, 1, 0.3, 0.5, 0.5],
    [generate_prompt("東京で訪れるべき素晴らしい場所とその紹介をいくつか挙げてください。"), gen_limit, 1, 0.3, 0.5, 0.5],
    [generate_prompt("Write a story using the following information.", "A man named Alex chops a tree down."), gen_limit, 1, 0.3, 0.5, 0.5],
    ["A few light taps upon the pane made her turn to the window. It had begun to snow again.", gen_limit, 1, 0.3, 0.5, 0.5],
    ['''Edward: I am Edward Elric from Fullmetal Alchemist.\n\nUser: Hello Edward. What have you been up to recently?\n\nEdward:''', gen_limit, 1, 0.3, 0.5, 0.5],
    [generate_prompt("Write a simple website in HTML. When a user clicks the button, it shows a random joke from a list of 4 jokes."), 500, 1, 0.3, 0.5, 0.5],
    ['''Japanese: 春の初め、桜の花が満開になる頃、小さな町の片隅にある古びた神社の境内は、特別な雰囲気に包まれていた。\n\nEnglish:''', gen_limit, 1, 0.3, 0.5, 0.5],
    ["En una pequeña aldea escondida entre las montañas de Andalucía, donde las calles aún conservaban el eco de antiguas leyendas, vivía un joven llamado Alejandro.", gen_limit, 1, 0.3, 0.5, 0.5],
    ["Dans le cœur battant de Paris, sous le ciel teinté d'un crépuscule d'or et de pourpre, se tenait une petite librairie oubliée par le temps.", gen_limit, 1, 0.3, 0.5, 0.5],
    ["في تطور مذهل وغير مسبوق، أعلنت السلطات المحلية في العاصمة عن اكتشاف أثري قد يغير مجرى التاريخ كما نعرفه.", gen_limit, 1, 0.3, 0.5, 0.5],
    ['''“当然可以，大宇宙不会因为这五公斤就不坍缩了。”关一帆说，他还有一个没说出来的想法：也许大宇宙真的会因为相差一个原子的质量而由封闭转为开放。大自然的精巧有时超出想象，比如生命的诞生，就需要各项宇宙参数在几亿亿分之一精度上的精确配合。但程心仍然可以留下她的生态球，因为在那无数文明创造的无数小宇宙中，肯定有相当一部分不响应回归运动的号召，所以，大宇宙最终被夺走的质量至少有几亿吨，甚至可能是几亿亿亿吨。\n但愿大宇宙能够忽略这个误差。\n程心和关一帆进入了飞船，智子最后也进来了。她早就不再穿那身华丽的和服了，她现在身着迷彩服，再次成为一名轻捷精悍的战士，她的身上佩带着许多武器和生存装备，最引人注目的是那把插在背后的武士刀。\n“放心，我在，你们就在！”智子对两位人类朋友说。\n聚变发动机启动了，推进器发出幽幽的蓝光，''', gen_limit, 1, 0.3, 0.5, 0.5],
]

examples_eng = [
    ["How can I craft an engaging story featuring vampires on Mars?", gen_limit_long, 1, 0.2, 0.3, 0.3],
    ["Compare the business models of Apple and Google.", gen_limit_long, 1, 0.2, 0.3, 0.3],
    ["In JSON format, list the top 5 tourist attractions in Paris.", gen_limit_long, 1, 0.2, 0.3, 0.3],
    ["Write an outline for a fantasy novel where dreams can alter reality.", gen_limit_long, 1, 0.2, 0.3, 0.3],
    ["Can fish get thirsty?", gen_limit_long, 1, 0.2, 0.3, 0.3],
    ["Write a Bash script to check disk usage and send alerts if it's too high.", gen_limit_long, 1, 0.2, 0.3, 0.3],
    ["Write a simple website in HTML. When a user clicks the button, it shows a random joke from a list of 4 jokes.", gen_limit_long, 1, 0.2, 0.3, 0.3],
]

examples_chn = [
    ["怎样写一个在火星上的吸血鬼的有趣故事？", gen_limit_long, 1, 0.2, 0.3, 0.3],
    ["比较苹果和谷歌的商业模式。", gen_limit_long, 1, 0.2, 0.3, 0.3],
    ["鱼会口渴吗？", gen_limit_long, 1, 0.2, 0.3, 0.3],
    ["以 JSON 格式解释冰箱是如何工作的。", gen_limit_long, 1, 0.2, 0.3, 0.3],
    ["编写一个Bash脚本来检查磁盘使用情况，如果使用量过高则发送警报。", gen_limit_long, 1, 0.2, 0.3, 0.3],
    ["用HTML编写一个简单的网站。当用户点击按钮时，从4个笑话的列表中随机显示一个笑话。", gen_limit_long, 1, 0.2, 0.3, 0.3],
]

examples_wyw = [
    ["我和前男友分手了", gen_limit_long, 1, 0.2, 0.3, 0.3],
    ["量子计算机的原理", gen_limit_long, 1, 0.2, 0.3, 0.3],
    ["李白和杜甫的结拜故事", gen_limit_long, 1, 0.2, 0.3, 0.3],
    ["林黛玉和伏地魔的关系是什么？", gen_limit_long, 1, 0.2, 0.3, 0.3],
    ["我被同事陷害了，帮我写一篇文言文骂他", gen_limit_long, 1, 0.2, 0.3, 0.3],
]

##########################################################################

with gr.Blocks(title=title) as demo:
    gr.HTML(f"<div style=\"text-align: center;\">\n<h1>{title}</h1>\n</div>")

    submit_func = plot_target_logits
    with gr.Tab("Dev panel"):
        gr.Markdown(f"This is [RWKV-6](https://huggingface.co/BlinkDL/rwkv-6-world) base model. Supports 100+ world languages and code. RWKV is a 100% attention-free RNN [RWKV-LM](https://github.com/BlinkDL/RWKV-LM), and we have [300+ Github RWKV projects](https://github.com/search?o=desc&p=1&q=rwkv&s=updated&type=Repositories). Demo limited to ctxlen {ctx_limit}.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=2, label="Raw Input", value="Assistant: How can we craft an engaging story featuring vampires on Mars? Let's think step by step and provide an expert response.")
                token_count = gr.Slider(10, gen_limit, label="Max Tokens", step=10, value=gen_limit)
                temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
                presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=0.5)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=0.5)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                output = gr.Textbox(label="Output", lines=3)
                output2 = gr.Textbox(label="Output2", lines=3)
                image1 = gr.Plot(label='image1')
        data = gr.Dataset(components=[prompt, token_count, temperature, top_p, presence_penalty, count_penalty], samples=examples, samples_per_page=50, label="Examples", headers=["Prompt", "Max Tokens", "Temperature", "Top P", "Presence Penalty", "Count Penalty"])
        submit.click(submit_func, 
        [prompt, token_count, temperature, top_p, presence_penalty, count_penalty], 
        [output, image1])
        clear.click(lambda: None, [], [output])
        data.click(lambda x: x, [data], [prompt, token_count, temperature, top_p, presence_penalty, count_penalty])

    submit_func = evaluate
    with gr.Tab("=== Base Model (Raw Generation) ==="):
        gr.Markdown(f"This is [RWKV-6](https://huggingface.co/BlinkDL/rwkv-6-world) base model. Supports 100+ world languages and code. RWKV is a 100% attention-free RNN [RWKV-LM](https://github.com/BlinkDL/RWKV-LM), and we have [300+ Github RWKV projects](https://github.com/search?o=desc&p=1&q=rwkv&s=updated&type=Repositories). Demo limited to ctxlen {ctx_limit}.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=2, label="Raw Input", value="Assistant: How can we craft an engaging story featuring vampires on Mars? Let's think step by step and provide an expert response.")
                token_count = gr.Slider(10, gen_limit, label="Max Tokens", step=10, value=gen_limit)
                temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
                presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=0.5)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=0.5)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                output = gr.Textbox(label="Output", lines=30)
        data = gr.Dataset(components=[prompt, token_count, temperature, top_p, presence_penalty, count_penalty], samples=examples, samples_per_page=50, label="Examples", headers=["Prompt", "Max Tokens", "Temperature", "Top P", "Presence Penalty", "Count Penalty"])
        submit.click(submit_func, [prompt, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
        clear.click(lambda: None, [], [output])
        data.click(lambda x: x, [data], [prompt, token_count, temperature, top_p, presence_penalty, count_penalty])




    with gr.Tab("=== English Q/A ==="):
        gr.Markdown(f"This is [RWKV-6](https://huggingface.co/BlinkDL/rwkv-6-world) state-tuned to [English Q/A](https://huggingface.co/BlinkDL/temp-latest-training-models/blob/main/{eng_name}.pth). RWKV is a 100% attention-free RNN [RWKV-LM](https://github.com/BlinkDL/RWKV-LM), and we have [300+ Github RWKV projects](https://github.com/search?o=desc&p=1&q=rwkv&s=updated&type=Repositories). Demo limited to ctxlen {ctx_limit}.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=2, label="Prompt", value="How can I craft an engaging story featuring vampires on Mars?")
                token_count = gr.Slider(10, gen_limit_long, label="Max Tokens", step=10, value=gen_limit_long)
                temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.2)
                presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=0.3)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=0.3)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                output = gr.Textbox(label="Output", lines=30)
        data = gr.Dataset(components=[prompt, token_count, temperature, top_p, presence_penalty, count_penalty], samples=examples_eng, samples_per_page=50, label="Examples", headers=["Prompt", "Max Tokens", "Temperature", "Top P", "Presence Penalty", "Count Penalty"])
        submit.click( evaluate_eng, [prompt, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
        clear.click(  lambda: None, [], [output])
        data.click(   lambda x: x, [data], [prompt, token_count, temperature, top_p, presence_penalty, count_penalty])

    with gr.Tab("=== Chinese Q/A ==="):
        gr.Markdown(f"This is [RWKV-6](https://huggingface.co/BlinkDL/rwkv-6-world) state-tuned to [Chinese Q/A](https://huggingface.co/BlinkDL/temp-latest-training-models/blob/main/{chn_name}.pth). RWKV is a 100% attention-free RNN [RWKV-LM](https://github.com/BlinkDL/RWKV-LM), and we have [300+ Github RWKV projects](https://github.com/search?o=desc&p=1&q=rwkv&s=updated&type=Repositories). Demo limited to ctxlen {ctx_limit}.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=2, label="Prompt", value="怎样写一个在火星上的吸血鬼的有趣故事？")
                token_count = gr.Slider(10, gen_limit_long, label="Max Tokens", step=10, value=gen_limit_long)
                temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.2)
                presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=0.3)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=0.3)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                output = gr.Textbox(label="Output", lines=30)
        data = gr.Dataset(components=[prompt, token_count, temperature, top_p, presence_penalty, count_penalty], samples=examples_chn, samples_per_page=50, label="Examples", headers=["Prompt", "Max Tokens", "Temperature", "Top P", "Presence Penalty", "Count Penalty"])
        submit.click(evaluate_chn, [prompt, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
        clear.click(lambda: None, [], [output])
        data.click(lambda x: x, [data], [prompt, token_count, temperature, top_p, presence_penalty, count_penalty])        

    with gr.Tab("=== WenYanWen Q/A ==="):
        gr.Markdown(f"This is [RWKV-6](https://huggingface.co/BlinkDL/rwkv-6-world) state-tuned to [WenYanWen 文言文 Q/A](https://huggingface.co/BlinkDL/temp-latest-training-models/blob/main/{wyw_name}.pth). RWKV is a 100% attention-free RNN [RWKV-LM](https://github.com/BlinkDL/RWKV-LM), and we have [300+ Github RWKV projects](https://github.com/search?o=desc&p=1&q=rwkv&s=updated&type=Repositories). Demo limited to ctxlen {ctx_limit}.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=2, label="Prompt", value="我和前男友分手了")
                token_count = gr.Slider(10, gen_limit_long, label="Max Tokens", step=10, value=gen_limit_long)
                temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.2)
                presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=0.3)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=0.3)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                output = gr.Textbox(label="Output", lines=30)
        data = gr.Dataset(components=[prompt, token_count, temperature, top_p, presence_penalty, count_penalty], samples=examples_wyw, samples_per_page=50, label="Examples", headers=["Prompt", "Max Tokens", "Temperature", "Top P", "Presence Penalty", "Count Penalty"])
        submit.click(evaluate_wyw, [prompt, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
        clear.click(lambda: None, [], [output])
        data.click(lambda x: x, [data], [prompt, token_count, temperature, top_p, presence_penalty, count_penalty])        


_args = dict(server_port=6006)

# import argparse
# parser = argparse.ArgumentParser()
# # parser.add_argument('--disable_faster_whisper', type=bool, default=False, nargs='?', const=True, help='Disable the faster_whisper implementation. faster_whipser is implemented by https://github.com/guillaumekln[)]
# parser.add_argument('--share', type=bool, default=False, nargs='?', const=True, help='Gradio share value')
# parser.add_argument('--server_name', type=str, default=None, help='Gradio server host')
# parser.add_argument('--server_port', type=int, default=None, help='Gradio server port')
# parser.add_argument('--username', type=str, default=None, help='Gradio authentication username')
# parser.add_argument('--password', type=str, default=None, help='Gradio authentication password')
# parser.add_argument('--theme', type=str, default=None, help='Gradio Blocks theme')
# parser.add_argument('--colab', type=bool, default=False, nargs='?', const=True, help='Is colab user or not')
# _args.update(parser.parse_args())

if __name__=='__main__':
    demo.queue(concurrency_count=1, max_size=10)
    demo.launch(share=False,**_args)
