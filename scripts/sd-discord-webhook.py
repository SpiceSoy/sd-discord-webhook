import copy
from datetime import datetime, timedelta
import imghdr
import io
import json
import os
from typing import Any
from urllib import parse

import gradio as gr
import numpy
import torch
import torchvision
from PIL import ImageFilter
from PIL import Image

import PIL
from discord_webhook import DiscordWebhook, DiscordEmbed
from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse
from transformers import AutoFeatureExtractor

from modules import script_callbacks, scripts
from modules import shared
from modules.script_callbacks import ImageSaveParams

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

sd_locale_dict = {
    "ko": {
        "section_name": "디스코드 웹훅",
        "webhook_enable_desc": "디스코드 웹훅 적용 여부",
        "webhook_url_desc": "웹훅 키 URL",
        "webhook_trigger_only_grid": "그리드 이미지인 경우에만 웹훅",
        "webhook_image_url_desc": "이미지 링크 시작 주소",
        "webhook_image_embed_desc": "이미지 임베드 포함 여부",
        "webhook_image_embed_color_desc": "임베드 컬러",
        "webhook_show_used_model": "사용 모델 출력 여부",
        "webhook_show_used_sampler": "사용 샘플러 출력 여부",
        "webhook_show_prompt_desc": "긍정 프롬프트 출력 여부",
        "webhook_show_neg_prompt_desc": "부정 프롬프트 출력 여부",
        "webhook_show_etc_desc": "기타 정보 표시 여부",
        "webhook_blur_image": "이미지 블러 처리 여부",
        "webhook_blur_image_nsfw": "NSFW 감지 이미지 블러 처리 여부",
        "webhook_blur_ratio": "이미지 블러 강도",
        "embed_title": "이미지 생성 완료",
        "embed_Image_link_title": "이미지",
        "embed_Image_link": "보기",
        "embed_resolution_title": "전체 해상도",
        "embed_src_resolution_title": "원본 해상도",
        "embed_model_name": "사용 모델",
        "embed_sampler_name": "사용 샘플러",
        "embed_prompt_title": "프롬프트",
        "embed_neg_prompt_title": "부정 프롬프트",
        "embed_etc_title": "기타 정보",
        "footer_blur": "블러 적용됨",
        "warning_nsfw": "NSFW 이미지 조심",
    },
    "en": {
        "section_name": "Discord Webhook",
        "webhook_enable_desc": "Enable discord webhook",
        "webhook_url_desc": "Webhook token URL",
        "webhook_trigger_only_grid": "Only push in grid image",
        "webhook_image_url_desc": "Image URL start link address",
        "webhook_image_embed_desc": "Image is contained by embed",
        "webhook_image_embed_color_desc": "Embed point color",
        "webhook_show_used_model": "Show Used Model",
        "webhook_show_used_sampler": "Show Used Sampler",
        "webhook_show_prompt_desc": "Show prompt",
        "webhook_show_neg_prompt_desc": "Shot negative prompt",
        "webhook_show_etc_desc": "Show ETC",
        "webhook_blur_image": "Apply blur to image",
        "webhook_blur_image_nsfw": "Apply blur to NSFW image",
        "webhook_blur_ratio": "Image blur ratio",
        "embed_title": "Complete Generate Image",
        "embed_Image_link_title": "Image",
        "embed_Image_link": "Go",
        "embed_resolution_title": "Resolution",
        "embed_src_resolution_title": "Source Resolution",
        "embed_model_name": "Used Model",
        "embed_sampler_name": "Used Sampler",
        "embed_prompt_title": "Prompt",
        "embed_neg_prompt_title": "Negative Prompt",
        "embed_etc_title": "ETC",
        "footer_blur": "Apply Blur",
        "warning_nsfw": "Warning NSFW Image",
    }
}

sd_img_dirs = [
    "outdir_txt2img_samples",
    "outdir_img2img_samples",
    "outdir_save",
    "outdir_extras_samples",
    "outdir_grids",
    "outdir_img2img_grids",
    "outdir_samples",
    "outdir_txt2img_grids",
]


class Locale:
    def __init__(self, lang_key):
        if lang_key in sd_locale_dict.keys():
            self._locale = sd_locale_dict[lang_key]
        else:
            self._locale = sd_locale_dict["ko"]

    def get(self, key):
        if key in self._locale:
            return self._locale[key]
        return key


safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = None
safety_checker = None
nsfw_post_process_id = set()


# check nsfw


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


# check and replace nsfw content
def check_safety(x_image):
    global safety_feature_extractor, safety_checker

    if safety_feature_extractor is None:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)

    return has_nsfw_concept


def censor_batch(x):
    x = copy.deepcopy(x)
    x_samples_ddim_numpy = x.permute(0, 2, 3, 1).numpy()
    has_nsfw_concept = check_safety(x_samples_ddim_numpy)
    return has_nsfw_concept

# Image Host

def get_sd_webui_conf(**kwargs):
    try:
        from modules.shared import opts
        return opts.data
    except:
        pass
    try:
        with open(kwargs.get("sd_webui_config"), "r") as f:
            import json
            return json.loads(f.read())
    except:
        pass
    return {}


def get_valid_img_dirs(
    conf,
    keys=sd_img_dirs,
):
    # 获取配置项
    paths = [conf.get(key) for key in keys]

    # 判断路径是否有效并转为绝对路径
    abs_paths = []
    for path in paths:
        if not path or len(path.strip()) == 0:
            continue
        if os.path.isabs(path):  # 已经是绝对路径
            abs_path = path
        else:  # 转为绝对路径
            abs_path = os.path.join(os.getcwd(), path)
        if os.path.exists(abs_path):  # 判断路径是否存在
            abs_paths.append(os.path.normpath(abs_path))

    return abs_paths


def is_valid_image_path(path):
    abs_path = os.path.abspath(path)  # 转为绝对路径
    if not os.path.exists(abs_path):  # 判断路径是否存在
        return False
    if not os.path.isfile(abs_path):  # 判断是否是文件
        return False
    if not imghdr.what(abs_path):  # 判断是否是图像文件
        return False
    return True


def discord_image_host(_: Any, app: FastAPI, **kwargs):
    pre = "/discord_webhook"


    img_search_dirs = []
    try:
        img_search_dirs = get_valid_img_dirs(get_sd_webui_conf(**kwargs))
    except:
        pass

    def need_cache(path, parent_paths=img_search_dirs):
        try:
            for parent_path in parent_paths:
                if (
                    os.path.commonpath([os.path.normpath(path), parent_path])
                    == parent_path
                ):
                    return True
        except:
            pass
        return False

    @app.get(pre + "/img")
    async def get_file(path: str):
        filename = path
        import mimetypes

        if not os.path.relpath(filename, os.getcwd() ) == filename:
            raise HTTPException(status_code=403, detail=f"{filename} is not allowed { os.getcwd() } || { filename }")
        if not os.path.exists(filename):
            raise HTTPException(status_code=404, detail=f"{filename} is not exists")
        if not os.path.isfile(filename):
            raise HTTPException(status_code=400, detail=f"{filename} is not a file")

        media_type, _ = mimetypes.guess_type(filename)
        headers = {}
        if need_cache(filename) and is_valid_image_path(filename):
            headers[
                "Cache-Control"
            ] = "public, max-age=31536000"
            headers["Expires"] = (datetime.now() + timedelta(days=365)).strftime(
                "%a, %d %b %Y %H:%M:%S GMT"
            )

        return FileResponse(
            filename,
            media_type=media_type,
            headers=headers,
        )


def on_ui_settings():
    if shared.opts.get_default("discord_webhook_locale_target") is not None:
        loc = Locale(shared.opts.discord_webhook_locale_target)
    else:
        loc = Locale("ko")

    section = ('discord', loc.get("section_name"))

    shared.opts.add_option(
        'discord_webhook_enable',
        shared.OptionInfo(
            True,
            loc.get("webhook_enable_desc"),
            gr.Checkbox,
            {"interactive": True},
            section=section)
    )

    shared.opts.add_option(
        'discord_webhook_url',
        shared.OptionInfo(
            "",
            loc.get("webhook_url_desc"),
            gr.Textbox,
            {"interactive": True},
            section=section)
    )

    shared.opts.add_option(
        'discord_webhook_image_url',
        shared.OptionInfo(
            "http://localhost:7860/",
            loc.get("webhook_image_url_desc"),
            gr.Textbox,
            {"interactive": True},
            section=section)
    )

    shared.opts.add_option(
        'discord_webhook_trigger_only_grid',
        shared.OptionInfo(
            False,
            loc.get("webhook_trigger_only_grid"),
            gr.Checkbox,
            {"interactive": True},
            section=section)
    )

    shared.opts.add_option(
        'discord_webhook_image_embed',
        shared.OptionInfo(
            True,
            loc.get("webhook_image_embed_desc"),
            gr.Checkbox,
            {"interactive": True},
            section=section)
    )

    shared.opts.add_option(
        'discord_webhook_blur_image',
        shared.OptionInfo(
            False,
            loc.get("webhook_blur_image"),
            gr.Checkbox,
            {"interactive": True},
            section=section)
    )

    shared.opts.add_option(
        'discord_webhook_blur_image_nsfw',
        shared.OptionInfo(
            True,
            loc.get("webhook_blur_image_nsfw"),
            gr.Checkbox,
            {"interactive": True},
            section=section)
    )

    shared.opts.add_option(
        "discord_webhook_blur_ratio",
        shared.OptionInfo(
            10,
            loc.get("webhook_blur_ratio"),
            gr.Slider,
            {"interactive": True},
            section=section)
    )

    shared.opts.add_option(
        'discord_webhook_embed_color',
        shared.OptionInfo(
            '#03b2f8',
            loc.get("webhook_image_embed_color_desc"),
            gr.ColorPicker,
            {"interactive": True},
            section=section)
    )

    shared.opts.add_option(
        'discord_webhook_show_used_model',
        shared.OptionInfo(
            True,
            loc.get("webhook_show_used_model"),
            gr.Checkbox,
            {"interactive": True},
            section=section)
    )

    shared.opts.add_option(
        'discord_webhook_show_used_sampler',
        shared.OptionInfo(
            True,
            loc.get("webhook_show_used_sampler"),
            gr.Checkbox,
            {"interactive": True},
            section=section)
    )

    shared.opts.add_option(
        'discord_webhook_show_prompt_desc',
        shared.OptionInfo(
            True,
            loc.get("webhook_show_prompt_desc"),
            gr.Checkbox,
            {"interactive": True},
            section=section)
    )

    shared.opts.add_option(
        'discord_webhook_show_neg_prompt_desc',
        shared.OptionInfo(
            True,
            loc.get("webhook_show_neg_prompt_desc"),
            gr.Checkbox,
            {"interactive": True},
            section=section)
    )

    shared.opts.add_option(
        'discord_webhook_show_etc',
        shared.OptionInfo(
            True,
            loc.get("webhook_show_etc_desc"),
            gr.Checkbox,
            {"interactive": True},
            section=section)
    )

    shared.opts.add_option(
        'discord_webhook_locale_target',
        shared.OptionInfo(
            "ko",
            "Locale Json",
            gr.Dropdown,
            {"interactive": True, "choices": list(sd_locale_dict.keys())},
            section=section)
    )


def slice_field(in_str):
    return in_str[0:max(len(in_str), 1000)]


class PramParser:
    def __init__(self, param: str):
        self.param = param
        token_negative = "Negative prompt:"
        token_steps = "Steps:"

        negative_start = self.param.rfind(token_negative)
        steps_start = self.param.find(token_steps)

        self.prompt = slice_field(self.param[0:negative_start].strip())
        self.negative_prompt = slice_field(self.param[negative_start + len(token_negative):steps_start].strip())
        self.etc_params = slice_field(self.param[steps_start:].strip())

        sample_start = self.etc_params.find("Sampler:") + len("Sampler:")
        sample_end = self.etc_params.find(",", sample_start)
        self.sampler = self.etc_params[sample_start:sample_end].strip()

        size_start = self.etc_params.find("Size:") + len("Size:")
        size_end = self.etc_params.find(",", size_start)
        self.source_size = self.etc_params[size_start:size_end].strip()


# https://pypi.org/project/discord-webhook/
def image_saved(param: ImageSaveParams):
    if not shared.opts.discord_webhook_enable:
        return

    print(param.filename[:param.filename.rindex("\\")].endswith("-grids"))

    if shared.opts.discord_webhook_trigger_only_grid and not param.filename[:param.filename.rindex("\\")].endswith("-grids"):
        return

    loc = Locale(shared.opts.discord_webhook_locale_target)

    webhook = DiscordWebhook(url=shared.opts.discord_webhook_url)

    is_nsfw = id(param.p) in nsfw_post_process_id
    use_blur = shared.opts.discord_webhook_blur_image or (shared.opts.discord_webhook_blur_image_nsfw and is_nsfw)

    target_image = param.image.filter(ImageFilter.GaussianBlur(shared.opts.discord_webhook_blur_ratio)) if use_blur else param.image

    buffer = io.BytesIO()
    target_image.save(buffer, format='PNG')

    with open(param.filename, "rb") as f:
        webhook.add_file(file=buffer.getvalue(), filename='output.png')

    abs_path = os.path.abspath(param.filename)
    rel_path = os.path.relpath(param.filename, os.getcwd())
    link_url = f"{shared.opts.discord_webhook_image_url}discord_webhook/img?path={parse.quote(rel_path)}"

    params = PramParser(param.pnginfo['parameters'])

    str_prompt = params.prompt
    str_neg_prompt = params.negative_prompt
    str_etc = params.etc_params
    str_checkpoint_name = shared.sd_model.sd_checkpoint_info.model_name.strip()
    str_sampler_name = params.sampler.strip()
    str_source_size = params.source_size.strip()

    embed = DiscordEmbed(title=loc.get("embed_title"), color=shared.opts.discord_webhook_embed_color[1:])
    embed.add_embed_field(loc.get("embed_Image_link_title"), f"[{loc.get('embed_Image_link')}]({link_url})", True)
    embed.add_embed_field(loc.get("embed_resolution_title"), f"({param.image.size[0]}x{param.image.size[1]})", True)
    embed.add_embed_field(loc.get("embed_src_resolution_title"), f"({str_source_size})", True)

    if shared.opts.discord_webhook_show_used_model:
        embed.add_embed_field(loc.get("embed_model_name"), str_checkpoint_name, True)

    if shared.opts.discord_webhook_show_used_sampler:
        embed.add_embed_field(loc.get("embed_sampler_name"), str_sampler_name, True)

    if shared.opts.discord_webhook_show_prompt_desc:
        embed.add_embed_field(loc.get("embed_prompt_title"), str_prompt, False)

    if shared.opts.discord_webhook_show_neg_prompt_desc:
        embed.add_embed_field(loc.get("embed_neg_prompt_title"), str_neg_prompt, False)

    if shared.opts.discord_webhook_show_etc:
        embed.add_embed_field(loc.get("embed_etc_title"), str_etc, False)

    if shared.opts.discord_webhook_image_embed:
        embed.set_image(url='attachment://output.png')

    footer_str = []

    if is_nsfw:
        footer_str.append(loc.get("warning_nsfw"))

    if use_blur:
        footer_str.append(loc.get("footer_blur"))

        embed.set_footer(text='•'.join(footer_str))

    embed.set_timestamp()
    webhook.add_embed(embed)

    response = webhook.execute()
    pass



class NsfwCheckScript(scripts.Script):
    def title(self):
        return "NSFW check"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess_batch(self, p, *args, **kwargs):
        images = kwargs['images']
        if censor_batch(images):
            nsfw_post_process_id.add(id(p))


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_image_saved(image_saved)
script_callbacks.on_app_started(discord_image_host)