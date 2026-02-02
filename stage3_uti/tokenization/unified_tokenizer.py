from typing import Any, Dict, List, Optional, Tuple

import math
import os
import sys

import numpy as np
from PIL import Image, ImageOps
import torch
import yaml

from stage3_uti.tokenization.token_space import build_token_space_from_config, TokenSpace


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return str(obj)


class ImageTokenizerAdapter:
    codebook_size: int
    handles_preprocess: bool = False

    def encode(self, image: Image.Image) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError

    def decode(self, tokens: np.ndarray, meta: Dict[str, Any]) -> Image.Image:
        raise NotImplementedError


class AudioCodecAdapter:
    codebook_size: int
    n_codebooks: int
    sample_rate: int
    channels: int

    def encode(self, wav: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError

    def decode(self, codes: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
        raise NotImplementedError


class DummyImageTokenizer(ImageTokenizerAdapter):
    def __init__(self, grid_size: int = 16, codebook_size: int = 256) -> None:
        self.grid_size = int(grid_size)
        self.codebook_size = int(codebook_size)

    def encode(self, image: Image.Image) -> Tuple[np.ndarray, Dict[str, Any]]:
        img = image.convert("L").resize((self.grid_size, self.grid_size), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.uint8)
        tokens = arr.flatten().astype(np.int64)
        meta = {"grid": [self.grid_size, self.grid_size]}
        return tokens, meta

    def decode(self, tokens: np.ndarray, meta: Dict[str, Any]) -> Image.Image:
        grid = meta.get("grid", [self.grid_size, self.grid_size])
        arr = np.array(tokens, dtype=np.uint8).reshape(grid[0], grid[1])
        img = Image.fromarray(arr, mode="L").convert("RGB")
        return img


class DummyAudioCodec(AudioCodecAdapter):
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        n_codebooks: int = 2,
        codebook_size: int = 16,
        frame_size: int = 160,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.frame_size = int(frame_size)

    def encode(self, wav: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        if sample_rate != self.sample_rate:
            raise ValueError("DummyAudioCodec expects sample_rate matched to config")
        if wav.ndim == 1:
            wav = wav[None, :]
        if wav.shape[0] != self.channels:
            raise ValueError("DummyAudioCodec channel mismatch")
        length = wav.shape[-1]
        n_frames = max(1, int(np.ceil(length / self.frame_size)))
        padded = np.zeros((self.channels, n_frames * self.frame_size), dtype=np.float32)
        padded[:, :length] = wav
        frames = padded.reshape(self.channels, n_frames, self.frame_size)
        frame_vals = frames.mean(axis=-1)
        # Quantize to [0, codebook_size-1]
        norm = np.clip((frame_vals + 1.0) / 2.0, 0.0, 1.0)
        codes = (norm * (self.codebook_size - 1)).round().astype(np.int64)
        # Repeat across codebooks for determinism
        codes = np.repeat(codes[:1, :], self.n_codebooks, axis=0)
        meta = {"n_frames": int(n_frames)}
        return codes, meta

    def decode(self, codes: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
        if codes.ndim != 2:
            raise ValueError("codes must be [n_codebooks, n_frames]")
        n_frames = codes.shape[1]
        vals = codes[0].astype(np.float32) / float(self.codebook_size - 1)
        vals = vals * 2.0 - 1.0
        wav = np.repeat(vals, self.frame_size)
        return wav.astype(np.float32)


class SeedImageTokenizerAdapter(ImageTokenizerAdapter):
    handles_preprocess = True

    def __init__(
        self,
        *,
        model_path: str,
        seed2_path: Optional[str] = None,
        diffusion_model_path: Optional[str] = None,
        load_diffusion: bool = False,
        image_size: int = 224,
        device: str = "cuda",
        fp16: bool = True,
    ) -> None:
        if not os.path.exists(model_path) and model_path.startswith("/media/mark/SHARED/lumoOmni"):
            for repl in ("/home/mark/shared/lumoOmni", "/workspace/lumoOmni"):
                candidate = model_path.replace("/media/mark/SHARED/lumoOmni", repl)
                if os.path.exists(candidate):
                    model_path = candidate
                    break
        if os.path.isdir(model_path):
            candidate = os.path.join(model_path, "seed_quantizer.pt")
            if os.path.exists(candidate):
                model_path = candidate
        self.model_path = model_path
        self.seed2_path = seed2_path
        if diffusion_model_path and not os.path.exists(diffusion_model_path):
            if diffusion_model_path.startswith("/media/mark/SHARED/lumoOmni"):
                for repl in ("/home/mark/shared/lumoOmni", "/workspace/lumoOmni"):
                    candidate = diffusion_model_path.replace("/media/mark/SHARED/lumoOmni", repl)
                    if os.path.exists(candidate):
                        diffusion_model_path = candidate
                        break
        self.diffusion_model_path = diffusion_model_path
        self.load_diffusion = load_diffusion
        self.image_size = int(image_size)
        self.device = device
        self.fp16 = bool(fp16)

        ImageTokenizer = _import_seed_image_tokenizer(seed2_path)
        self._tokenizer = ImageTokenizer(
            model_path=model_path,
            diffusion_model_path=diffusion_model_path,
            load_diffusion=load_diffusion,
            image_size=self.image_size,
            device=device,
            fp16=self.fp16,
        )
        self.codebook_size = int(len(self._tokenizer))

    def encode(self, image: Image.Image) -> Tuple[np.ndarray, Dict[str, Any]]:
        image = image.convert("RGB")
        orig_w, orig_h = image.size
        img_tensor = self._tokenizer.processor(image).to(self.device)
        with torch.no_grad():
            codes = self._tokenizer.encode(img_tensor)
        codes = codes.detach().cpu().numpy().astype(np.int64)
        tokens = codes.reshape(-1)
        grid = None
        if codes.ndim == 2:
            grid = [int(codes.shape[1])]
        meta = {
            "orig_size": [int(orig_h), int(orig_w)],
            "proc_size": [self.image_size, self.image_size],
            "grid": grid,
        }
        return tokens, meta

    def decode(self, tokens: np.ndarray, meta: Dict[str, Any]) -> Image.Image:
        decode_mode = None
        if isinstance(meta, dict):
            decode_mode = meta.get("decode_mode")
        if decode_mode in {"deterministic", "grid"} or not self.load_diffusion or getattr(
            self._tokenizer, "diffusion_model", None
        ) is None:
            return _tokens_to_grid_image(tokens, meta, self.codebook_size, self.image_size)
        indices = torch.tensor(tokens, dtype=torch.long, device=self.device).view(1, -1)
        with torch.no_grad():
            images = self._tokenizer.decode(indices)
        if isinstance(images, list):
            return images[0]
        return images


class EncodecAudioCodecAdapter(AudioCodecAdapter):
    def __init__(
        self,
        *,
        name_or_path: str,
        device: str = "cuda",
        target_bandwidth: Optional[float] = None,
        n_codebooks: Optional[int] = None,
    ) -> None:
        from transformers import EncodecModel

        if not os.path.exists(name_or_path) and name_or_path.startswith("/media/mark/SHARED/lumoOmni"):
            for repl in ("/home/mark/shared/lumoOmni", "/workspace/lumoOmni"):
                candidate = name_or_path.replace("/media/mark/SHARED/lumoOmni", repl)
                if os.path.exists(candidate):
                    name_or_path = candidate
                    break
        self.name_or_path = name_or_path
        self.device = device
        self.model = EncodecModel.from_pretrained(name_or_path).to(device)
        self.model.eval()

        cfg = self.model.config
        self.sample_rate = int(cfg.sampling_rate)
        self.channels = int(cfg.audio_channels)
        self.codebook_size = int(cfg.codebook_size)

        hop_length = int(np.prod(cfg.upsampling_ratios))
        frame_rate = float(self.sample_rate) / float(hop_length)
        possible = []
        for bw in cfg.target_bandwidths:
            nc = bw * 1000.0 / (frame_rate * math.log2(self.codebook_size))
            possible.append(int(round(nc)))
        self._bandwidth_map = {int(nc): float(cfg.target_bandwidths[i]) for i, nc in enumerate(possible)}

        if n_codebooks is not None:
            if int(n_codebooks) not in self._bandwidth_map:
                raise ValueError(f"n_codebooks must be one of {sorted(self._bandwidth_map)}")
            self.n_codebooks = int(n_codebooks)
            self.bandwidth = self._bandwidth_map[self.n_codebooks]
        elif target_bandwidth is not None:
            self.bandwidth = float(target_bandwidth)
            closest = min(self._bandwidth_map.items(), key=lambda kv: abs(kv[1] - self.bandwidth))
            self.n_codebooks = int(closest[0])
        else:
            self.n_codebooks = max(self._bandwidth_map.keys())
            self.bandwidth = self._bandwidth_map[self.n_codebooks]

    def encode(self, wav: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        if int(sample_rate) != int(self.sample_rate):
            raise ValueError("EncodecAudioCodecAdapter expects audio at model sample_rate")
        if wav.ndim != 2:
            raise ValueError("wav must be [C, T]")
        wav_t = torch.from_numpy(wav).to(self.device)
        wav_t = wav_t.unsqueeze(0)
        with torch.no_grad():
            try:
                out = self.model.encode(wav_t, None, self.bandwidth)
            except TypeError:
                out = self.model.encode(wav_t, bandwidth=self.bandwidth)

        codes, scales, last_frame_pad_length = _unpack_encodec_encode(out)
        codes_np, codes_meta = _flatten_encodec_codes(codes, last_frame_pad_length)
        scale = _extract_encodec_scale(scales)

        meta = {}
        meta.update(codes_meta)
        if scale is not None:
            meta["scales"] = scale
        return codes_np, meta

    def decode(self, codes: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
        codes_t, scale_t, last_frame_pad_length = _inflate_encodec_codes(codes, meta, self.device)
        with torch.no_grad():
            try:
                wav = self.model.decode(codes_t, scale_t, last_frame_pad_length=last_frame_pad_length)
            except TypeError:
                wav = self.model.decode(codes_t, scale_t)
        if hasattr(wav, "audio_values"):
            wav = wav.audio_values
        if isinstance(wav, (list, tuple)):
            wav = wav[0]
        wav = wav.detach().cpu().numpy()
        if wav.ndim == 3:
            wav = wav[0]
        return wav.astype(np.float32)


def _unpack_encodec_encode(out):
    if isinstance(out, tuple):
        codes = out[0] if len(out) > 0 else None
        scales = out[1] if len(out) > 1 else None
        last_frame_pad_length = out[2] if len(out) > 2 else 0
        return codes, scales, int(last_frame_pad_length or 0)
    codes = getattr(out, "audio_codes", None)
    if codes is None:
        codes = getattr(out, "codes", None)
    scales = getattr(out, "audio_scales", None)
    if scales is None:
        scales = getattr(out, "scales", None)
    last_frame_pad_length = getattr(out, "last_frame_pad_length", 0)
    return codes, scales, int(last_frame_pad_length or 0)


def _flatten_encodec_codes(codes, last_frame_pad_length: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    if isinstance(codes, (list, tuple)):
        if len(codes) >= 1:
            codes = codes[0]
    if torch.is_tensor(codes):
        codes = codes.detach().cpu().numpy()
    if not isinstance(codes, np.ndarray):
        raise TypeError("encodec codes must be numpy array or torch tensor")

    meta: Dict[str, Any] = {}
    if codes.ndim == 4:
        nb_frames, batch, n_q, frame_len = codes.shape
        if batch != 1:
            raise ValueError("encodec batch size >1 not supported")
        codes = codes[:, 0]  # (nb_frames, n_q, frame_len)
        codes = np.transpose(codes, (1, 0, 2)).reshape(n_q, nb_frames * frame_len)
        meta = {
            "nb_frames": int(nb_frames),
            "frame_len": int(frame_len),
            "last_frame_pad_length": int(last_frame_pad_length),
        }
    elif codes.ndim == 3:
        batch, n_q, frame_len = codes.shape
        if batch != 1:
            raise ValueError("encodec batch size >1 not supported")
        codes = codes[0]
        meta = {
            "nb_frames": 1,
            "frame_len": int(frame_len),
            "last_frame_pad_length": int(last_frame_pad_length),
        }
    elif codes.ndim == 2:
        meta = {"last_frame_pad_length": int(last_frame_pad_length)}
    else:
        raise ValueError("unexpected encodec codes shape")
    return codes.astype(np.int64), meta


def _extract_encodec_scale(scales):
    if scales is None:
        return None
    if isinstance(scales, (list, tuple)):
        if len(scales) == 1:
            return _extract_encodec_scale(scales[0])
        return [_extract_encodec_scale(s) for s in scales]
    if torch.is_tensor(scales):
        scales = scales.detach().cpu().numpy()
    if isinstance(scales, np.ndarray):
        return scales.tolist()
    return scales


def _inflate_encodec_codes(
    codes: np.ndarray, meta: Optional[Dict[str, Any]], device: str
) -> Tuple[torch.Tensor, Optional[list], int]:
    if codes.ndim != 2:
        raise ValueError("encodec codes must be [n_codebooks, n_frames]")
    nb_frames = None
    frame_len = None
    last_frame_pad_length = 0
    scales = None
    if meta:
        nb_frames = meta.get("nb_frames")
        frame_len = meta.get("frame_len")
        last_frame_pad_length = int(meta.get("last_frame_pad_length", 0) or 0)
        if "scales" in meta:
            scales = meta.get("scales")
        elif "scale" in meta:
            scales = meta.get("scale")

    if nb_frames and frame_len:
        n_q = codes.shape[0]
        expected = int(nb_frames) * int(frame_len)
        if codes.shape[1] != expected:
            raise ValueError("encodec codes length does not match frame metadata")
        reshaped = codes.reshape(n_q, int(nb_frames), int(frame_len))
        reshaped = np.transpose(reshaped, (1, 0, 2))  # (nb_frames, n_q, frame_len)
        codes_t = torch.from_numpy(reshaped).to(device).unsqueeze(1)
    else:
        codes_t = torch.from_numpy(codes).to(device).unsqueeze(0)

    scale_t = None
    if scales is not None:
        if isinstance(scales, list):
            scale_t = [torch.tensor(s, device=device) for s in scales]
        else:
            scale_t = [torch.tensor(scales, device=device)]
    return codes_t, scale_t, last_frame_pad_length


def _import_seed_image_tokenizer(seed2_path: Optional[str]):
    try:
        import diffusers.utils as _du
        if not hasattr(_du, "randn_tensor"):
            from diffusers.utils.torch_utils import randn_tensor as _randn_tensor

            _du.randn_tensor = _randn_tensor
    except Exception:
        pass
    try:
        import transformers.modeling_utils as _mu
        if not hasattr(_mu, "apply_chunking_to_forward"):
            from transformers.pytorch_utils import apply_chunking_to_forward as _acf

            _mu.apply_chunking_to_forward = _acf
        if not hasattr(_mu, "find_pruneable_heads_and_indices"):
            try:
                from transformers.pytorch_utils import (
                    find_pruneable_heads_and_indices as _fphi,
                    prune_linear_layer as _pll,
                )
            except Exception:
                import torch
                import torch.nn as nn

                def _fphi(heads, n_heads, head_size, already_pruned_heads):
                    mask = torch.ones(n_heads, head_size)
                    heads = set(heads) - already_pruned_heads
                    for head in heads:
                        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                        mask[head] = 0
                    mask = mask.view(-1).contiguous().eq(1)
                    index = torch.arange(len(mask))[mask].long()
                    return heads, index

                def _pll(layer, index, dim=0):
                    weight = layer.weight.index_select(dim, index).clone().detach()
                    if layer.bias is not None:
                        if dim == 1:
                            bias = layer.bias.clone().detach()
                        else:
                            bias = layer.bias[index].clone().detach()
                    new_size = list(layer.weight.size())
                    new_size[dim] = len(index)
                    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(
                        layer.weight.device
                    )
                    new_layer.weight.requires_grad = False
                    new_layer.weight.copy_(weight)
                    new_layer.weight.requires_grad = True
                    if layer.bias is not None:
                        new_layer.bias.requires_grad = False
                        new_layer.bias.copy_(bias)
                        new_layer.bias.requires_grad = True
                    return new_layer

            _mu.find_pruneable_heads_and_indices = _fphi
            _mu.prune_linear_layer = _pll
        try:
            from transformers import PreTrainedModel as _PTM

            if not hasattr(_PTM, "_lumo_patched_tied_weights"):
                orig_mark = _PTM.mark_tied_weights_as_initialized

                def _patched_mark(self):
                    if not hasattr(self, "all_tied_weights_keys"):
                        keys = getattr(self, "_tied_weights_keys", None)
                        if isinstance(keys, dict):
                            self.all_tied_weights_keys = dict(keys)
                        elif isinstance(keys, (list, tuple, set)):
                            self.all_tied_weights_keys = {k: k for k in keys}
                        else:
                            self.all_tied_weights_keys = {}
                    return orig_mark(self)

                _PTM.mark_tied_weights_as_initialized = _patched_mark
                _PTM._lumo_patched_tied_weights = True
        except Exception:
            pass
    except Exception:
        pass

    if seed2_path:
        if not os.path.exists(seed2_path) and seed2_path.startswith("/media/mark/SHARED/lumoOmni"):
            for repl in ("/home/mark/shared/lumoOmni", "/workspace/lumoOmni"):
                candidate = seed2_path.replace("/media/mark/SHARED/lumoOmni", repl)
                if os.path.exists(candidate):
                    seed2_path = candidate
                    break
        if not os.path.exists(seed2_path):
            alt = os.path.join(os.getcwd(), "refs", "stage3", "AnyGPT")
            if os.path.exists(alt):
                seed2_path = alt
        seed2_path = os.path.abspath(seed2_path)
        if os.path.isdir(seed2_path):
            if os.path.basename(seed2_path) == "seed2":
                parent = os.path.dirname(seed2_path)
            else:
                parent = seed2_path
            if parent not in sys.path:
                sys.path.insert(0, parent)
    try:
        from seed2.seed_llama_tokenizer import ImageTokenizer  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Failed to import seed2 ImageTokenizer. Set image_tokenizer.seed2_path to AnyGPT repo root."
        ) from exc
    return ImageTokenizer


def _normalize_image(img: Image.Image, resolution: int, resize: str, crop: str) -> Tuple[Image.Image, Dict[str, Any]]:
    img = img.convert("RGB")
    orig_w, orig_h = img.size
    if crop == "center":
        img = ImageOps.fit(img, (resolution, resolution), method=Image.BICUBIC, centering=(0.5, 0.5))
    else:
        img = img.resize((resolution, resolution), Image.BICUBIC)
    proc_w, proc_h = img.size
    meta = {
        "orig_size": [orig_h, orig_w],
        "proc_size": [proc_h, proc_w],
        "image_mode": "RGB",
        "preprocess": {"resize": resize, "crop": crop},
    }
    return img, meta


def _tokens_to_grid_image(
    tokens: np.ndarray,
    meta: Optional[Dict[str, Any]],
    codebook_size: int,
    image_size: int,
) -> Image.Image:
    tokens = np.asarray(tokens, dtype=np.int64).flatten()
    grid = None
    if isinstance(meta, dict):
        grid = meta.get("grid")
    if isinstance(grid, list) and len(grid) == 2:
        grid_h, grid_w = int(grid[0]), int(grid[1])
    elif isinstance(grid, list) and len(grid) == 1:
        grid_h, grid_w = 1, int(grid[0])
    else:
        grid_h, grid_w = 1, int(tokens.shape[0])

    total = grid_h * grid_w
    if tokens.shape[0] < total:
        pad = np.zeros((total - tokens.shape[0],), dtype=np.int64)
        tokens = np.concatenate([tokens, pad], axis=0)
    elif tokens.shape[0] > total:
        tokens = tokens[:total]

    if codebook_size <= 1:
        norm = np.zeros_like(tokens, dtype=np.float32)
    else:
        norm = tokens.astype(np.float32) / float(codebook_size - 1)
    grid_img = (norm.reshape(grid_h, grid_w) * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(grid_img, mode="L").resize((image_size, image_size), Image.NEAREST)
    return img.convert("RGB")


def _normalize_audio(
    wav: np.ndarray,
    sample_rate: int,
    target_sample_rate: int,
    channels: int,
    clip_seconds: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if wav.ndim == 1:
        wav = wav[None, :]
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)
    wav = np.clip(wav, -1.0, 1.0)

    preprocess = {"resample": None, "mono": False, "trim_or_pad": "pad_end"}

    if wav.shape[0] != channels:
        if channels == 1:
            wav = np.mean(wav, axis=0, keepdims=True)
            preprocess["mono"] = True
        elif wav.shape[0] == 1:
            wav = np.repeat(wav, channels, axis=0)
            preprocess["mono"] = True
            preprocess["upmix"] = f"repeat_{channels}"
        else:
            raise ValueError("channel policy mismatch")

    if sample_rate != target_sample_rate:
        wav = _resample_audio(wav, sample_rate, target_sample_rate)
        preprocess["resample"] = "torchaudio|scipy"

    target_len = int(round(float(clip_seconds) * float(target_sample_rate)))
    if wav.shape[1] < target_len:
        pad = target_len - wav.shape[1]
        wav = np.pad(wav, ((0, 0), (0, pad)), mode="constant")
    elif wav.shape[1] > target_len:
        wav = wav[:, :target_len]

    meta = {
        "orig_sample_rate": int(sample_rate),
        "sample_rate": int(target_sample_rate),
        "channels": int(channels),
        "clip_seconds": float(clip_seconds),
        "preprocess": preprocess,
    }
    return wav.astype(np.float32), meta


def _resample_audio(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    try:
        import torchaudio

        tensor = np.asarray(wav)
        t = torch.from_numpy(tensor)
        resampled = torchaudio.functional.resample(t, orig_sr, target_sr)
        return resampled.numpy()
    except Exception:
        try:
            from scipy.signal import resample_poly

            resampled = resample_poly(wav, target_sr, orig_sr, axis=-1)
            return resampled.astype(np.float32)
        except Exception as exc:
            raise ImportError("No resampler available; install torchaudio or scipy") from exc


class UnifiedTokenizer:
    def __init__(
        self,
        *,
        text_tokenizer,
        image_tokenizer: ImageTokenizerAdapter,
        audio_codec: AudioCodecAdapter,
        token_space: TokenSpace,
        cfg: Dict[str, Any],
    ) -> None:
        self.text_tokenizer = text_tokenizer
        self.image_tokenizer = image_tokenizer
        self.audio_codec = audio_codec
        self.token_space = token_space
        self.cfg = cfg

        self.image_cfg = cfg.get("image_tokenizer", {})
        self.audio_cfg = cfg.get("audio_codec", {})

    @classmethod
    def from_config(
        cls,
        cfg_path: str,
        *,
        text_tokenizer=None,
        image_tokenizer: Optional[ImageTokenizerAdapter] = None,
        audio_codec: Optional[AudioCodecAdapter] = None,
    ) -> "UnifiedTokenizer":
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        def _remap_path(path: Optional[str]) -> Optional[str]:
            if not path or not isinstance(path, str):
                return path
            if path.startswith("/media/mark/SHARED/lumoOmni"):
                for repl in ("/home/mark/shared/lumoOmni", "/workspace"):
                    candidate = path.replace("/media/mark/SHARED/lumoOmni", repl)
                    if os.path.exists(candidate):
                        return candidate
            return path

        if text_tokenizer is None:
            from transformers import AutoTokenizer

            text_tokenizer = AutoTokenizer.from_pretrained(cfg["base_text_model"])

        if image_tokenizer is None:
            image_cfg = cfg.get("image_tokenizer", {})
            if os.getenv("UTI_SKIP_IMAGE") == "1":
                codebook_size = int(image_cfg.get("codebook_size", 8192))
                image_tokenizer = DummyImageTokenizer(grid_size=16, codebook_size=codebook_size)
            else:
                seed2_path = _remap_path(image_cfg.get("seed2_path"))
                model_path = _remap_path(image_cfg.get("local_path")) or image_cfg.get("name_or_path")
                diffusion_path = _remap_path(image_cfg.get("diffusion_model_path"))
                device = os.getenv("UTI_IMAGE_DEVICE", str(image_cfg.get("device", "cuda")))
                fp16 = bool(image_cfg.get("fp16", True))
                try:
                    import torch

                    if device.startswith("cuda") and not torch.cuda.is_available():
                        device = "cpu"
                    if device == "cpu":
                        fp16 = False
                except Exception:
                    pass
                if model_path is None:
                    raise ValueError("image_tokenizer.name_or_path or local_path required")
                skip_diffusion = os.getenv("UTI_SKIP_DIFFUSION") == "1"
                load_diffusion = bool(image_cfg.get("load_diffusion", False)) and not skip_diffusion
                image_tokenizer = SeedImageTokenizerAdapter(
                    model_path=str(model_path),
                    seed2_path=seed2_path,
                    diffusion_model_path=diffusion_path,
                    load_diffusion=load_diffusion,
                    image_size=int(image_cfg.get("image_size", 224)),
                    device=device,
                    fp16=fp16,
                )
        if audio_codec is None:
            audio_cfg = cfg.get("audio_codec", {})
            name_or_path = _remap_path(audio_cfg.get("local_path")) or audio_cfg.get("name_or_path")
            if name_or_path is None:
                raise ValueError("audio_codec.name_or_path or local_path required")
            device = os.getenv("UTI_AUDIO_DEVICE", str(audio_cfg.get("device", "cuda")))
            try:
                import torch

                if device.startswith("cuda") and not torch.cuda.is_available():
                    device = "cpu"
            except Exception:
                pass
            audio_codec = EncodecAudioCodecAdapter(
                name_or_path=str(name_or_path),
                device=device,
                target_bandwidth=audio_cfg.get("target_bandwidth"),
                n_codebooks=audio_cfg.get("n_codebooks"),
            )

        text_vocab_size = len(text_tokenizer)
        image_codebook_size = int(getattr(image_tokenizer, "codebook_size"))
        audio_codebook_size = int(getattr(audio_codec, "codebook_size"))
        n_codebooks = int(getattr(audio_codec, "n_codebooks"))

        token_space = build_token_space_from_config(
            cfg,
            text_vocab_size=text_vocab_size,
            image_codebook_size=image_codebook_size,
            audio_codebook_size=audio_codebook_size,
            n_codebooks=n_codebooks,
        )
        return cls(
            text_tokenizer=text_tokenizer,
            image_tokenizer=image_tokenizer,
            audio_codec=audio_codec,
            token_space=token_space,
            cfg=cfg,
        )

    def encode_text(self, text: str) -> List[int]:
        ids = self.text_tokenizer.encode(text, add_special_tokens=False)
        return [int(i) for i in ids]

    def encode_image(self, img: Image.Image | np.ndarray) -> Tuple[List[int], Dict[str, Any]]:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if not isinstance(img, Image.Image):
            raise TypeError("encode_image expects PIL.Image or np.ndarray")

        if getattr(self.image_tokenizer, "handles_preprocess", False):
            img_proc = img.convert("RGB")
            meta = {
                "orig_size": [img.height, img.width],
                "proc_size": [img.height, img.width],
                "image_mode": "RGB",
                "preprocess": {"resize": "adapter", "crop": "adapter"},
            }
            tokens, tok_meta = self.image_tokenizer.encode(img_proc)
            proc_size = tok_meta.get("proc_size")
            if proc_size is not None:
                meta["proc_size"] = proc_size
        else:
            resolution = int(self.image_cfg.get("resolution", 256))
            resize = str(self.image_cfg.get("resize", "bicubic"))
            crop = str(self.image_cfg.get("crop", "center"))

            img_proc, meta = _normalize_image(img, resolution, resize, crop)
            tokens, tok_meta = self.image_tokenizer.encode(img_proc)
        tokens = np.asarray(tokens, dtype=np.int64).flatten().tolist()
        global_tokens = [self.token_space.img_to_global(t) for t in tokens]

        meta.update(
            {
                "modality": "image",
                "tokenizer": self.image_cfg.get("name_or_path"),
                "grid": tok_meta.get("grid"),
                "n_tokens": int(len(tokens)),
                "dtype": "uint8",
            }
        )
        meta.update({"token_meta": _to_jsonable(tok_meta)})
        return global_tokens, _to_jsonable(meta)

    def encode_audio(self, wav: np.ndarray, sample_rate: int) -> Tuple[List[int], Dict[str, Any]]:
        if not isinstance(wav, np.ndarray):
            raise TypeError("encode_audio expects np.ndarray")
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)
        wav = np.clip(wav, -1.0, 1.0)

        target_sr = int(self.audio_cfg.get("target_sample_rate", self.audio_codec.sample_rate))
        channels = int(self.audio_cfg.get("channels", self.audio_codec.channels))
        clip_seconds = float(self.audio_cfg.get("clip_seconds", 10.0))

        wav_proc, meta = _normalize_audio(wav, sample_rate, target_sr, channels, clip_seconds)
        codes, codec_meta = self.audio_codec.encode(wav_proc, target_sr)
        if codes.ndim != 2:
            raise ValueError("audio codec must return codes [n_codebooks, n_frames]")

        n_codebooks = int(codes.shape[0])
        n_frames = int(codes.shape[1])
        tokens = []
        for t in range(n_frames):
            for cb in range(n_codebooks):
                tokens.append(self.token_space.aud_to_global(cb, int(codes[cb, t])))

        meta.update(
            {
                "modality": "audio",
                "codec": self.audio_cfg.get("name_or_path"),
                "n_codebooks": n_codebooks,
                "codebook_size": int(self.audio_codec.codebook_size),
                "n_frames": n_frames,
                "serialization": self.audio_cfg.get("serialization", "flatten_by_time_interleaved_codebooks"),
                "token_count": int(len(tokens)),
            }
        )
        meta.update({"codec_meta": _to_jsonable(codec_meta)})
        return [int(t) for t in tokens], _to_jsonable(meta)

    def decode_image(self, tokens: List[int], meta: Dict[str, Any]) -> Image.Image:
        local_tokens = [self.token_space.img_from_global(int(t)) for t in tokens]
        tok_meta = meta.get("token_meta") or {}
        if "decode_mode" in meta and "decode_mode" not in tok_meta:
            tok_meta["decode_mode"] = meta["decode_mode"]
        if "grid" in meta and "grid" not in tok_meta:
            tok_meta["grid"] = meta["grid"]
        return self.image_tokenizer.decode(np.array(local_tokens, dtype=np.int64), tok_meta)

    def decode_audio(self, tokens: List[int], meta: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        n_codebooks = int(meta.get("n_codebooks"))
        n_frames = int(meta.get("n_frames"))
        expected = n_codebooks * n_frames
        if len(tokens) != expected:
            raise ValueError("token_count mismatch for audio decode")

        codes = np.zeros((n_codebooks, n_frames), dtype=np.int64)
        idx = 0
        for t in range(n_frames):
            for cb in range(n_codebooks):
                cb_idx, tok = self.token_space.aud_from_global(int(tokens[idx]))
                if cb_idx != cb:
                    raise ValueError("audio token codebook mismatch")
                codes[cb, t] = tok
                idx += 1

        codec_meta = meta.get("codec_meta") if isinstance(meta, dict) else None
        wav = self.audio_codec.decode(codes, codec_meta)
        return wav, int(meta.get("sample_rate"))
