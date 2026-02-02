from typing import Dict, List, Optional

from stage3_uti.tokenization.token_space import TokenSpace


class SequenceBuilder:
    def __init__(self, token_space: TokenSpace) -> None:
        self.token_space = token_space
        self.special = token_space.special_tokens

    def _tok(self, name: str) -> int:
        if name not in self.special:
            raise KeyError(f"special token missing: {name}")
        return int(self.special[name])

    def build_text(self, text_ids: List[int]) -> List[int]:
        return [self._tok("<|text_start|>")] + text_ids + [self._tok("<|text_end|>")]

    def build_image(self, image_ids: List[int]) -> List[int]:
        return [self._tok("<|img_start|>")] + image_ids + [self._tok("<|img_end|>")]

    def build_audio(self, audio_ids: List[int]) -> List[int]:
        return [self._tok("<|aud_start|>")] + audio_ids + [self._tok("<|aud_end|>")]

    def build_t2i(self, text_ids: List[int], image_ids: List[int]) -> List[int]:
        return self.build_text(text_ids) + [self._tok("<|gen_img|>")] + self.build_image(image_ids)

    def build_t2a(self, text_ids: List[int], audio_ids: List[int]) -> List[int]:
        return self.build_text(text_ids) + [self._tok("<|gen_aud|>")] + self.build_audio(audio_ids)

    def build_a2t(self, audio_ids: List[int], text_ids: List[int]) -> List[int]:
        return self.build_audio(audio_ids) + [self._tok("<|gen_text|>")] + self.build_text(text_ids)

    def build_i2t(self, image_ids: List[int], text_ids: List[int]) -> List[int]:
        return self.build_image(image_ids) + [self._tok("<|gen_text|>")] + self.build_text(text_ids)

    def build_multi(
        self,
        *,
        text_ids: Optional[List[int]] = None,
        image_ids: Optional[List[int]] = None,
        audio_ids: Optional[List[int]] = None,
        gen: Optional[str] = None,
    ) -> List[int]:
        seq: List[int] = []
        if text_ids is not None:
            seq.extend(self.build_text(text_ids))
        if image_ids is not None:
            seq.extend(self.build_image(image_ids))
        if audio_ids is not None:
            seq.extend(self.build_audio(audio_ids))
        if gen:
            seq.append(self._tok(gen))
        return seq


def build_supervised_labels(input_ids: List[int], target_span: slice) -> List[int]:
    labels = [-100] * len(input_ids)
    for i in range(target_span.start, target_span.stop):
        if 0 <= i < len(labels):
            labels[i] = input_ids[i]
    return labels


def find_span(input_ids: List[int], start_tok: int, end_tok: int) -> Optional[slice]:
    try:
        start_idx = input_ids.index(start_tok)
        end_idx = input_ids.index(end_tok, start_idx + 1)
    except ValueError:
        return None
    return slice(start_idx, end_idx + 1)


def sequence_to_record(sequence: List[int], meta: Dict[str, int]) -> Dict[str, object]:
    record = {
        "input_ids": sequence,
        "length": len(sequence),
    }
    record.update(meta)
    return record
