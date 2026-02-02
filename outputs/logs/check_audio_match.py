import json
from pathlib import Path
import numpy as np
from stage3_uti.stage2.wds_io import iter_tar_samples
from stage3_uti.tokenization.unified_tokenizer import UnifiedTokenizer
import torchaudio
import soundfile as sf

sample_id='clotho:BI-PLANE TAXI.wav:3'

# find stored tokens
stored=None
for shard in sorted(Path('stage3_uti/data/tokenized/clotho/train').glob('shard-*.tar')):
    for _, sample in iter_tar_samples(shard):
        meta = sample.get('json') or {}
        if str(meta.get('id'))==sample_id:
            stored = sample.get('audio')
            break
    if stored is not None:
        break
print('stored found', stored is not None, 'len', None if stored is None else stored.size)

# load manifest
manifest = None
with open('stage3_uti/data/manifests/clotho.jsonl','r',encoding='utf-8') as f:
    for line in f:
        item=json.loads(line)
        if item.get('id')==sample_id:
            manifest=item
            break
print('manifest found', manifest is not None)

uti = UnifiedTokenizer.from_config('stage3_uti/configs/uti.yaml')
if manifest and stored is not None:
    audio = manifest.get('modalities',{}).get('audio',{})
    path = audio.get('path')
    sr_expected = audio.get('sr')

    # torchaudio
    wav_t, sr_t = torchaudio.load(path)
    tokens_t, _ = uti.encode_audio(wav_t.detach().cpu().numpy().astype(np.float32), int(sr_t))
    tokens_t = np.asarray(tokens_t, dtype=np.int32)
    eq_t = np.array_equal(stored.astype(np.int32), tokens_t)
    print('torchaudio sr', sr_t, 'expected', sr_expected, 'eq', eq_t)

    # soundfile
    data, sr_s = sf.read(path, always_2d=True)
    wav_s = data.T.astype(np.float32)
    tokens_s, _ = uti.encode_audio(wav_s, int(sr_s))
    tokens_s = np.asarray(tokens_s, dtype=np.int32)
    eq_s = np.array_equal(stored.astype(np.int32), tokens_s)
    print('soundfile sr', sr_s, 'expected', sr_expected, 'eq', eq_s)

    if not eq_t:
        diff = stored.astype(np.int64) - tokens_t.astype(np.int64)
        print('torchaudio diff unique', len(np.unique(diff)))
    if not eq_s:
        diff = stored.astype(np.int64) - tokens_s.astype(np.int64)
        print('soundfile diff unique', len(np.unique(diff)))
