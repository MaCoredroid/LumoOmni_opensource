from PIL import Image
from torch.utils.data import Dataset


class DummyVLMData(Dataset):
    def __init__(self, num_samples, image_size, prompt, image_token):
        self.num_samples = num_samples
        self.image_size = image_size
        self.prompt = prompt
        self.image_token = image_token

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        color = (idx % 255, (idx * 3) % 255, (idx * 7) % 255)
        image = Image.new("RGB", (self.image_size, self.image_size), color)
        prompt = self.prompt
        if self.image_token not in prompt:
            prompt = f"{prompt} {self.image_token}"
        answer = f"sample {idx}"
        return {
            "prompt": prompt,
            "answer": answer,
            "images": [image],
        }
