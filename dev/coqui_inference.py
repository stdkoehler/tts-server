from pathlib import Path


import torch
import torchaudio

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


voice_dir = Path("models/ready")
config_dir = Path(__file__).parents[0]
print(config_dir)

[print(i) for i in voice_dir.iterdir()]

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("Cuda available")
else:
    raise ValueError("Cuda not available")

config = XttsConfig()
config.load_json(config_dir / "config.json")
xtts_model = Xtts.init_from_config(config)

xtts_model.load_checkpoint(config, checkpoint_dir=voice_dir)  # , use_deepspeed=True)
if torch.cuda.is_available():
    xtts_model.cuda()
print("Model loaded")

gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
    audio_path=voice_dir / "reference.wav",
)

input_text = """
Your fixer follows your gaze to the watcher, nodding in understanding. "That's Tank. He's a street samurai, one of the best. He's been keeping an eye on the target for Wuxing. He knows the layout of the target's base, and he's got some connections in Kowloon Walled City. He'll be your muscle for this job."
They pause, then add, "But don't get too close. He's got his own reasons for being here, and they don't always align with yours. Just remember, he's on your side for now, but don't turn your back on him."
You nod, acknowledging the warning. You've got a Decker, a street samurai, and a fixer. Time to put the team together and start planning. You thank your fixer, then head back to your hoverbike, ready to start the job."""
out = xtts_model.inference(
    input_text, "en", gpt_cond_latent, speaker_embedding, enable_text_splitting=True
)
torchaudio.save("xtts.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)
