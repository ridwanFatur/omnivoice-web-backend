import logging
from consts import DEBUG

logger = logging.getLogger(__name__)

model = None

def load_model():
    global model

    if DEBUG:
        logger.info("Skipping OmniVoice model loading because DEBUG=True")
        return

    logger.info("Start preparing OmniVoice model")

    from omnivoice import OmniVoice
    import torch

    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda:0",
        dtype=torch.float16,
        # load_asr=True,
    )

    logger.info("OmniVoice model loaded successfully")
        
        