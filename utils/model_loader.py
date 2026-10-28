import logging
from consts import DEBUG, LOAD_ASR
import state

logger = logging.getLogger(__name__)

def load_model():
    if DEBUG:
        logger.info("Skipping OmniVoice model loading because DEBUG=True")
        return

    logger.info("Start preparing OmniVoice model")
    logger.info(f"Loading ASR model: {LOAD_ASR}")

    from omnivoice import OmniVoice
    import torch

    state.model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda:0",
        dtype=torch.float16,
        load_asr=LOAD_ASR,
    )
    state.model.eval()

    logger.info("OmniVoice model loaded successfully")