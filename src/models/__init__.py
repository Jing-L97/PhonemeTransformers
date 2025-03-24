from transformers import PreTrainedModel
# typing imports
from ..config import ModelParams
from .gpt2 import *
from .llama import *
from .gptneox import *
from .lstm import LSTMConfig, LSTMForLanguageModeling
from .registry import CONFIG_REGISTRY, MODEL_REGISTRY
from ..preprocessing import create_phoneme_map

import logging
logger = logging.getLogger(__name__)

# Register LSTM models to the registry
CONFIG_REGISTRY['lstm_lm'] = LSTMConfig
MODEL_REGISTRY['lstm_lm'] = LSTMForLanguageModeling



def load_model(
    cfg: ModelParams, tokenizer,
) -> PreTrainedModel:
    """Loads the model from the config file
    Args:
        cfg (TransformerSegmentationConfig): hydra config object
        tokenizer (PreTrainedTokenizer): tokenizer object
    """
    model_kwargs = dict(cfg.model_kwargs)
    model_kwargs["vocab_size"] = tokenizer.vocab_size
    model_kwargs["bos_token_id"] = tokenizer.bos_token_id
    model_kwargs["eos_token_id"] = tokenizer.eos_token_id
    
    if cfg.name in MODEL_REGISTRY:
        config = CONFIG_REGISTRY[cfg.name](**model_kwargs)
        
        if config.name_or_path:
            model = MODEL_REGISTRY[cfg.name].from_pretrained(config.name_or_path)
            logger.info(f"Loaded model config from {config.name_or_path}")
        else:
            logging.info(f"Initialising model {cfg.name} with config {config} from scratch")
            
        if cfg.name == 'gpt2_feature_lm' or cfg.name == 'lstm_feature_lm':  # Add LSTM feature model
            phoneme_map = create_phoneme_map(tokenizer)
            model = MODEL_REGISTRY[cfg.name](config, phoneme_map)
        else:
            model = MODEL_REGISTRY[cfg.name](config)
    else:
        raise ValueError(f"Model {cfg.name} not found in registry")
        
    return model