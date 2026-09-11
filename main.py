import os
import json
import sys
import numpy as np
import onnxruntime as ort

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

sys.path.insert(0, r".\indictrans2-en-indic-int8")

from tokenization_indictrans import IndicTransTokenizer
from tokenizers import Tokenizer


MODEL_PATH = r".\indictrans2-en-indic-int8"
INDIC_MODEL_PATH = r".\indictrans2-indic-indic-int8"
INDIC_EN_MODEL_PATH = r".\indictrans2-indic-en-200m"


tokenizer = IndicTransTokenizer(
    src_vocab_fp=os.path.join(MODEL_PATH, "dict.SRC.json"),
    tgt_vocab_fp=os.path.join(MODEL_PATH, "dict.TGT.json"),
    src_spm_fp=os.path.join(MODEL_PATH, "model.SRC"),
    tgt_spm_fp=os.path.join(MODEL_PATH, "model.TGT")
)


encoder = ort.InferenceSession(
    os.path.join(MODEL_PATH, "encoder_model.onnx"),
    providers=["CPUExecutionProvider"]
)

decoder = ort.InferenceSession(
    os.path.join(MODEL_PATH, "decoder_model.onnx"),
    providers=["CPUExecutionProvider"]
)

decoder_with_past = ort.InferenceSession(
    os.path.join(MODEL_PATH, "decoder_with_past_model.onnx"),
    providers=["CPUExecutionProvider"]
)


num_layers = (
    len(decoder.get_outputs()) - 1
) // 4


indic_src_tokenizer = Tokenizer.from_file(
    os.path.join(INDIC_MODEL_PATH, "tokenizer_src.json")
)

indic_tgt_tokenizer = Tokenizer.from_file(
    os.path.join(INDIC_MODEL_PATH, "tokenizer_tgt.json")
)

indic_meta = json.load(
    open(
        os.path.join(INDIC_MODEL_PATH, "tokenizer_meta.json"),
        encoding="utf-8"
    )
)

indic_gen_config = json.load(
    open(
        os.path.join(INDIC_MODEL_PATH, "generation_config.json"),
        encoding="utf-8"
    )
)

indic_encoder = ort.InferenceSession(
    os.path.join(INDIC_MODEL_PATH, "encoder_model.onnx"),
    providers=["CPUExecutionProvider"]
)

indic_decoder = ort.InferenceSession(
    os.path.join(INDIC_MODEL_PATH, "decoder_model.onnx"),
    providers=["CPUExecutionProvider"]
)

indic_decoder_with_past = ort.InferenceSession(
    os.path.join(INDIC_MODEL_PATH, "decoder_with_past_model.onnx"),
    providers=["CPUExecutionProvider"]
)

indic_decoder_start_id = int(
    indic_gen_config["decoder_start_token_id"]
)

indic_eos_id = int(
    indic_gen_config["eos_token_id"]
)

indic_num_layers = (
    len(indic_decoder.get_outputs()) - 1
) // 4


indic_en_src_tokenizer = Tokenizer.from_file(
    os.path.join(INDIC_EN_MODEL_PATH, "tokenizer_src.json")
)

indic_en_tgt_tokenizer = Tokenizer.from_file(
    os.path.join(INDIC_EN_MODEL_PATH, "tokenizer_tgt.json")
)

indic_en_meta = json.load(
    open(
        os.path.join(INDIC_EN_MODEL_PATH, "tokenizer_meta.json"),
        encoding="utf-8"
    )
)

indic_en_gen_config = json.load(
    open(
        os.path.join(INDIC_EN_MODEL_PATH, "generation_config.json"),
        encoding="utf-8"
    )
)

indic_en_encoder = ort.InferenceSession(
    os.path.join(INDIC_EN_MODEL_PATH, "encoder_model.onnx"),
    providers=["CPUExecutionProvider"]
)

indic_en_decoder = ort.InferenceSession(
    os.path.join(INDIC_EN_MODEL_PATH, "decoder_model.onnx"),
    providers=["CPUExecutionProvider"]
)

indic_en_decoder_with_past = ort.InferenceSession(
    os.path.join(INDIC_EN_MODEL_PATH, "decoder_with_past_model.onnx"),
    providers=["CPUExecutionProvider"]
)

indic_en_decoder_start_id = int(
    indic_en_gen_config["decoder_start_token_id"]
)

indic_en_eos_id = int(
    indic_en_gen_config["eos_token_id"]
)

indic_en_num_layers = (
    len(indic_en_decoder.get_outputs()) - 1
) // 4


languages = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
    "Odia": "ory_Orya",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Punjabi": "pan_Guru",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Assamese": "asm_Beng",
    "Nepali": "npi_Deva"
}


script_codes = {
    "Hindi": "hi",
    "Bengali": "bn",
    "Odia": "or",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Assamese": "as",
    "Nepali": "ne"
}


app = FastAPI(
    title="VernacAI",
    description="AI-Powered Vernacular Translation Demo API",
    version="4.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str


def past_feed(past_outputs, num_layers):
    feed = {}

    for i in range(num_layers):
        base = i * 4

        feed[
            f"past_key_values.{i}.decoder.key"
        ] = past_outputs[base]

        feed[
            f"past_key_values.{i}.decoder.value"
        ] = past_outputs[base + 1]

        feed[
            f"past_key_values.{i}.encoder.key"
        ] = past_outputs[base + 2]

        feed[
            f"past_key_values.{i}.encoder.value"
        ] = past_outputs[base + 3]

    return feed


def translate_english_to_indic(
    text,
    src_lang,
    tgt_lang
):
    source_text = (
        src_lang
        + " "
        + tgt_lang
        + " "
        + text
    )

    tokens = tokenizer._src_tokenize(
        source_text
    )

    input_ids = [
        tokenizer.src_encoder.get(
            token,
            tokenizer.unk_token_id
        )
        for token in tokens
    ]

    input_ids = np.array(
        [input_ids],
        dtype=np.int64
    )

    attention_mask = np.ones(
        input_ids.shape,
        dtype=np.int64
    )

    encoder_output = encoder.run(
        ["last_hidden_state"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    )[0]

    decoder_start_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    decoder_input_ids = np.array(
        [[decoder_start_id]],
        dtype=np.int64
    )

    output_ids = [
        decoder_start_id
    ]

    past_outputs = None

    for step in range(128):

        if step == 0:
            decoder_output = decoder.run(
                None,
                {
                    "input_ids": decoder_input_ids,
                    "encoder_hidden_states": encoder_output,
                    "encoder_attention_mask": attention_mask
                }
            )

        else:
            decoder_output = decoder_with_past.run(
                None,
                {
                    "input_ids": decoder_input_ids,
                    "encoder_attention_mask": attention_mask,
                    **past_feed(
                        past_outputs,
                        num_layers
                    )
                }
            )

        logits = decoder_output[0]

        past_outputs = list(
            decoder_output[1:]
        )

        next_id = int(
            np.argmax(
                logits[0, -1, :]
            )
        )

        output_ids.append(next_id)

        if next_id == eos_id:
            break

        decoder_input_ids = np.array(
            [[next_id]],
            dtype=np.int64
        )

    safe_ids = [
        i if i < 122672
        else tokenizer.unk_token_id
        for i in output_ids
    ]

    return tokenizer._decode(
        safe_ids,
        skip_special_tokens=True
    )


def translate_indic_to_indic(
    text,
    src_lang,
    tgt_lang
):
    source_text = (
        src_lang
        + " "
        + tgt_lang
        + " "
        + text
    )

    encoded = indic_src_tokenizer.encode(
        source_text
    )

    input_ids = [
        i
        if i < indic_meta["src_dict_size"]
        else indic_meta["unk_id"]
        for i in encoded.ids
    ]

    input_ids = np.array(
        [input_ids],
        dtype=np.int64
    )

    attention_mask = np.array(
        [encoded.attention_mask],
        dtype=np.int64
    )

    encoder_output = indic_encoder.run(
        ["last_hidden_state"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    )[0]

    decoder_input_ids = np.array(
        [[indic_decoder_start_id]],
        dtype=np.int64
    )

    output_ids = [
        indic_decoder_start_id
    ]

    past_outputs = None

    for step in range(128):

        if step == 0:
            decoder_output = indic_decoder.run(
                None,
                {
                    "input_ids": decoder_input_ids,
                    "encoder_hidden_states": encoder_output,
                    "encoder_attention_mask": attention_mask
                }
            )

        else:
            decoder_output = indic_decoder_with_past.run(
                None,
                {
                    "input_ids": decoder_input_ids,
                    "encoder_attention_mask": attention_mask,
                    **past_feed(
                        past_outputs,
                        indic_num_layers
                    )
                }
            )

        logits = decoder_output[0]

        past_outputs = list(
            decoder_output[1:]
        )

        next_id = int(
            np.argmax(
                logits[0, -1, :]
            )
        )

        output_ids.append(next_id)

        if next_id == indic_eos_id:
            break

        decoder_input_ids = np.array(
            [[next_id]],
            dtype=np.int64
        )

    safe_ids = [
        i
        if i < indic_meta["tgt_dict_size"]
        else indic_meta["unk_id"]
        for i in output_ids
    ]

    return indic_tgt_tokenizer.decode(
        safe_ids,
        skip_special_tokens=True
    )


def translate_indic_to_english(
    text,
    src_lang
):
    tgt_lang = "eng_Latn"

    source_text = (
        src_lang
        + " "
        + tgt_lang
        + " "
        + text
    )

    encoded = indic_en_src_tokenizer.encode(
        source_text
    )

    input_ids = [
        i
        if i < indic_en_meta["src_dict_size"]
        else indic_en_meta["unk_id"]
        for i in encoded.ids
    ]

    input_ids = np.array(
        [input_ids],
        dtype=np.int64
    )

    attention_mask = np.array(
        [encoded.attention_mask],
        dtype=np.int64
    )

    encoder_output = indic_en_encoder.run(
        ["last_hidden_state"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    )[0]

    decoder_input_ids = np.array(
        [[indic_en_decoder_start_id]],
        dtype=np.int64
    )

    output_ids = [
        indic_en_decoder_start_id
    ]

    past_outputs = None

    for step in range(128):

        if step == 0:
            decoder_output = indic_en_decoder.run(
                None,
                {
                    "input_ids": decoder_input_ids,
                    "encoder_hidden_states": encoder_output,
                    "encoder_attention_mask": attention_mask
                }
            )

        else:
            decoder_output = indic_en_decoder_with_past.run(
                None,
                {
                    "input_ids": decoder_input_ids,
                    "encoder_attention_mask": attention_mask,
                    **past_feed(
                        past_outputs,
                        indic_en_num_layers
                    )
                }
            )

        logits = decoder_output[0]

        past_outputs = list(
            decoder_output[1:]
        )

        next_id = int(
            np.argmax(
                logits[0, -1, :]
            )
        )

        output_ids.append(next_id)

        if next_id == indic_en_eos_id:
            break

        decoder_input_ids = np.array(
            [[next_id]],
            dtype=np.int64
        )

    safe_ids = [
        i
        if i < indic_en_meta["tgt_dict_size"]
        else indic_en_meta["unk_id"]
        for i in output_ids
    ]

    return indic_en_tgt_tokenizer.decode(
        safe_ids,
        skip_special_tokens=True
    )


def get_script_code(language):
    return script_codes.get(language)


def convert_to_native_script(
    text,
    target_language
):
    target_script = get_script_code(
        target_language
    )

    if not target_script:
        return text

    return UnicodeIndicTransliterator.transliterate(
        text,
        "hi",
        target_script
    )


def convert_source_to_unified_script(
    text,
    source_language
):
    source_script = get_script_code(
        source_language
    )

    if not source_script:
        return text

    if source_script == "hi":
        return text

    return UnicodeIndicTransliterator.transliterate(
        text,
        source_script,
        "hi"
    )


def translate_text(
    text,
    source_language,
    target_language,
    src_lang,
    tgt_lang
):

    if source_language == "English":

        if target_language == "English":
            return text

        result = translate_english_to_indic(
            text,
            src_lang,
            tgt_lang
        )

        return convert_to_native_script(
            result,
            target_language
        )

    unified_text = convert_source_to_unified_script(
        text,
        source_language
    )

    if target_language == "English":

        return translate_indic_to_english(
            unified_text,
            src_lang
        )

    result = translate_indic_to_indic(
        unified_text,
        src_lang,
        tgt_lang
    )

    return convert_to_native_script(
        result,
        target_language
    )


@app.get("/")
def home():
    return {
        "message": "VernacAI API is running",
        "model": "IndicTrans2 ONNX INT8",
        "version": "4.0"
    }


@app.get("/languages")
def get_languages():
    return languages


@app.post("/translate")
def translate(
    request: TranslationRequest
):

    if request.source_language not in languages:
        return {
            "error": "Source language not supported"
        }

    if request.target_language not in languages:
        return {
            "error": "Target language not supported"
        }

    if not request.text.strip():
        return {
            "error": "Text cannot be empty"
        }

    if (
        request.source_language
        == request.target_language
    ):
        return {
            "source_language": request.source_language,
            "target_language": request.target_language,
            "original_text": request.text,
            "translated_text": request.text
        }

    src_lang = languages[
        request.source_language
    ]

    tgt_lang = languages[
        request.target_language
    ]

    try:

        result = translate_text(
            request.text,
            request.source_language,
            request.target_language,
            src_lang,
            tgt_lang
        )

        return {
            "source_language": request.source_language,
            "target_language": request.target_language,
            "original_text": request.text,
            "translated_text": result
        }

    except Exception as e:

        return {
            "error": "Translation failed",
            "details": str(e)
        }