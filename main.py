from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI(
    title="VernacAI",
    description="AI-Powered Vernacular Translation Demo API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

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


class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str


@app.get("/")
def home():
    return {
        "message": "VernacAI API is running",
        "model": "facebook/nllb-200-distilled-600M"
    }


@app.get("/languages")
def get_languages():
    return languages


@app.post("/translate")
def translate(request: TranslationRequest):

    if request.source_language not in languages:
        return {
            "error": "Source language not supported"
        }

    if request.target_language not in languages:
        return {
            "error": "Target language not supported"
        }

    if request.source_language == request.target_language:
        return {
            "source_language": request.source_language,
            "target_language": request.target_language,
            "original_text": request.text,
            "translated_text": request.text
        }

    source_code = languages[request.source_language]
    target_code = languages[request.target_language]

    try:
        tokenizer.src_lang = source_code

        inputs = tokenizer(
            request.text,
            return_tensors="pt"
        )

        output = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),
            max_length=128,
            num_beams=5
        )

        result = tokenizer.batch_decode(
            output,
            skip_special_tokens=True
        )[0]

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