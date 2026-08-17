import os, time, json, re, io, math, concurrent.futures
import pymysql
import pymysql.cursors
from pymysql.err import IntegrityError
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from google import genai
    from google.genai import types
    GEMINI_SDK_AVAILABLE = True
except Exception:
    GEMINI_SDK_AVAILABLE = False
import numpy as np
from PIL import Image
import tensorflow as tf
import requests

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except Exception:
    TRANSLATOR_AVAILABLE = False

try:
    from langdetect import detect_langs as _langdetect_detect_langs, DetectorFactory as _LangDetectFactory
    _LangDetectFactory.seed = 0  # deterministic results
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False

def detect_message_language(text, fallback="en"):
    """Detect what language a free-typed chat message is in, so the assistant
    can reply in that same language ('any language' chat search/output), even
    if it differs from the language currently selected in the site's dropdown.
    Falls back to the site language whenever detection is short, ambiguous, or
    low-confidence (e.g. short romanized/mixed-script messages like "plant ko
    kaise bachaye" are very easy to misdetect as an unrelated language such as
    Estonian or French) — in that case we'd rather keep replying in the site's
    selected language than guess wrong and reply in the wrong language."""
    text = (text or "").strip()
    if not text or not LANGDETECT_AVAILABLE or len(text) < 15:
        return fallback
    try:
        candidates = _langdetect_detect_langs(text)
        if not candidates:
            return fallback
        top = candidates[0]
        code, confidence = top.lang, top.prob
    except Exception:
        return fallback
    if confidence < 0.90:
        return fallback
    # langdetect uses ISO 639-1 codes; a couple differ from our LANG_MAP keys.
    code = {"zh-cn": "zh", "zh-tw": "zh", "iw": "he"}.get(code, code)
    return code if code in LANG_MAP else fallback

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "smart-crop-ai-change-this-secret")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "model.tflite")

# MySQL connection settings (set these in .env / Render environment variables).
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "smart_crop_ai")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg","jpeg","png","webp"}
SUPPORTED_PLANTS = {
    "Apple": "Apple", "Corn_(maize)": "Corn", "Grape": "Grape",
    "Peach": "Peach", "Pepper,_bell": "Bell Pepper", "Potato": "Potato",
    "Strawberry": "Strawberry", "Tomato": "Tomato"
}
# Strict rejection: a prediction is accepted only if it is confident AND
# stable across several image views. This prevents forcing every image into a class.
MIN_CONFIDENCE = 0.72
MIN_MARGIN = 0.12
AUGMENT_VIEWS = 3

LANGUAGES = [
("en","English"),("hi","Hindi"),("mr","Marathi"),("bn","Bengali"),("gu","Gujarati"),("ta","Tamil"),
("te","Telugu"),("kn","Kannada"),("ml","Malayalam"),("pa","Punjabi"),("ur","Urdu"),("or","Odia"),
("as","Assamese"),("ne","Nepali"),("si","Sinhala"),("sa","Sanskrit"),("ar","Arabic"),("fa","Persian"),
("he","Hebrew"),("tr","Turkish"),("az","Azerbaijani"),("hy","Armenian"),("ka","Georgian"),("el","Greek"),
("ru","Russian"),("uk","Ukrainian"),("bg","Bulgarian"),("sr","Serbian"),("hr","Croatian"),("bs","Bosnian"),
("sl","Slovenian"),("sk","Slovak"),("cs","Czech"),("pl","Polish"),("ro","Romanian"),("hu","Hungarian"),
("de","German"),("nl","Dutch"),("da","Danish"),("sv","Swedish"),("no","Norwegian"),("fi","Finnish"),
("is","Icelandic"),("et","Estonian"),("lv","Latvian"),("lt","Lithuanian"),("ga","Irish"),("cy","Welsh"),
("fr","French"),("es","Spanish"),("it","Italian"),("pt","Portuguese"),("ca","Catalan"),("eu","Basque"),
("gl","Galician"),("sw","Swahili"),("am","Amharic"),("so","Somali"),("zu","Zulu"),("xh","Xhosa"),
("af","Afrikaans"),("yo","Yoruba"),("ig","Igbo"),("ha","Hausa"),("rw","Kinyarwanda"),("ny","Chichewa"),
("mg","Malagasy"),("sn","Shona"),("st","Sesotho"),("tn","Tswana"),("wo","Wolof"),("ko","Korean"),
("ja","Japanese"),("zh-CN","Chinese Simplified"),("zh-TW","Chinese Traditional"),("vi","Vietnamese"),
("th","Thai"),("id","Indonesian"),("ms","Malay"),("tl","Filipino"),("km","Khmer"),("lo","Lao"),
("my","Burmese"),("mn","Mongolian"),("ne","Nepali"),("jv","Javanese"),("su","Sundanese"),
("ceb","Cebuano"),("haw","Hawaiian"),("mi","Maori"),("sm","Samoan"),("to","Tongan"),("fj","Fijian"),
("ht","Haitian Creole"),("la","Latin"),("eo","Esperanto"),("lb","Luxembourgish"),("mt","Maltese"),
("sq","Albanian"),("mk","Macedonian"),("be","Belarusian"),("kk","Kazakh"),("uz","Uzbek"),("tg","Tajik"),
("ky","Kyrgyz"),("tk","Turkmen"),("tt","Tatar"),("ps","Pashto"),("ku","Kurdish"),("sd","Sindhi"),
("dv","Dhivehi"),("bo","Tibetan"),("ug","Uyghur"),("yi","Yiddish"),("fy","Frisian"),("co","Corsican"),
("la","Latin"),("br","Breton"),("gd","Scots Gaelic"),("jv","Javanese"),("su","Sundanese")
]
LANG_MAP = dict(LANGUAGES)

PLANTNET_API_KEY = os.environ.get("PLANTNET_API_KEY", "").strip()
PLANTNET_PROJECT = os.environ.get("PLANTNET_PROJECT", "all").strip() or "all"
PLANTNET_MIN_CONFIDENCE = float(os.environ.get("PLANTNET_MIN_CONFIDENCE", "0.45"))
PLANTNET_ENABLED = bool(PLANTNET_API_KEY)
PLANTNET_URL = "https://my-api.plantnet.org/v2/identify"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
# Hybrid vision settings: local TFLite first, Gemini Vision for unsupported/uncertain images.
GEMINI_VERIFY_LOCAL = os.environ.get("GEMINI_VERIFY_LOCAL", "false").strip().lower() in {"1", "true", "yes", "on"}
GEMINI_ENABLED = bool(GEMINI_API_KEY and GEMINI_SDK_AVAILABLE)
gemini_client = None
if GEMINI_ENABLED:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print("Gemini client error:", e)
        GEMINI_ENABLED = False

def _gemini_text(prompt, image_path=None):
    if not GEMINI_ENABLED or gemini_client is None:
        raise RuntimeError("Gemini API is not configured. Set GEMINI_API_KEY in .env.")
    contents = [prompt]
    if image_path:
        mime = "image/jpeg"
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".png":
            mime = "image/png"
        elif ext == ".webp":
            mime = "image/webp"
        with open(image_path, "rb") as f:
            image_part = types.Part.from_bytes(data=f.read(), mime_type=mime)
        contents = [image_part, prompt]
    response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=contents)
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text.strip()

def gemini_detect(path, lang):
    language_name = LANG_MAP.get(lang, "English")
    prompt = f"""
You are the AI fallback plant-disease expert for Smart Crop AI.
Analyze the attached plant/leaf image. The local PlantVillage model may not support this plant.
Return ONLY valid JSON with these keys:
plant, disease, confidence, explanation, home_remedies, natural, field, chemical, prevention.
confidence must be a number from 0 to 100.
Use "{language_name}" for all text values.
If the image is not a plant/leaf or cannot be assessed reliably, set plant and disease to "Unknown" and confidence below 30.
Identify the plant at the broadest reliable level (common name, and scientific name inside explanation when useful).
Disease may be "Healthy", "Unknown", or a likely disease; do not invent a precise disease when the visual evidence is weak.
Give practical farmer-safe guidance.
For chemical advice, do not give dangerous mixing instructions; recommend following a locally approved product label or agricultural expert.
"""
    text = _gemini_text(prompt, path)
    # Remove common markdown JSON fences if Gemini adds them.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        data = json.loads(cleaned)
    except Exception:
        m = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not m:
            raise RuntimeError("Gemini response was not valid JSON.")
        data = json.loads(m.group(0))
    data.setdefault("plant", "Unknown")
    data.setdefault("disease", "Unknown")
    data.setdefault("confidence", 0)
    data.setdefault("explanation", "")
    for key in ("home_remedies", "natural", "field", "chemical", "prevention"):
        value = data.get(key, [])
        data[key] = value if isinstance(value, list) else [str(value)]
    data["confidence"] = float(data.get("confidence") or 0)
    data["unknown"] = str(data["plant"]).lower() == "unknown" or data["confidence"] < 30
    return data

def gemini_chat(message, lang, history=None):
    language_name = LANG_MAP.get(lang, "English")
    history = history or []
    compact = []
    for item in history[-8:]:
        if isinstance(item, dict):
            role = "User" if item.get("role") == "user" else "Assistant"
            content = str(item.get("content", ""))[:1200]
            compact.append(f"{role}: {content}")
    history_text = "\n".join(compact)
    current_plant = session.get("last_plant", "Unknown")
    current_disease = session.get("last_class", "Unknown")
    current_conf = session.get("last_confidence", "")
    context = f"Current detection context: plant={current_plant}; disease={current_disease}; confidence={current_conf}."
    prompt = f"""
You are Smart Crop AI Farmer Assistant. Reply in {language_name}.
{context}
Help with the current detected plant/disease when relevant, as well as plant identification, common crop diseases, basic prevention, irrigation,
field hygiene and safe next steps. Be concise and practical for Indian farmers.
Do not claim certainty from symptoms alone. Do not prescribe unsafe pesticide mixing.
If chemical control is discussed, advise using a locally approved product exactly as
its label says and consulting a local agriculture expert when needed.
Conversation:
{history_text}

User: {message}
Assistant:
"""
    return _gemini_text(prompt)

TEXTS = {
"en":{"title":"Smart Crop AI","brand":"Smart Crop AI","home":"Home","about":"About","feedback":"Farmer Feedback",
"login":"Login","dashboard":"Dashboard","logout":"Logout","camera":"Live Camera","upload_title":"AI Plant Disease Detection",
"hero_title":"Smart Plant Disease Detection for Farmers","hero_text":"Upload a leaf photo or use your camera. The AI rejects uncertain images instead of guessing.",
"upload":"Choose Image","detect":"Detect","location":"Use My Location","weather":"Weather & Environment","result":"Detection Result",
"uploaded":"Uploaded Image","plant":"Plant","disease":"Disease","confidence":"Confidence","status":"Status","precautions":"Farmer Guidance",
"home_remedies":"Home / Basic Care","natural":"Natural Management","field":"Field Management","chemical":"Chemical Control","prevention":"Prevention",
"unknown":"Plant Not Identified","unknown_text":"The image is not confidently recognized as one of the supported PlantVillage classes. Please use a clear leaf photo.",
"feedback_title":"Farmer Feedback","name":"Name","message":"Message","send":"Send Feedback","about_title":"About Smart Crop AI",
"dashboard_title":"Farmer Dashboard","history":"Recent Scans","no_history":"No scans yet.","email":"Email","password":"Password","register":"Create Account",
"login_title":"Farmer Login","register_title":"Farmer Registration","new_user":"New farmer? Register","existing_user":"Already registered? Login",
"camera_help":"Allow camera access, center one leaf, then capture.","capture":"Capture","retake":"Retake","use_photo":"Use Photo",
"location_help":"Location is used only to fetch local weather. You can also enter a city manually.","city":"City","get_weather":"Get Weather",
"not_available":"Not available","weather_error":"Could not fetch weather. Check internet/location.","safe_note":"AI result is a screening aid; confirm important decisions with a local agricultural expert.",
"language":"Language","welcome":"Welcome","email_exists":"Email already registered.","invalid_login":"Invalid email or password.","feedback_saved":"Thank you for your feedback.",
"contact_title":"Contact Us","contact_text":"Have a question or need help? Send us a message and our team will get back to you.","send_message":"Send Message","contact_saved":"Thank you, we received your message.",
"suggestion_title":"Suggestions","suggestion_text":"Share ideas to help us improve Smart Crop AI for farmers like you.","suggestion_placeholder":"Write your suggestion...","send_suggestion":"Send Suggestion","suggestion_saved":"Thank you for your suggestion.","recent_suggestions":"Recent Suggestions","no_suggestions":"No suggestions yet. Be the first!",
"rating":"Rating","your_rating":"Your Rating","recent_feedback":"Recent Farmer Feedback","avg_rating":"Average Rating","no_feedback":"No feedback yet."},
"hi":{"title":"स्मार्ट क्रॉप AI","brand":"स्मार्ट क्रॉप AI","home":"होम","about":"हमारे बारे में","feedback":"किसान प्रतिक्रिया","login":"लॉगिन","dashboard":"डैशबोर्ड","logout":"लॉगआउट","camera":"लाइव कैमरा",
"upload_title":"AI पौधा रोग पहचान","hero_title":"किसानों के लिए स्मार्ट पौधा रोग पहचान","hero_text":"पत्ती की फोटो अपलोड करें या कैमरा उपयोग करें। AI अनिश्चित फोटो पर जबरदस्ती अनुमान नहीं लगाएगा।",
"upload":"फोटो चुनें","detect":"पहचानें","location":"मेरी लोकेशन","weather":"मौसम और वातावरण","result":"पहचान परिणाम","uploaded":"अपलोड फोटो","plant":"पौधा","disease":"रोग","confidence":"विश्वास स्तर","status":"स्थिति","precautions":"किसान मार्गदर्शन","home_remedies":"घरेलू/बुनियादी देखभाल","natural":"प्राकृतिक प्रबंधन","field":"खेत प्रबंधन","chemical":"रासायनिक नियंत्रण","prevention":"रोकथाम",
"unknown":"पौधा पहचाना नहीं गया","unknown_text":"फोटो समर्थित PlantVillage वर्गों में पर्याप्त विश्वास के साथ नहीं पहचानी गई। साफ पत्ती की फोटो लें।","feedback_title":"किसान प्रतिक्रिया","name":"नाम","message":"संदेश","send":"प्रतिक्रिया भेजें","about_title":"स्मार्ट क्रॉप AI के बारे में",
"dashboard_title":"किसान डैशबोर्ड","history":"हाल की स्कैन","no_history":"अभी कोई स्कैन नहीं है।","email":"ईमेल","password":"पासवर्ड","register":"खाता बनाएं","login_title":"किसान लॉगिन","register_title":"किसान पंजीकरण","new_user":"नए किसान? पंजीकरण करें","existing_user":"पहले से खाता है? लॉगिन करें",
"camera_help":"कैमरा अनुमति दें, एक पत्ती को बीच में रखें और फोटो लें।","capture":"फोटो लें","retake":"दोबारा लें","use_photo":"फोटो उपयोग करें","location_help":"लोकेशन का उपयोग केवल स्थानीय मौसम के लिए होता है। शहर भी डाल सकते हैं।","city":"शहर","get_weather":"मौसम देखें","not_available":"उपलब्ध नहीं","weather_error":"मौसम नहीं मिल सका। इंटरनेट/लोकेशन जांचें।","safe_note":"AI परिणाम केवल स्क्रीनिंग सहायता है; महत्वपूर्ण निर्णय के लिए स्थानीय कृषि विशेषज्ञ से पुष्टि करें।","language":"भाषा","welcome":"स्वागत है","email_exists":"यह ईमेल पहले से पंजीकृत है।","invalid_login":"ईमेल या पासवर्ड गलत है।","feedback_saved":"आपकी प्रतिक्रिया के लिए धन्यवाद।",
"contact_title":"संपर्क करें","contact_text":"कोई सवाल है या मदद चाहिए? हमें संदेश भेजें, हमारी टीम जल्द जवाब देगी।","send_message":"संदेश भेजें","contact_saved":"धन्यवाद, आपका संदेश मिल गया है।",
"suggestion_title":"सुझाव","suggestion_text":"स्मार्ट क्रॉप AI को बेहतर बनाने के लिए अपने विचार साझा करें।","suggestion_placeholder":"अपना सुझाव लिखें...","send_suggestion":"सुझाव भेजें","suggestion_saved":"आपके सुझाव के लिए धन्यवाद।","recent_suggestions":"हाल के सुझाव","no_suggestions":"अभी कोई सुझाव नहीं है। सबसे पहले आप बनें!",
"rating":"रेटिंग","your_rating":"आपकी रेटिंग","recent_feedback":"हाल की किसान प्रतिक्रिया","avg_rating":"औसत रेटिंग","no_feedback":"अभी कोई प्रतिक्रिया नहीं है।"},
"mr":{"title":"स्मार्ट क्रॉप AI","brand":"स्मार्ट क्रॉप AI","home":"मुख्यपृष्ठ","about":"आमच्याबद्दल","feedback":"शेतकरी अभिप्राय","login":"लॉगिन","dashboard":"डॅशबोर्ड","logout":"लॉगआउट","camera":"लाइव्ह कॅमेरा",
"upload_title":"AI वनस्पती रोग ओळख","hero_title":"शेतकऱ्यांसाठी स्मार्ट वनस्पती रोग ओळख","hero_text":"पानाचा फोटो अपलोड करा किंवा कॅमेरा वापरा. AI अनिश्चित फोटोवर जबरदस्ती अंदाज लावणार नाही.",
"upload":"फोटो निवडा","detect":"ओळखा","location":"माझे स्थान","weather":"हवामान आणि पर्यावरण","result":"ओळख परिणाम","uploaded":"अपलोड फोटो","plant":"वनस्पती","disease":"रोग","confidence":"विश्वास पातळी","status":"स्थिती","precautions":"शेतकरी मार्गदर्शन","home_remedies":"घरगुती/मूलभूत काळजी","natural":"नैसर्गिक व्यवस्थापन","field":"शेत व्यवस्थापन","chemical":"रासायनिक नियंत्रण","prevention":"प्रतिबंध",
"unknown":"वनस्पती ओळखली नाही","unknown_text":"फोटो समर्थित PlantVillage वर्गांमध्ये पुरेशा विश्वासाने ओळखला गेला नाही. स्वच्छ पानाचा फोटो घ्या.","feedback_title":"शेतकरी अभिप्राय","name":"नाव","message":"संदेश","send":"अभिप्राय पाठवा","about_title":"स्मार्ट क्रॉप AI बद्दल",
"dashboard_title":"शेतकरी डॅशबोर्ड","history":"अलीकडील स्कॅन","no_history":"अजून स्कॅन नाही.","email":"ईमेल","password":"पासवर्ड","register":"खाते तयार करा","login_title":"शेतकरी लॉगिन","register_title":"शेतकरी नोंदणी","new_user":"नवीन शेतकरी? नोंदणी करा","existing_user":"खाते आहे? लॉगिन करा",
"camera_help":"कॅमेरा परवानगी द्या, एक पान मध्यभागी ठेवा आणि फोटो घ्या.","capture":"फोटो घ्या","retake":"पुन्हा घ्या","use_photo":"फोटो वापरा","location_help":"स्थानाचा वापर फक्त स्थानिक हवामानासाठी केला जातो. शहर टाकू शकता.","city":"शहर","get_weather":"हवामान पहा","not_available":"उपलब्ध नाही","weather_error":"हवामान मिळू शकले नाही. इंटरनेट/स्थान तपासा.","safe_note":"AI निकाल फक्त स्क्रीनिंगसाठी आहे; महत्त्वाच्या निर्णयासाठी स्थानिक कृषी तज्ज्ञांचा सल्ला घ्या.","language":"भाषा","welcome":"स्वागत","email_exists":"हा ईमेल आधीच नोंदणीकृत आहे.","invalid_login":"ईमेल किंवा पासवर्ड चुकीचा आहे.","feedback_saved":"अभिप्रायाबद्दल धन्यवाद.",
"contact_title":"संपर्क करा","contact_text":"काही प्रश्न आहे किंवा मदत हवी आहे? आम्हाला संदेश पाठवा, आमची टीम लवकरच उत्तर देईल.","send_message":"संदेश पाठवा","contact_saved":"धन्यवाद, तुमचा संदेश मिळाला आहे.",
"suggestion_title":"सूचना","suggestion_text":"स्मार्ट क्रॉप AI अधिक चांगले करण्यासाठी तुमच्या कल्पना शेअर करा.","suggestion_placeholder":"तुमची सूचना लिहा...","send_suggestion":"सूचना पाठवा","suggestion_saved":"तुमच्या सूचनेबद्दल धन्यवाद.","recent_suggestions":"अलीकडील सूचना","no_suggestions":"अजून सूचना नाहीत. सर्वप्रथम तुम्ही व्हा!",
"rating":"रेटिंग","your_rating":"तुमची रेटिंग","recent_feedback":"अलीकडील शेतकरी अभिप्राय","avg_rating":"सरासरी रेटिंग","no_feedback":"अजून अभिप्राय नाही."}
}

DISEASE_DATA = {'Apple___Apple_scab': {'en': {'name': 'Apple Scab', 'home': ['Remove and destroy infected leaves.', 'Keep the area around trees clean.', 'Remove fallen infected leaves.'], 'natural': ['Improve air circulation by pruning.', 'Avoid excessive moisture on leaves.', 'Use healthy planting material.'], 'field': ['Prune dense branches.', 'Remove infected plant debris.', 'Maintain proper plant spacing.'], 'chemical': ['Use a suitable fungicide according to local agricultural recommendations.', 'Follow the product label carefully.'], 'prevention': ['Use resistant varieties where available.', 'Maintain orchard sanitation.', 'Avoid prolonged leaf wetness.']}, 'hi': {'name': 'सेब का स्कैब रोग', 'home': ['संक्रमित पत्तियों को हटाकर नष्ट करें।', 'पेड़ के आसपास की जगह साफ रखें।', 'गिरी हुई संक्रमित पत्तियों को हटाएं।'], 'natural': ['छंटाई करके हवा का संचार बढ़ाएं।', 'पत्तियों पर अत्यधिक नमी से बचें।', 'स्वस्थ पौध सामग्री का उपयोग करें।'], 'field': ['घनी शाखाओं की छंटाई करें।', 'संक्रमित पौध अवशेष हटाएं।', 'पौधों के बीच उचित दूरी रखें।'], 'chemical': ['स्थानीय कृषि सलाह के अनुसार उपयुक्त फफूंदनाशक का उपयोग करें।', 'उत्पाद के लेबल पर दिए निर्देशों का पालन करें।'], 'prevention': ['जहाँ उपलब्ध हो रोग प्रतिरोधी किस्मों का उपयोग करें।', 'बगीचे की स्वच्छता बनाए रखें।', 'पत्तियों पर लंबे समय तक नमी से बचें।']}, 'mr': {'name': 'सफरचंद स्कॅब रोग', 'home': ['संक्रमित पाने काढून नष्ट करा.', 'झाडाच्या आजूबाजूचा परिसर स्वच्छ ठेवा.', 'गळलेली संक्रमित पाने काढून टाका.'], 'natural': ['छाटणी करून हवेचे योग्य वहन ठेवा.', 'पानांवर जास्त ओलावा राहू देऊ नका.', 'निरोगी रोपांचा वापर करा.'], 'field': ['दाट फांद्यांची छाटणी करा.', 'संक्रमित वनस्पती अवशेष काढा.', 'झाडांमध्ये योग्य अंतर ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्यानुसार योग्य बुरशीनाशक वापरा.', 'उत्पादनाच्या लेबलवरील सूचना पाळा.'], 'prevention': ['उपलब्ध असल्यास रोगप्रतिकारक वाण वापरा.', 'बागेची स्वच्छता राखा.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.']}}, 'Apple___Black_rot': {'en': {'name': 'Apple Black Rot', 'home': ['Remove infected fruits and leaves.', 'Remove dead branches.', 'Keep fallen plant material away from the orchard.'], 'natural': ['Improve sunlight and air circulation.', 'Avoid unnecessary leaf wetness.', 'Maintain healthy tree growth.'], 'field': ['Prune dead and infected branches.', 'Remove mummified fruits.', 'Maintain orchard sanitation.'], 'chemical': ['Apply an appropriate fungicide according to local recommendations.', 'Follow the product label.'], 'prevention': ['Remove diseased plant material regularly.', 'Avoid tree injuries.', 'Use healthy planting material.']}, 'hi': {'name': 'सेब का ब्लैक रॉट रोग', 'home': ['संक्रमित फल और पत्तियाँ हटा दें।', 'पेड़ की सूखी शाखाएँ काट दें।', 'गिरी हुई संक्रमित सामग्री हटा दें।'], 'natural': ['धूप और हवा का अच्छा संचार रखें।', 'पत्तियों पर अनावश्यक नमी से बचें।', 'पेड़ को स्वस्थ रखें।'], 'field': ['सूखी और संक्रमित शाखाओं की छंटाई करें।', 'सूखे संक्रमित फलों को हटाएं।', 'बगीचे की सफाई बनाए रखें।'], 'chemical': ['स्थानीय कृषि सलाह के अनुसार उचित फफूंदनाशक का प्रयोग करें।', 'लेबल पर दी गई मात्रा से अधिक उपयोग न करें।'], 'prevention': ['रोगग्रस्त सामग्री नियमित रूप से हटाएं।', 'पेड़ को चोट लगने से बचाएं।', 'स्वस्थ पौध सामग्री का उपयोग करें।']}, 'mr': {'name': 'सफरचंद ब्लॅक रॉट रोग', 'home': ['संक्रमित फळे आणि पाने काढा.', 'झाडाच्या वाळलेल्या फांद्या काढा.', 'संक्रमित वनस्पती अवशेष दूर ठेवा.'], 'natural': ['सूर्यप्रकाश आणि हवेचे योग्य वहन ठेवा.', 'पानांवर अनावश्यक ओलावा टाळा.', 'झाडाची वाढ निरोगी ठेवा.'], 'field': ['वाळलेल्या आणि संक्रमित फांद्यांची छाटणी करा.', 'वाळलेली संक्रमित फळे काढा.', 'बागेची स्वच्छता राखा.'], 'chemical': ['स्थानिक कृषी सल्ल्यानुसार योग्य बुरशीनाशक वापरा.', 'लेबलवरील सूचनांचे पालन करा.'], 'prevention': ['रोगग्रस्त अवशेष नियमित काढा.', 'झाडाला इजा होणार नाही याची काळजी घ्या.', 'निरोगी रोपांचा वापर करा.']}}, 'Apple___Cedar_apple_rust': {'en': {'name': 'Apple Cedar Rust', 'home': ['Remove badly infected leaves.', 'Keep the orchard clean.'], 'natural': ['Improve air circulation.', 'Avoid excessive moisture.'], 'field': ['Remove infected plant material.', 'Maintain adequate spacing.'], 'chemical': ['Use a recommended fungicide when necessary.', 'Follow the product label.'], 'prevention': ['Use resistant varieties where available.', 'Maintain orchard sanitation.']}, 'hi': {'name': 'सेब सीडर रस्ट', 'home': ['बहुत अधिक संक्रमित पत्तियों को हटा दें।', 'बगीचे को साफ रखें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'अत्यधिक नमी से बचें।'], 'field': ['संक्रमित पौध सामग्री हटाएं।', 'पौधों के बीच पर्याप्त दूरी रखें।'], 'chemical': ['आवश्यकता होने पर अनुशंसित फफूंदनाशक का उपयोग करें।', 'उत्पाद के लेबल का पालन करें।'], 'prevention': ['जहाँ उपलब्ध हो प्रतिरोधी किस्मों का उपयोग करें।', 'बगीचे की स्वच्छता बनाए रखें।']}, 'mr': {'name': 'सफरचंद सीडर रस्ट', 'home': ['जास्त संक्रमित पाने काढून टाका.', 'बाग स्वच्छ ठेवा.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'जास्त ओलावा टाळा.'], 'field': ['संक्रमित वनस्पती अवशेष काढा.', 'झाडांमध्ये योग्य अंतर ठेवा.'], 'chemical': ['गरजेनुसार शिफारस केलेले बुरशीनाशक वापरा.', 'उत्पादनाच्या लेबलवरील सूचना पाळा.'], 'prevention': ['उपलब्ध असल्यास रोगप्रतिकारक वाण वापरा.', 'बागेची स्वच्छता राखा.']}}, 'Apple___healthy': {'en': {'name': 'Healthy Apple Leaf', 'home': ['No disease treatment is required.', 'Continue regular plant care.'], 'natural': ['Provide adequate sunlight.', 'Maintain balanced watering.'], 'field': ['Monitor leaves regularly.', 'Remove weeds around the plant.'], 'chemical': ['Do not use fungicides unnecessarily.'], 'prevention': ['Maintain good sanitation.', 'Inspect plants regularly.']}, 'hi': {'name': 'स्वस्थ सेब की पत्ती', 'home': ['बीमारी के उपचार की आवश्यकता नहीं है।', 'नियमित पौधों की देखभाल जारी रखें।'], 'natural': ['पर्याप्त धूप दें।', 'संतुलित सिंचाई करें।'], 'field': ['पत्तियों की नियमित जांच करें।', 'पौधे के आसपास की खरपतवार हटाएं।'], 'chemical': ['बिना आवश्यकता फफूंदनाशक का उपयोग न करें।'], 'prevention': ['अच्छी स्वच्छता बनाए रखें।', 'पौधों की नियमित जांच करें।']}, 'mr': {'name': 'निरोगी सफरचंद पान', 'home': ['रोगासाठी उपचाराची आवश्यकता नाही.', 'नियमित वनस्पती काळजी सुरू ठेवा.'], 'natural': ['पुरेसा सूर्यप्रकाश द्या.', 'पाण्याचे संतुलित व्यवस्थापन करा.'], 'field': ['पानांची नियमित तपासणी करा.', 'झाडाभोवतीची तण काढा.'], 'chemical': ['गरज नसताना बुरशीनाशक वापरू नका.'], 'prevention': ['चांगली स्वच्छता राखा.', 'वनस्पतींची नियमित तपासणी करा.']}}, 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {'en': {'name': 'Corn Gray Leaf Spot', 'home': ['Remove badly affected leaves and crop debris.', 'Avoid working through wet foliage when possible.', 'Keep the field clean around plants.'], 'natural': ['Improve airflow with proper plant spacing.', 'Use balanced nutrition and avoid excessive nitrogen.', 'Avoid prolonged leaf wetness where possible.'], 'field': ['Rotate corn with non-host crops when practical.', 'Manage crop residue after harvest.', 'Monitor lower leaves regularly for expanding spots.'], 'chemical': ['Use a labeled fungicide only when disease pressure and local recommendations justify it.', 'Follow the product label and local agricultural guidance.'], 'prevention': ['Use tolerant varieties where available.', 'Practice crop rotation and residue management.', 'Scout fields early and regularly.']}, 'hi': {'name': 'Corn Gray Leaf Spot', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Corn Gray Leaf Spot', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Corn_(maize)___Common_rust_': {'en': {'name': 'Corn Common Rust', 'home': ['Remove heavily affected leaves when practical.', 'Keep volunteer plants and weeds under control.', 'Avoid unnecessary overhead watering.'], 'natural': ['Maintain good airflow.', 'Support healthy growth with balanced fertilizer.', 'Avoid prolonged leaf wetness.'], 'field': ['Scout lower and middle leaves for rust pustules.', 'Remove heavily infected debris after harvest.', 'Use crop rotation as part of an integrated plan.'], 'chemical': ['Use a labeled fungicide when justified by local disease pressure.', 'Follow the label and local agricultural advice.'], 'prevention': ['Choose resistant/tolerant hybrids where available.', 'Scout early during favorable weather.', 'Maintain field sanitation.']}, 'hi': {'name': 'Corn Common Rust', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Corn Common Rust', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Corn_(maize)___Northern_Leaf_Blight': {'en': {'name': 'Corn Northern Leaf Blight', 'home': ['Remove severely infected leaves when practical.', 'Keep crop debris managed after harvest.', 'Avoid unnecessary leaf wetness.'], 'natural': ['Maintain balanced nutrition and plant vigor.', 'Improve airflow through appropriate spacing.', 'Avoid repeated overhead irrigation late in the day.'], 'field': ['Rotate crops where practical.', 'Manage infected residue.', 'Scout lower leaves early and often.'], 'chemical': ['Use a labeled fungicide when recommended locally.', 'Follow the product label and pre-harvest requirements.'], 'prevention': ['Plant resistant or tolerant hybrids where available.', 'Use crop rotation and residue management.', 'Monitor during cool, humid periods.']}, 'hi': {'name': 'Corn Northern Leaf Blight', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Corn Northern Leaf Blight', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Corn_(maize)___healthy': {'en': {'name': 'Healthy Corn Leaf', 'home': ['No disease treatment is needed.', 'Remove weeds competing with the crop.', 'Keep the field clean and well drained.'], 'natural': ['Maintain balanced irrigation and nutrition.', 'Provide adequate sunlight and airflow.', 'Monitor plant vigor regularly.'], 'field': ['Scout leaves and stems regularly.', 'Maintain sensible plant spacing.', 'Manage weeds and crop residue.'], 'chemical': ['Avoid routine fungicide use when there is no disease indication.'], 'prevention': ['Use healthy seed and tolerant hybrids where available.', 'Practice crop rotation.', 'Inspect plants early for changes.']}, 'hi': {'name': 'Healthy Corn Leaf', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Healthy Corn Leaf', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Grape___Black_rot': {'en': {'name': 'Grape Black Rot', 'home': ['Remove and destroy mummified berries and infected leaves.', 'Keep fallen infected fruit away from vines.', 'Improve vineyard sanitation.'], 'natural': ['Prune for sunlight and airflow.', 'Avoid prolonged leaf wetness.', 'Maintain balanced vine nutrition.'], 'field': ['Remove infected clusters and plant debris.', 'Manage weeds to improve airflow.', 'Scout leaves and fruit regularly.'], 'chemical': ['Use a labeled fungicide according to local recommendations and label directions.', 'Observe harvest and re-entry instructions on the label.'], 'prevention': ['Remove mummified fruit promptly.', 'Use clean planting material.', 'Maintain an open canopy.']}, 'hi': {'name': 'Grape Black Rot', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Grape Black Rot', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Grape___Esca_(Black_Measles)': {'en': {'name': 'Grape Esca / Black Measles', 'home': ['Remove severely affected plant parts where appropriate.', 'Disinfect pruning tools between vines.', 'Do not leave infected pruning waste in the vineyard.'], 'natural': ['Maintain balanced vine vigor and avoid unnecessary stress.', 'Improve canopy airflow.', 'Avoid injuries to trunks and pruning wounds.'], 'field': ['Mark symptomatic vines for monitoring.', 'Remove severely diseased wood according to local extension advice.', 'Keep vineyard sanitation high.'], 'chemical': ['There is no simple curative chemical treatment; use only locally recommended products/practices.', 'Follow agricultural extension guidance for trunk-disease management.'], 'prevention': ['Use healthy planting material.', 'Make clean, careful pruning cuts.', 'Avoid unnecessary trunk injuries.']}, 'hi': {'name': 'Grape Esca / Black Measles', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Grape Esca / Black Measles', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {'en': {'name': 'Grape Leaf Blight', 'home': ['Remove heavily infected leaves and debris.', 'Keep fallen leaves away from vine rows.', 'Avoid unnecessary leaf wetness.'], 'natural': ['Open the canopy for better airflow and sunlight.', 'Use balanced irrigation.', 'Avoid dense, humid canopy conditions.'], 'field': ['Prune crowded growth.', 'Scout leaves regularly.', 'Remove infected debris after pruning.'], 'chemical': ['Use an appropriate labeled fungicide only when locally recommended.', 'Follow the product label.'], 'prevention': ['Maintain vineyard sanitation.', 'Use healthy planting material.', 'Manage canopy humidity.']}, 'hi': {'name': 'Grape Leaf Blight', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Grape Leaf Blight', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Grape___healthy': {'en': {'name': 'Healthy Grape Leaf', 'home': ['No disease treatment is needed.', 'Remove weeds and damaged plant material.', 'Keep vines clean and well supported.'], 'natural': ['Maintain good canopy airflow.', 'Water consistently without prolonged leaf wetness.', 'Provide balanced nutrition.'], 'field': ['Prune and train vines properly.', 'Scout leaves and fruit regularly.', 'Keep vineyard floor managed.'], 'chemical': ['Avoid unnecessary pesticide applications.'], 'prevention': ['Use healthy planting material.', 'Maintain sanitation and canopy management.', 'Monitor regularly.']}, 'hi': {'name': 'Healthy Grape Leaf', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Healthy Grape Leaf', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Peach___Bacterial_spot': {'en': {'name': 'Peach Bacterial Spot', 'home': ['Remove severely affected leaves and fruit where practical.', 'Avoid moving through wet foliage.', 'Remove fallen infected debris.'], 'natural': ['Improve airflow through pruning.', 'Avoid overhead irrigation when possible.', 'Maintain balanced tree nutrition.'], 'field': ['Prune crowded branches.', 'Maintain orchard sanitation.', 'Monitor young leaves and fruit after wet weather.'], 'chemical': ['Use only a locally recommended bactericide and follow its label.', 'Do not exceed labeled rates or spray frequency.'], 'prevention': ['Choose tolerant varieties where available.', 'Use clean planting material.', 'Avoid unnecessary tree wounds.']}, 'hi': {'name': 'Peach Bacterial Spot', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Peach Bacterial Spot', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Peach___healthy': {'en': {'name': 'Healthy Peach Leaf', 'home': ['No disease treatment is needed.', 'Remove weeds and dead plant material.', 'Keep the orchard clean.'], 'natural': ['Provide adequate sunlight and airflow.', 'Maintain balanced watering.', 'Support healthy tree growth with balanced nutrition.'], 'field': ['Prune appropriately and monitor leaves and fruit.', 'Keep orchard floor clean.', 'Scout after wet weather.'], 'chemical': ['Avoid unnecessary pesticide applications.'], 'prevention': ['Use healthy planting material.', 'Maintain orchard sanitation.', 'Inspect trees regularly.']}, 'hi': {'name': 'Healthy Peach Leaf', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Healthy Peach Leaf', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Pepper,_bell___Bacterial_spot': {'en': {'name': 'Bell Pepper Bacterial Spot', 'home': ['Remove badly infected leaves and fruit.', 'Avoid handling plants when foliage is wet.', 'Remove infected plant debris.'], 'natural': ['Use drip irrigation where possible.', 'Improve airflow and avoid overcrowding.', 'Keep leaves dry as practical.'], 'field': ['Rotate away from susceptible crops.', 'Remove volunteer plants and weeds.', 'Sanitize tools and hands after handling infected plants.'], 'chemical': ['Use only locally recommended bactericides and follow the label.', 'Do not mix or exceed products unless the label permits it.'], 'prevention': ['Use certified clean seed/transplants.', 'Choose tolerant varieties where available.', 'Practice crop rotation and sanitation.']}, 'hi': {'name': 'Bell Pepper Bacterial Spot', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Bell Pepper Bacterial Spot', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Pepper,_bell___healthy': {'en': {'name': 'Healthy Bell Pepper Leaf', 'home': ['No disease treatment is needed.', 'Remove weeds and damaged leaves.', 'Keep the growing area clean.'], 'natural': ['Use balanced irrigation.', 'Provide good airflow and sunlight.', 'Avoid prolonged leaf wetness.'], 'field': ['Scout leaves and fruit regularly.', 'Maintain sensible plant spacing.', 'Manage weeds around plants.'], 'chemical': ['Avoid unnecessary pesticides.'], 'prevention': ['Use healthy transplants.', 'Maintain field sanitation.', 'Rotate crops where practical.']}, 'hi': {'name': 'Healthy Bell Pepper Leaf', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Healthy Bell Pepper Leaf', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Potato___Early_blight': {'en': {'name': 'Potato Early Blight', 'home': ['Remove badly affected lower leaves when practical.', 'Remove infected crop debris.', 'Avoid unnecessary leaf wetness.'], 'natural': ['Maintain balanced fertilizer, especially adequate potassium.', 'Use mulch or practices that reduce soil splash where appropriate.', 'Avoid plant stress from irregular watering.'], 'field': ['Rotate crops.', 'Remove volunteer potatoes.', 'Scout older leaves first.'], 'chemical': ['Use a labeled fungicide only when locally recommended.', 'Rotate fungicide modes of action according to the label and local advice.'], 'prevention': ['Use healthy seed tubers.', 'Practice crop rotation.', 'Maintain good field sanitation.']}, 'hi': {'name': 'Potato Early Blight', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Potato Early Blight', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Potato___Late_blight': {'en': {'name': 'Potato Late Blight', 'home': ['Remove and destroy severely infected foliage when advised.', 'Do not leave infected tubers or debris in the field.', 'Avoid handling plants when wet.'], 'natural': ['Improve airflow and avoid prolonged leaf wetness.', 'Use well-managed irrigation.', 'Monitor closely during cool, humid weather.'], 'field': ['Remove cull piles and volunteer potatoes.', 'Scout frequently after favorable weather.', 'Harvest carefully to reduce tuber injury.'], 'chemical': ['Use locally recommended late-blight fungicides when disease risk is high.', 'Follow the label and resistance-management guidance.'], 'prevention': ['Plant certified disease-free seed.', 'Destroy cull piles.', 'Use resistant/tolerant varieties where available.']}, 'hi': {'name': 'Potato Late Blight', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Potato Late Blight', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Potato___healthy': {'en': {'name': 'Healthy Potato Leaf', 'home': ['No disease treatment is needed.', 'Keep weeds controlled.', 'Remove damaged debris from the field.'], 'natural': ['Maintain balanced irrigation and nutrition.', 'Avoid prolonged leaf wetness.', 'Support good airflow between plants.'], 'field': ['Scout leaves and stems regularly.', 'Rotate crops.', 'Use proper hilling and field sanitation.'], 'chemical': ['Avoid routine fungicide use without disease evidence.'], 'prevention': ['Use certified seed tubers.', 'Rotate crops.', 'Monitor regularly.']}, 'hi': {'name': 'Healthy Potato Leaf', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Healthy Potato Leaf', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Strawberry___Leaf_scorch': {'en': {'name': 'Strawberry Leaf Scorch', 'home': ['Remove severely affected leaves.', 'Remove dead plant debris from beds.', 'Avoid prolonged leaf wetness.'], 'natural': ['Improve airflow by avoiding overcrowding.', 'Water at the soil level when possible.', 'Maintain balanced nutrition.'], 'field': ['Remove old infected leaves after harvest as appropriate.', 'Keep beds weed-free.', 'Monitor new growth regularly.'], 'chemical': ['Use a labeled fungicide only if the disease diagnosis and local recommendation support it.', 'Follow the product label.'], 'prevention': ['Use healthy planting material.', 'Avoid overcrowded beds.', 'Maintain sanitation and irrigation management.']}, 'hi': {'name': 'Strawberry Leaf Scorch', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Strawberry Leaf Scorch', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Strawberry___healthy': {'en': {'name': 'Healthy Strawberry Leaf', 'home': ['No disease treatment is needed.', 'Remove damaged leaves and weeds.', 'Keep beds clean.'], 'natural': ['Water at the root zone.', 'Provide sunlight and airflow.', 'Maintain balanced nutrition.'], 'field': ['Scout leaves and fruit regularly.', 'Remove old debris after harvest.', 'Maintain appropriate plant spacing.'], 'chemical': ['Avoid unnecessary pesticides.'], 'prevention': ['Use healthy runners/transplants.', 'Maintain bed sanitation.', 'Monitor regularly.']}, 'hi': {'name': 'Healthy Strawberry Leaf', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Healthy Strawberry Leaf', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Tomato___Bacterial_spot': {'en': {'name': 'Tomato Bacterial Spot', 'home': ['Remove severely affected leaves and fruit.', 'Avoid working with wet plants.', 'Remove infected debris from the growing area.'], 'natural': ['Use drip irrigation where possible.', 'Improve airflow and avoid overcrowding.', 'Avoid leaf wetness lasting overnight.'], 'field': ['Rotate away from susceptible crops.', 'Sanitize tools.', 'Scout new growth and fruit regularly.'], 'chemical': ['Use only locally recommended bactericides and follow the label.', 'Do not exceed label rates.'], 'prevention': ['Use clean seed/transplants.', 'Choose tolerant varieties where available.', 'Practice crop rotation and sanitation.']}, 'hi': {'name': 'Tomato Bacterial Spot', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Tomato Bacterial Spot', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Tomato___Early_blight': {'en': {'name': 'Tomato Early Blight', 'home': ['Remove lower infected leaves.', 'Remove fallen infected debris.', 'Avoid soil splash onto leaves where practical.'], 'natural': ['Mulch to reduce splash.', 'Maintain balanced irrigation and nutrition.', 'Improve airflow around plants.'], 'field': ['Rotate crops where practical.', 'Stake or trellis plants to improve airflow.', 'Scout lower leaves regularly.'], 'chemical': ['Use a labeled fungicide when locally recommended.', 'Follow label directions and resistance-management advice.'], 'prevention': ['Use healthy transplants.', 'Rotate crops.', 'Maintain sanitation and mulch/soil-splash control.']}, 'hi': {'name': 'Tomato Early Blight', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Tomato Early Blight', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Tomato___Late_blight': {'en': {'name': 'Tomato Late Blight', 'home': ['Remove and destroy severely infected plant material.', 'Do not compost heavily infected debris.', 'Avoid handling plants while wet.'], 'natural': ['Improve airflow.', 'Use irrigation that minimizes leaf wetness.', 'Monitor closely during cool, humid weather.'], 'field': ['Scout frequently.', 'Remove volunteer tomatoes and potatoes.', 'Keep infected debris out of the field.'], 'chemical': ['Use locally recommended late-blight products when risk is high.', 'Follow the label and resistance-management guidance.'], 'prevention': ['Use healthy transplants.', 'Remove volunteer hosts.', 'Monitor weather and plants early.']}, 'hi': {'name': 'Tomato Late Blight', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Tomato Late Blight', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}, 'Tomato___healthy': {'en': {'name': 'Healthy Tomato Leaf', 'home': ['No disease treatment is needed.', 'Remove weeds and damaged leaves.', 'Keep the growing area clean.'], 'natural': ['Water at the root zone.', 'Provide good airflow and sunlight.', 'Maintain balanced nutrition.'], 'field': ['Stake or trellis plants.', 'Scout leaves and fruit regularly.', 'Rotate crops where practical.'], 'chemical': ['Avoid unnecessary pesticide applications.'], 'prevention': ['Use healthy transplants.', 'Maintain sanitation.', 'Monitor plants regularly.']}, 'hi': {'name': 'Healthy Tomato Leaf', 'home': ['इस रोग से प्रभावित भागों को हटाएं और खेत/पौधे का क्षेत्र साफ रखें।', 'संक्रमित अवशेषों को पौधे के पास न छोड़ें।', 'पत्तियों पर लंबे समय तक नमी रहने से बचें।'], 'natural': ['हवा का अच्छा संचार रखें।', 'संतुलित सिंचाई और पोषण दें।', 'पौधों को अनावश्यक तनाव से बचाएं।'], 'field': ['संक्रमित अवशेष हटाएं।', 'पौधों की नियमित निगरानी करें।', 'जहाँ संभव हो उचित दूरी और फसल चक्र अपनाएं।'], 'chemical': ['केवल स्थानीय कृषि विशेषज्ञ की सलाह और उत्पाद के लेबल के अनुसार दवा का उपयोग करें।', 'लेबल में दी गई मात्रा और सुरक्षा निर्देशों का पालन करें।'], 'prevention': ['स्वस्थ बीज/रोपण सामग्री का उपयोग करें।', 'खेत और उपकरणों की स्वच्छता रखें।', 'रोग के शुरुआती लक्षणों की नियमित जांच करें।']}, 'mr': {'name': 'Healthy Tomato Leaf', 'home': ['रोगग्रस्त भाग काढून टाका आणि परिसर स्वच्छ ठेवा.', 'संक्रमित अवशेष झाडाजवळ ठेवू नका.', 'पानांवर जास्त वेळ ओलावा राहू देऊ नका.'], 'natural': ['हवेचे योग्य वहन ठेवा.', 'संतुलित पाणी व पोषण द्या.', 'झाडांवर अनावश्यक ताण येऊ देऊ नका.'], 'field': ['संक्रमित अवशेष काढून टाका.', 'पिकाची नियमित तपासणी करा.', 'शक्य असल्यास योग्य अंतर आणि पीक फेरपालट ठेवा.'], 'chemical': ['स्थानिक कृषी तज्ज्ञांच्या सल्ल्याने आणि उत्पादनाच्या लेबलनुसारच औषध वापरा.', 'लेबलवरील मात्रा व सुरक्षा सूचना पाळा.'], 'prevention': ['निरोगी बियाणे/रोपांचा वापर करा.', 'शेत व साधने स्वच्छ ठेवा.', 'रोगाची सुरुवातीची लक्षणे नियमित तपासा.']}}}


class DBConn:
    """Thin wrapper so the rest of app.py can keep using the same
    sqlite-style calls (conn.execute("...WHERE email=?", (x,)), row["col"],
    conn.commit(), conn.close()) while actually talking to MySQL via PyMySQL.
    """
    def __init__(self):
        self._conn = pymysql.connect(
            host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
            password=MYSQL_PASSWORD, database=MYSQL_DATABASE,
            cursorclass=pymysql.cursors.DictCursor, autocommit=False, charset="utf8mb4",
        )
        self._cursor = self._conn.cursor()

    def execute(self, sql, params=()):
        # sqlite uses "?" placeholders; PyMySQL/MySQL uses "%s".
        self._cursor.execute(sql.replace("?", "%s"), params)
        return self._cursor

    def executescript(self, script):
        for stmt in [s.strip() for s in script.split(";") if s.strip()]:
            self._cursor.execute(stmt)
        self._conn.commit()

    def commit(self):
        self._conn.commit()

    def close(self):
        try:
            self._cursor.close()
        finally:
            self._conn.close()

def db():
    return DBConn()

def init_db():
    conn=db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS users(id INT AUTO_INCREMENT PRIMARY KEY,email VARCHAR(255) UNIQUE NOT NULL,password VARCHAR(255) NOT NULL,name VARCHAR(255) NOT NULL,is_admin INT DEFAULT 0,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS scans(id INT AUTO_INCREMENT PRIMARY KEY,user_id INT,image VARCHAR(255),plant VARCHAR(255),disease VARCHAR(255),confidence FLOAT,status VARCHAR(255),created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS feedback(id INT AUTO_INCREMENT PRIMARY KEY,user_id INT,name VARCHAR(255),message TEXT,rating INT DEFAULT 5,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS suggestions(id INT AUTO_INCREMENT PRIMARY KEY,user_id INT,name VARCHAR(255),message TEXT,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS contact_messages(id INT AUTO_INCREMENT PRIMARY KEY,name VARCHAR(255),email VARCHAR(255),message TEXT,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    """)
    # Safe migrations for older databases.
    try:
        conn.execute("ALTER TABLE feedback ADD COLUMN rating INT DEFAULT 5")
        conn.commit()
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE users ADD COLUMN is_admin INT DEFAULT 0")
        conn.commit()
    except Exception:
        pass

    # Create/promote the project owner as admin without deleting existing users.
    admin_email = os.environ.get("ADMIN_EMAIL", "raginibabhare55@gmail.con").strip().lower()
    admin_password = os.environ.get("ADMIN_PASSWORD", "SmartCrop@Admin2026!")
    admin = conn.execute("SELECT id FROM users WHERE email=?", (admin_email,)).fetchone()
    if admin:
        conn.execute("UPDATE users SET is_admin=1 WHERE id=?", (admin["id"],))
    else:
        conn.execute(
            "INSERT INTO users(name,email,password,is_admin) VALUES(?,?,?,1)",
            ("Ragini Babhare", admin_email, generate_password_hash(admin_password))
        )
    conn.commit(); conn.close()

# CRITICAL FIX: this must run when the MODULE is imported, not only inside
# `if __name__=="__main__"` further below. Gunicorn (used by Render, and
# most production hosts) starts the app with `gunicorn app:app` -- it
# IMPORTS app.py as a module and never runs that block, so the database
# tables (and the admin account) were never created on the live server.
init_db()

def current_lang():
    lang=request.args.get("lang") or session.get("lang") or "en"
    if lang not in LANG_MAP: lang="en"
    session["lang"]=lang
    return lang

def tx(key, lang=None):
    lang=lang or session.get("lang","en")
    return TEXTS.get(lang,TEXTS["en"]).get(key,TEXTS["en"].get(key,key))

def login_required(fn):
    @wraps(fn)
    def wrapper(*a,**kw):
        if "user_id" not in session:
            return redirect(url_for("login", next=request.path))
        return fn(*a,**kw)
    return wrapper

@app.context_processor
def context():
    lang=session.get("lang","en")
    return {"language":lang,"texts":TEXTS.get(lang,TEXTS["en"]),"languages":LANGUAGES,
            "user":session.get("user_name"),"is_admin":bool(session.get("is_admin")),"gemini_enabled":GEMINI_ENABLED,"plantnet_enabled":PLANTNET_ENABLED}

def load_labels():
    if not os.path.exists(LABELS_PATH): return []
    return [x.strip() for x in open(LABELS_PATH,encoding="utf-8") if x.strip()]
labels=load_labels()

interpreter=None; input_details=None; output_details=None
if os.path.exists(MODEL_PATH):
    try:
        interpreter=tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details=interpreter.get_input_details(); output_details=interpreter.get_output_details()
        print("TFLite:", input_details[0]["shape"], input_details[0]["dtype"], "labels:",len(labels))
    except Exception as e: print("Model load error:",e)

def preprocess(img):
    shape=input_details[0]["shape"]; h,w=int(shape[1]),int(shape[2])
    im=Image.open(img).convert("RGB").resize((w,h))
    arr=np.asarray(im,dtype=np.float32)/255.0
    arr=np.expand_dims(arr,0)
    if input_details[0]["dtype"]==np.float32: return arr
    scale,zp=input_details[0]["quantization"]
    return np.round(arr/scale+zp).astype(input_details[0]["dtype"]) if scale else arr.astype(input_details[0]["dtype"])

def raw_predict(arr):
    interpreter.set_tensor(input_details[0]["index"],arr)
    interpreter.invoke()
    out=interpreter.get_tensor(output_details[0]["index"])[0].astype(np.float32)
    # If model returns logits, softmax them. If already probabilities, normalize safely.
    if np.any(out<0) or out.max()>1.001 or abs(out.sum()-1)>0.05:
        e=np.exp(out-out.max()); out=e/e.sum()
    else:
        out=out/(out.sum() or 1)
    return out


def image_quality(path):
    """Fast, conservative quality gate. Returns (ok, message, metrics)."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w,h=im.size
            if w < 256 or h < 256:
                return False, "Please upload a clearer image (at least 256×256 pixels).", {"width":w,"height":h}
            arr=np.asarray(im.resize((256,256)),dtype=np.float32)
            brightness=float(arr.mean())
            contrast=float(arr.std())
            # Laplacian-like sharpness without OpenCV: variance of adjacent differences.
            gray=arr.mean(axis=2)
            sharp=float(np.var(np.diff(gray,axis=0))+np.var(np.diff(gray,axis=1)))
            if brightness < 22:
                return False, "The image is too dark. Please take the photo in better lighting.", {"width":w,"height":h,"brightness":brightness,"sharpness":sharp}
            if brightness > 245:
                return False, "The image is overexposed. Please retake the photo without direct glare.", {"width":w,"height":h,"brightness":brightness,"sharpness":sharp}
            if sharp < 2.0:
                return False, "The image looks blurry. Keep the leaf/plant steady and retake the photo.", {"width":w,"height":h,"brightness":brightness,"sharpness":sharp}
            return True, "", {"width":w,"height":h,"brightness":brightness,"sharpness":sharp}
    except Exception:
        return False, "The image could not be read safely. Please upload a JPG or PNG photo.", {}

def plantnet_identify(path, lang="en"):
    """Primary broad plant-species identification using the official Pl@ntNet API."""
    if not PLANTNET_ENABLED:
        raise RuntimeError("PLANTNET_API_KEY is not configured.")
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width > 2200 or im.height > 2200:
                im.thumbnail((2200, 2200), Image.Resampling.LANCZOS)
            bio = io.BytesIO()
            im.save(bio, format="JPEG", quality=90, optimize=True)
            bio.seek(0)
            filename = os.path.splitext(os.path.basename(path))[0] + ".jpg"
            files = [("images", (filename, bio, "image/jpeg"))]
            # 'auto' is explicitly supported by Pl@ntNet and avoids guessing leaf/flower/fruit.
            data = [("organs", "auto")]
            supported_plantnet_langs = {"en","fr","es","pt","de","it","ar","cs"}
            plang = lang if lang in supported_plantnet_langs else "en"
            url = f"{PLANTNET_URL}/{PLANTNET_PROJECT}"
            params = {
                "api-key": PLANTNET_API_KEY,
                "lang": plang,
                "nb-results": 5,
                "detailed": "true"
            }
            r = requests.post(url, params=params, data=data, files=files, timeout=30)
            if r.status_code == 404:
                raise RuntimeError("Pl@ntNet could not find a reliable plant species in this image.")
            if r.status_code == 429:
                raise RuntimeError("Pl@ntNet daily identification quota has been reached.")
            if r.status_code in (401,403):
                raise RuntimeError("Pl@ntNet API key is invalid or not authorized.")
            r.raise_for_status()
            payload = r.json()
            results = payload.get("results") or []
            if not results:
                return None
            top = results[0]
            species = top.get("species") or {}
            score = float(top.get("score") or 0)
            common = species.get("commonNames") or []
            scientific = species.get("scientificNameWithoutAuthor") or species.get("scientificName") or "Unknown"
            plant = common[0] if common else scientific
            # Keep a moderate threshold; the UI still shows the score and never claims certainty.
            accepted = score >= PLANTNET_MIN_CONFIDENCE
            return {
                "plant": plant, "scientific_name": scientific, "confidence": round(score*100,2),
                "accepted": accepted, "family": ((species.get("family") or {}).get("scientificNameWithoutAuthor") or ""),
                "genus": ((species.get("genus") or {}).get("scientificNameWithoutAuthor") or ""),
                "common_names": common[:5], "gbif_id": str((top.get("gbif") or {}).get("id") or ""),
                "best_match": payload.get("bestMatch") or scientific,
                "engine_version": payload.get("version") or "",
                "remaining_requests": payload.get("remainingIdentificationRequests"),
                "top_results": [{
                    "name": ((x.get("species") or {}).get("scientificNameWithoutAuthor") or "Unknown"),
                    "score": round(float(x.get("score") or 0)*100,2),
                    "common": (((x.get("species") or {}).get("commonNames") or [""])[0])
                } for x in results[:5]]
            }
    except requests.RequestException as e:
        raise RuntimeError("Pl@ntNet service is temporarily unavailable. Please try again.") from e

def plantnet_disease_identify(path):
    """Use Pl@ntNet disease endpoint when available; it covers only a limited set of species/pathologies."""
    if not PLANTNET_ENABLED:
        return None
    try:
        with Image.open(path) as im:
            im=im.convert("RGB")
            if im.width > 2200 or im.height > 2200:
                im.thumbnail((2200,2200), Image.Resampling.LANCZOS)
            bio=io.BytesIO(); im.save(bio,format="JPEG",quality=90,optimize=True); bio.seek(0)
            files=[("images",(os.path.basename(path)+".jpg",bio,"image/jpeg"))]
            data=[("organs","auto")]
            r=requests.post("https://my-api.plantnet.org/v2/diseases/identify",params={"api-key":PLANTNET_API_KEY,"nb-results":5},data=data,files=files,timeout=30)
            if r.status_code in (404,429,401,403):
                return None
            r.raise_for_status()
            payload=r.json(); results=payload.get("results") or []
            if not results: return None
            top=results[0]
            score=float(top.get("score") or 0)
            return {"label":top.get("label") or top.get("name") or "Unknown disease","code":top.get("name") or "","confidence":round(score*100,2),"top_results":results[:5]}
    except Exception:
        return None

def build_plantnet_result(path, lang):
    q=plantnet_identify(path,lang)
    if not q: return None
    q["source"]="Pl@ntNet"
    q["unknown"]=not q["accepted"]
    q["disease"]="Unknown"
    q["explanation"]=f"Plant identification: {q['plant']} ({q['scientific_name']}). Disease identification requires symptom/leaf evidence and is kept separate from species identification."
    return q

def predict_strict(path):
    if interpreter is None: raise RuntimeError("model.tflite is missing or could not be loaded.")
    image=Image.open(path).convert("RGB")
    variants=[image, image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), image.rotate(4,expand=False)]
    preds=[]
    for v in variants:
        # preprocess accepts path; save temporary in memory isn't convenient, use same transform here
        shape=input_details[0]["shape"]; h,w=int(shape[1]),int(shape[2])
        a=np.asarray(v.resize((w,h)),dtype=np.float32)
        a=np.expand_dims(a,0)
        if input_details[0]["dtype"]!=np.float32:
            scale,zp=input_details[0]["quantization"]
            a=np.round(a/scale+zp).astype(input_details[0]["dtype"]) if scale else a.astype(input_details[0]["dtype"])
        preds.append(raw_predict(a))
    mean=np.mean(preds,axis=0)
    order=np.argsort(mean)[::-1]
    top=int(order[0]); second=float(mean[order[1]]) if len(order)>1 else 0
    conf=float(mean[top]); margin=conf-second
    cls=labels[top] if top<len(labels) else "Unknown"
    # Stability is the fraction of views agreeing on the same class.
    stable=sum(int(np.argmax(p)==top) for p in preds)/len(preds)
    accepted=bool(conf>=MIN_CONFIDENCE and margin>=MIN_MARGIN and stable>=2/3)
    plant=cls.split("___")[0] if "___" in cls else cls
    plant_name=SUPPORTED_PLANTS.get(plant, plant)
    return {"class":cls,"confidence":conf*100,"margin":margin*100,"stable":stable*100,
            "accepted":accepted,"plant":plant_name,"supported":plant in SUPPORTED_PLANTS}

def default_info(cls,lang):
    clean=cls.replace("___"," - ").replace("_"," ")
    disease_name=clean
    base={
        "en":{"name":disease_name,"home":[f"Inspect the affected {disease_name} symptoms and remove only severely affected tissue where appropriate.","Keep the plant area clean and remove fallen infected material.","Avoid unnecessary leaf wetness and overwatering."],"natural":["Improve air circulation and sunlight exposure.","Water at the root zone when practical.","Keep tools and hands clean when moving between plants."],"field":["Scout nearby plants for similar symptoms.","Remove and dispose of clearly infected debris appropriately.","Maintain suitable spacing and sanitation."],"chemical":["Use a locally approved product only if the diagnosis is confirmed and treatment is appropriate.","Follow the product label, protective-equipment instructions and local agricultural guidance; never mix products unless the label permits it."],"prevention":["Use healthy planting material.","Monitor the crop regularly for early symptoms.","Record recurring symptoms and consult an agricultural expert if the diagnosis is uncertain."]},
        "hi":{"name":disease_name,"home":["प्रभावित पौधे/पत्तियों के लक्षणों की जांच करें और बहुत प्रभावित भाग को उचित तरीके से हटाएं।","गिरे हुए संक्रमित अवशेष हटाकर क्षेत्र साफ रखें।","अनावश्यक पत्ती-नमी और अधिक पानी से बचें।"],"natural":["हवा का अच्छा संचार और पर्याप्त धूप रखें।","जहाँ संभव हो जड़ क्षेत्र में पानी दें।","पौधों के बीच काम करते समय औजार साफ रखें।"],"field":["आसपास के पौधों में समान लक्षण देखें।","स्पष्ट रूप से संक्रमित अवशेषों को उचित तरीके से हटाएं।","उचित दूरी और खेत की स्वच्छता रखें।"],"chemical":["रोग की पुष्टि और स्थानीय सलाह के बाद ही अनुमोदित दवा का उपयोग करें।","लेबल, सुरक्षा निर्देश और स्थानीय कृषि सलाह का पालन करें; बिना लेबल अनुमति के दवाओं को न मिलाएं।"],"prevention":["स्वस्थ रोपण सामग्री का उपयोग करें।","शुरुआती लक्षणों के लिए नियमित निगरानी करें।","संदेह होने पर कृषि विशेषज्ञ से पुष्टि लें।"]},
        "mr":{"name":disease_name,"home":["प्रभावित पानांची/भागांची लक्षणे तपासा आणि जास्त बाधित भाग योग्य पद्धतीने काढा.","संक्रमित अवशेष काढून परिसर स्वच्छ ठेवा.","पानांवर अनावश्यक ओलावा आणि जास्त पाणी टाळा."],"natural":["हवेचे योग्य वहन आणि पुरेसा सूर्यप्रकाश ठेवा.","शक्य असल्यास मुळाजवळ पाणी द्या.","रोपांमध्ये काम करताना साधने स्वच्छ ठेवा."],"field":["आजूबाजूच्या रोपांमध्ये समान लक्षणे तपासा.","स्पष्टपणे संक्रमित अवशेष योग्य पद्धतीने काढा.","योग्य अंतर आणि शेताची स्वच्छता राखा."],"chemical":["रोगाची खात्री आणि स्थानिक कृषी सल्ल्यानंतरच मान्य औषध वापरा.","लेबल व सुरक्षा सूचनांचे पालन करा; लेबल परवानगी देत नसल्यास औषधे मिसळू नका."],"prevention":["निरोगी लागवड साहित्य वापरा.","सुरुवातीची लक्षणे नियमित तपासा.","संशय असल्यास कृषी तज्ज्ञांकडून निदानाची खात्री करा."]}}
    return base.get(lang,base["en"])

def disease_info(cls,lang):
    data=DISEASE_DATA.get(cls) or default_info(cls,"en")
    if lang in data: return data[lang]
    # Translate on demand for 100+ languages if deep-translator/network is available.
    if not TRANSLATOR_AVAILABLE or lang=="en": return data["en"]
    cache_key = f"disease::{cls}"
    disk_cache = _load_translation_cache(lang)
    if cache_key in disk_cache:
        return disk_cache[cache_key]
    try:
        jobs=[]  # (field_key, item_index_or_None, text)
        for k,v in data["en"].items():
            if isinstance(v,list):
                for i,x in enumerate(v): jobs.append((k,i,x))
            else:
                jobs.append((k,None,v))
        results={}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(20,len(jobs) or 1)) as ex:
            futs={ex.submit(GoogleTranslator(source="en",target=lang).translate,text):(k,i) for k,i,text in jobs}
            for fut in concurrent.futures.as_completed(futs):
                k,i=futs[fut]
                try: results[(k,i)]=fut.result()
                except Exception: results[(k,i)]=None
        out={}
        for k,v in data["en"].items():
            if isinstance(v,list):
                out[k]=[results.get((k,i)) or v[i] for i in range(len(v))]
            else:
                out[k]=results.get((k,None)) or v
        disk_cache[cache_key]=out
        _save_translation_cache(lang, disk_cache)
        return out
    except Exception:
        return data["en"]

TRANSLATION_CACHE_DIR = os.path.join(BASE_DIR, "translation_cache")
os.makedirs(TRANSLATION_CACHE_DIR, exist_ok=True)

def _translation_cache_path(lang):
    return os.path.join(TRANSLATION_CACHE_DIR, f"{lang}.json")

def _load_translation_cache(lang):
    try:
        with open(_translation_cache_path(lang), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_translation_cache(lang, data):
    try:
        with open(_translation_cache_path(lang), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass

def _translate_one(args):
    key, text, lang = args
    try:
        return key, GoogleTranslator(source="en", target=lang).translate(text)
    except Exception:
        return key, text

def translate_texts(lang):
    """Return the full UI text dict for `lang`. On-disk + in-memory caches make
    every switch after the first one instant; the first switch to a brand-new
    language translates all missing strings concurrently (instead of one HTTP
    round trip at a time) so it lands in roughly one request's worth of time
    rather than dozens, keeping language changes to well under a second."""
    if lang in TEXTS: return TEXTS[lang]
    cache = TEXTS.setdefault(lang, {})
    cache.update(_load_translation_cache(lang))
    if not TRANSLATOR_AVAILABLE:
        for k, v in TEXTS["en"].items(): cache.setdefault(k, v)
        return cache
    missing = [(k, v, lang) for k, v in TEXTS["en"].items() if k not in cache]
    if missing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(20, len(missing))) as ex:
            for k, translated in ex.map(_translate_one, missing):
                cache[k] = translated
        _save_translation_cache(lang, cache)
    return cache

@app.route("/set-language/<lang>")
def set_language(lang):
    if lang not in LANG_MAP: lang="en"
    session["lang"]=lang
    return redirect(request.referrer or url_for("home"))

@app.route("/",methods=["GET","POST"])
@login_required
def home():
    lang=current_lang(); texts=translate_texts(lang)
    result=None; info=None; image_url=None; error=None
    if request.method=="POST":
        f=request.files.get("image")
        if not f or not f.filename: error=texts["unknown"]
        elif not allowed_file(f.filename): error="Unsupported image type. Use JPG, PNG or WebP."
        else:
            name=f"{int(time.time()*1000)}_{secure_filename(f.filename)}"; path=os.path.join(UPLOAD_FOLDER,name); f.save(path)
            try:
                quality_ok, quality_msg, quality=image_quality(path)
                if not quality_ok and quality.get("width", 9999) < 256:
                    result={"unknown":True,"confidence":0,"plant":"Unknown","class":"Not Identified","source":"Quality Check","quality":quality}
                    error=quality_msg
                else:
                    # Broad plant identification is now API-first. A soft quality warning does not block API detection.
                    if not quality_ok:
                        error=quality_msg
                    # Broad plant identification is now API-first. The TFLite disease model is retained as a local/offline fallback.
                    plantnet=None
                    if PLANTNET_ENABLED:
                        try: plantnet=build_plantnet_result(path,lang)
                        except Exception as pe: result={"plantnet_error":str(pe)}
                    if plantnet and not plantnet["unknown"]:
                        result=plantnet
                        # If this plant maps to an existing PlantVillage crop, run the local disease classifier too.
                        local=None
                        try: local=predict_strict(path)
                        except Exception: local=None
                        if local and local.get("accepted") and local.get("supported"):
                            info=disease_info(local["class"],lang)
                            result.update({"disease":local["class"],"class":local["class"],"disease_confidence":local["confidence"],"local_disease":True,"local_result":local})
                        else:
                            # First try Pl@ntNet's dedicated disease endpoint; it is limited to supported species/pathologies.
                            pd=plantnet_disease_identify(path)
                            if pd and pd.get("confidence",0) >= 35:
                                result.update({"disease":pd["label"],"class":pd["label"],"disease_confidence":pd["confidence"],"plantnet_disease":True})
                                info=default_info(pd["label"],lang)
                            # For unsupported disease cases, use Gemini Vision for symptom-specific guidance.
                            if not result.get("plantnet_disease") and GEMINI_ENABLED:
                                try:
                                    g=gemini_detect(path,lang)
                                    if not g.get("unknown"):
                                        result.update({"disease":g["disease"],"class":g["disease"],"disease_confidence":g["confidence"],"gemini":True,
                                                       "explanation":g.get("explanation","")})
                                        info={"name":g["disease"],"home":g["home_remedies"],"natural":g["natural"],"field":g["field"],"chemical":g["chemical"],"prevention":g["prevention"]}
                                except Exception as ge: result["gemini_error"]=str(ge)
                        result["unknown"]=False
                        session.update(last_class=result.get("class","Unknown"),last_plant=result.get("plant","Unknown"),last_confidence=round(result.get("confidence",0),2))
                        status="Pl@ntNet + Local Disease" if result.get("local_disease") else ("Pl@ntNet + Gemini" if result.get("gemini") else "Pl@ntNet Identified")
                    else:
                        # If PlantNet is missing/quota-limited, preserve the original local model + Gemini fallback.
                        local=predict_strict(path)
                        result=local
                        result["source"]="Local TFLite"
                        if local["accepted"] and local["supported"]:
                            info=disease_info(local["class"],lang)
                            session.update(last_class=local["class"],last_plant=local["plant"],last_confidence=round(local["confidence"],2))
                            status="Local TFLite"
                        else:
                            result["unknown"]=True; status="Not Identified"
                            if GEMINI_ENABLED:
                                try:
                                    g=gemini_detect(path,lang); result["gemini_result"]=g; result["gemini"]=True
                                    if not g.get("unknown"):
                                        result.update({"unknown":False,"plant":g["plant"],"class":g["disease"],"disease":g["disease"],"confidence":g["confidence"],"accepted":True,"supported":False})
                                        info={"name":g["disease"],"home":g["home_remedies"],"natural":g["natural"],"field":g["field"],"chemical":g["chemical"],"prevention":g["prevention"]}
                                        session.update(last_class=g["disease"],last_plant=g["plant"],last_confidence=round(g["confidence"],2)); status="Gemini Vision"
                                except Exception as ge: result["gemini_error"]=str(ge)
                    if "user_id" in session:
                        c=db(); c.execute("INSERT INTO scans(user_id,image,plant,disease,confidence,status) VALUES(?,?,?,?,?,?)",
                          (session["user_id"],name,result.get("plant","Unknown"),result.get("class",result.get("disease","Not Identified")),result.get("confidence",0),status)); c.commit(); c.close()
                image_url=url_for("static",filename="uploads/"+name)
            except Exception as e:
                error="The AI service could not complete this image. Please try a clearer plant photo or try again."
                result={"unknown":True,"confidence":0,"plant":"Unknown","class":"Not Identified","error":str(e)}
    return render_template("index.html",result=result,disease_info=info,image_url=image_url,error=error)

@app.route("/detect-camera",methods=["POST"])
def detect_camera():
    if "user_id" not in session:
        return jsonify({"ok":False,"error":"Please login first.","login_required":True}),401
    f=request.files.get("image")
    if not f: return jsonify({"ok":False,"error":"No image"}),400
    name=f"{int(time.time()*1000)}_camera.jpg"; path=os.path.join(UPLOAD_FOLDER,name); f.save(path)
    try:
        quality_ok, quality_msg, quality=image_quality(path)
        if not quality_ok: return jsonify({"ok":True,"unknown":True,"message":quality_msg,"quality":quality,"confidence":0})
        result=None
        if PLANTNET_ENABLED:
            try: result=build_plantnet_result(path,session.get("lang","en"))
            except Exception as pe: result={"plantnet_error":str(pe)}
        if result and not result.get("unknown"):
            result["ok"]=True
            try:
                local=predict_strict(path)
                if local.get("accepted") and local.get("supported"):
                    result.update({"class":local["class"],"disease":local["class"],"disease_confidence":local["confidence"],"local_disease":True})
            except Exception: pass
            if not result.get("local_disease") and GEMINI_ENABLED:
                try:
                    g=gemini_detect(path,session.get("lang","en"))
                    if not g.get("unknown"): result.update({"class":g["disease"],"disease":g["disease"],"disease_confidence":g["confidence"],"gemini":True,"explanation":g.get("explanation","")})
                except Exception: pass
            session.update(last_class=result.get("class","Unknown"),last_plant=result.get("plant","Unknown"),last_confidence=round(result.get("confidence",0),2))
            c=db(); c.execute("INSERT INTO scans(user_id,image,plant,disease,confidence,status) VALUES(?,?,?,?,?,?)",(session["user_id"],name,result.get("plant","Unknown"),result.get("class","Unknown"),result.get("confidence",0),"Pl@ntNet")); c.commit(); c.close()
            result["image_url"]=url_for("static",filename="uploads/"+name)
            return jsonify(result)
        # fallback to local model, then Gemini
        local=predict_strict(path); local["source"]="Local TFLite"
        if local.get("accepted") and local.get("supported"):
            session.update(last_class=local["class"],last_plant=local["plant"],last_confidence=round(local["confidence"],2))
            local.update({"ok":True,"image_url":url_for("static",filename="uploads/"+name)})
            return jsonify(local)
        if GEMINI_ENABLED:
            g=gemini_detect(path,session.get("lang","en"))
            if not g.get("unknown"):
                session.update(last_class=g["disease"],last_plant=g["plant"],last_confidence=round(g["confidence"],2))
                return jsonify({"ok":True,"unknown":False,"plant":g["plant"],"class":g["disease"],"confidence":g["confidence"],"gemini":True,"gemini_result":g,"image_url":url_for("static",filename="uploads/"+name)})
        return jsonify({"ok":True,"unknown":True,"message":"Plant could not be identified reliably. Try a clear leaf/flower photo.","confidence":local.get("confidence",0),"image_url":url_for("static",filename="uploads/"+name)})
    except Exception:
        return jsonify({"ok":False,"error":"Detection service is temporarily unavailable. Please try again."}),500

@app.route("/login",methods=["GET","POST"])
def login():
    lang=current_lang()
    next_url=request.values.get("next") or url_for("home")
    if next_url.startswith("//") or "://" in next_url:
        next_url=url_for("home")  # only allow same-site redirects
    if request.method=="POST":
        email=request.form.get("email","").strip().lower(); pw=request.form.get("password","")
        c=db(); u=c.execute("SELECT * FROM users WHERE email=?",(email,)).fetchone(); c.close()
        if u and check_password_hash(u["password"],pw):
            session.update(user_id=u["id"],user_name=u["name"],is_admin=bool(u["is_admin"] if "is_admin" in u.keys() else 0)); return redirect(next_url)
        flash(TEXTS[lang]["invalid_login"],"error")
    return render_template("auth.html",mode="login",next=next_url)

@app.route("/register",methods=["GET","POST"])
def register():
    lang=current_lang()
    next_url=request.values.get("next") or url_for("home")
    if next_url.startswith("//") or "://" in next_url:
        next_url=url_for("home")
    if request.method=="POST":
        name=request.form.get("name","").strip(); email=request.form.get("email","").strip().lower(); pw=request.form.get("password","")
        try:
            c=db(); c.execute("INSERT INTO users(name,email,password) VALUES(?,?,?)",(name,email,generate_password_hash(pw))); c.commit()
            uid=c.execute("SELECT id FROM users WHERE email=?",(email,)).fetchone()["id"]; c.close()
            session.update(user_id=uid,user_name=name,is_admin=False); return redirect(next_url)
        except IntegrityError: flash(TEXTS[lang]["email_exists"],"error")
    return render_template("auth.html",mode="register",next=next_url)

@app.route("/logout")
def logout(): session.clear(); return redirect(url_for("home"))


@app.route("/supported-plants")
def supported_plants():
    """Show the crops and disease classes supported by the local TFLite model."""
    current_lang()
    crop_map = {}
    for raw in labels:
        parts = raw.split("___", 1)
        if len(parts) != 2:
            continue
        crop, disease = parts
        crop = crop.replace("_", " ").replace("(maize)", "(Maize)")
        crop = re.sub(r"\s*,\s*", ", ", crop)
        crop = re.sub(r"\s+", " ", crop).strip()
        disease = disease.replace("_", " ").strip()
        disease = re.sub(r"\s+", " ", disease)
        crop_map.setdefault(crop, []).append(disease)
    return render_template(
        "supported_plants.html",
        crop_map=crop_map,
        total_labels=len(labels),
    )

@app.route("/dashboard")
@login_required
def dashboard():
    c=db(); scans=c.execute("SELECT * FROM scans WHERE user_id=? ORDER BY id DESC LIMIT 20",(session["user_id"],)).fetchall(); c.close()
    return render_template("dashboard.html",scans=scans)



def admin_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if "user_id" not in session:
            return redirect(url_for("login", next=request.path))
        c = db()
        u = c.execute("SELECT is_admin FROM users WHERE id=?", (session["user_id"],)).fetchone()
        c.close()
        if not u or not bool(u["is_admin"]):
            flash("Admin access required.", "error")
            return redirect(url_for("home"))
        session["is_admin"] = True
        return fn(*a, **kw)
    return wrapper

@app.route("/admin")
@admin_required
def admin_panel():
    c = db()
    users = c.execute("SELECT id,name,email,is_admin,created_at FROM users ORDER BY id DESC").fetchall()
    scans = c.execute("SELECT scans.*, users.name AS user_name, users.email AS user_email FROM scans LEFT JOIN users ON users.id=scans.user_id ORDER BY scans.id DESC LIMIT 100").fetchall()
    contacts = c.execute("SELECT * FROM contact_messages ORDER BY id DESC LIMIT 30").fetchall()
    feedback = c.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT 30").fetchall()
    suggestions = c.execute("SELECT * FROM suggestions ORDER BY id DESC LIMIT 30").fetchall()
    stats = {
        "users": c.execute("SELECT COUNT(*) AS cnt FROM users").fetchone()["cnt"],
        "scans": c.execute("SELECT COUNT(*) AS cnt FROM scans").fetchone()["cnt"],
        "feedback": c.execute("SELECT COUNT(*) AS cnt FROM feedback").fetchone()["cnt"],
        "contacts": c.execute("SELECT COUNT(*) AS cnt FROM contact_messages").fetchone()["cnt"],
        "suggestions": c.execute("SELECT COUNT(*) AS cnt FROM suggestions").fetchone()["cnt"],
    }
    c.close()
    return render_template("admin.html", users=users, scans=scans, contacts=contacts, feedback=feedback, suggestions=suggestions, stats=stats)

@app.route("/admin/delete-scan/<int:scan_id>", methods=["POST"])
@admin_required
def admin_delete_scan(scan_id):
    c = db()
    row = c.execute("SELECT image FROM scans WHERE id=?", (scan_id,)).fetchone()
    c.execute("DELETE FROM scans WHERE id=?", (scan_id,))
    c.commit(); c.close()
    if row and row["image"]:
        safe_name = os.path.basename(row["image"])
        file_path = os.path.join(UPLOAD_FOLDER, safe_name)
        if os.path.isfile(file_path):
            try: os.remove(file_path)
            except OSError: pass
    flash("Selected scan deleted.", "success")
    return redirect(url_for("admin_panel") + "#scans")

@app.route("/admin/delete-scans", methods=["POST"])
@admin_required
def admin_delete_scans():
    c = db()
    rows = c.execute("SELECT image FROM scans").fetchall()
    c.execute("DELETE FROM scans")
    c.commit(); c.close()
    for row in rows:
        if row["image"]:
            safe_name = os.path.basename(row["image"])
            file_path = os.path.join(UPLOAD_FOLDER, safe_name)
            if os.path.isfile(file_path):
                try: os.remove(file_path)
                except OSError: pass
    flash("All dashboard scan data has been deleted by admin.", "success")
    return redirect(url_for("admin_panel") + "#scans")

@app.route("/about")
def about(): current_lang(); return render_template("about.html")

@app.route("/feedback",methods=["GET","POST"])
def feedback():
    lang=current_lang()
    if request.method=="POST":
        name=request.form.get("name","Farmer").strip(); msg=request.form.get("message","").strip()
        try: rating=int(request.form.get("rating",5))
        except (TypeError,ValueError): rating=5
        rating=max(1,min(5,rating))
        if msg:
            c=db(); c.execute("INSERT INTO feedback(user_id,name,message,rating) VALUES(?,?,?,?)",(session.get("user_id"),name,msg,rating)); c.commit(); c.close()
            flash(TEXTS[lang]["feedback_saved"],"success"); return redirect(url_for("feedback"))
    c=db()
    recent=c.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT 10").fetchall()
    stats=c.execute("SELECT AVG(rating) AS avg_rating, COUNT(*) AS cnt FROM feedback").fetchone()
    c.close()
    return render_template("feedback.html",recent=recent,avg_rating=stats["avg_rating"],feedback_count=stats["cnt"])

@app.route("/contact",methods=["GET","POST"])
def contact():
    lang=current_lang()
    if request.method=="POST":
        name=request.form.get("name","").strip(); email=request.form.get("email","").strip(); msg=request.form.get("message","").strip()
        if name and msg:
            c=db(); c.execute("INSERT INTO contact_messages(name,email,message) VALUES(?,?,?)",(name,email,msg)); c.commit(); c.close()
            flash(TEXTS[lang]["contact_saved"],"success"); return redirect(url_for("contact"))
    return render_template("contact.html")

@app.route("/suggestions",methods=["GET","POST"])
def suggestions():
    lang=current_lang()
    if request.method=="POST":
        name=request.form.get("name","Farmer").strip(); msg=request.form.get("message","").strip()
        if msg:
            c=db(); c.execute("INSERT INTO suggestions(user_id,name,message) VALUES(?,?,?)",(session.get("user_id"),name,msg)); c.commit(); c.close()
            flash(TEXTS[lang]["suggestion_saved"],"success"); return redirect(url_for("suggestions"))
    c=db(); items=c.execute("SELECT * FROM suggestions ORDER BY id DESC LIMIT 10").fetchall(); c.close()
    return render_template("suggestions.html",items=items)



def offline_farmer_answer(message, lang):
    """Small offline safety-net so the assistant still works when Gemini is unavailable/quota-limited."""
    q = message.lower()
    packs = {
        "summer": {
            "en": "🌱 Summer care: water early morning or evening, mulch around the root zone, provide shade for sensitive plants, and check leaves regularly for heat stress and pests.",
            "hi": "🌱 गर्मियों में: सुबह जल्दी या शाम को पानी दें, जड़ों के आसपास मल्च रखें, संवेदनशील पौधों को छाया दें और पत्तियों में गर्मी/कीट के लक्षण देखें।",
            "mr": "🌱 उन्हाळ्यात: सकाळी लवकर किंवा संध्याकाळी पाणी द्या, मुळांभोवती आच्छादन ठेवा, संवेदनशील झाडांना सावली द्या आणि पाने/किडींची नियमित तपासणी करा।"
        },
        "water": {
            "en": "💧 Irrigation: water according to soil moisture and crop stage rather than a fixed schedule. Avoid waterlogging and wetting leaves unnecessarily.",
            "hi": "💧 सिंचाई: तय समय के बजाय मिट्टी की नमी और फसल की अवस्था देखकर पानी दें। जलभराव और पत्तियों को अनावश्यक रूप से गीला करने से बचें।",
            "mr": "💧 सिंचन: ठराविक वेळेपेक्षा मातीतील ओलावा आणि पिकाची अवस्था पाहून पाणी द्या. पाणी साचणे आणि पानांवर अनावश्यक ओलावा टाळा."
        },
        "disease": {
            "en": "🛡️ Disease prevention: remove severely infected leaves, keep the field clean, improve air circulation, avoid prolonged leaf wetness, and monitor plants regularly. For pesticides, follow the locally approved product label.",
            "hi": "🛡️ रोग से बचाव: बहुत संक्रमित पत्तियाँ हटाएं, खेत साफ रखें, हवा का संचार बढ़ाएं, पत्तियों पर लंबे समय तक नमी से बचें और नियमित निगरानी करें। कीटनाशक के लिए स्थानीय रूप से स्वीकृत उत्पाद का लेबल मानें।",
            "mr": "🛡️ रोग प्रतिबंध: जास्त संक्रमित पाने काढा, शेत स्वच्छ ठेवा, हवेचे वहन वाढवा, पानांवर दीर्घकाळ ओलावा टाळा आणि नियमित पाहणी करा. कीटकनाशकासाठी स्थानिक मान्य उत्पादनाच्या लेबलवरील सूचना पाळा."
        },
        "fertilizer": {
            "en": "🌾 Fertilizer: use a soil-test or crop-stage-based recommendation where possible. Avoid excess fertilizer, especially when plants are heat- or water-stressed.",
            "hi": "🌾 खाद: संभव हो तो मिट्टी परीक्षण या फसल की अवस्था के अनुसार खाद दें। अधिक खाद न दें, खासकर जब पौधे गर्मी या पानी की कमी से तनाव में हों।",
            "mr": "🌾 खत: शक्य असल्यास माती परीक्षण किंवा पिकाच्या अवस्थेनुसार खत द्या. विशेषतः उष्णता किंवा पाण्याच्या ताणात जास्त खत देऊ नका."
        }
    }
    keys = []
    if any(x in q for x in ("summer","heat","गर्मी","गर्म","उन्हाळा","उन्हाळ्यात")): keys.append("summer")
    if any(x in q for x in ("water","irrigat","पानी","सिंच","पाणी","सिंचन")): keys.append("water")
    if any(x in q for x in ("disease","protect","prevention","रोग","बीमारी","बचाव","रोकथाम","रोगां")): keys.append("disease")
    if any(x in q for x in ("fertilizer","fertiliser","खाद","उर्वरक","खत")): keys.append("fertilizer")
    no_match_text = {
        "en": "🌱 Smart Crop AI offline assistant: I can help with basic plant disease prevention, irrigation, summer care, field hygiene and fertilizer guidance. Gemini is currently unavailable, so please try again later for a full AI answer.",
        "hi": "🌱 स्मार्ट क्रॉप एआई ऑफ़लाइन सहायक: मैं बुनियादी पौधों की बीमारी की रोकथाम, सिंचाई, गर्मी की देखभाल, क्षेत्र की स्वच्छता और उर्वरक मार्गदर्शन में मदद कर सकता हूं। जेमिनी फिलहाल अनुपलब्ध है, इसलिए पूर्ण एआई उत्तर के लिए कृपया बाद में पुनः प्रयास करें।",
        "mr": "🌱 स्मार्ट क्रॉप एआय ऑफलाइन सहाय्यक: मी मूलभूत वनस्पती रोग प्रतिबंध, सिंचन, उन्हाळी काळजी, शेत स्वच्छता आणि खत मार्गदर्शनात मदत करू शकतो. जेमिनी सध्या उपलब्ध नाही, त्यामुळे संपूर्ण एआय उत्तरासाठी कृपया नंतर पुन्हा प्रयत्न करा."
    }
    # Built-in languages (en/hi/mr) already have hand-written translations for
    # every pack above — use those directly (instant, no network call) instead
    # of always building the English text and translating it live, which was
    # the bug making replies look "stuck" in one language.
    if lang in ("en","hi","mr"):
        if not keys:
            return no_match_text[lang]
        return "\n\n".join(packs[k].get(lang, packs[k]["en"]) for k in dict.fromkeys(keys))
    # Any other language: build the English answer, then translate on demand.
    answer = no_match_text["en"] if not keys else "\n\n".join(packs[k]["en"] for k in dict.fromkeys(keys))
    if TRANSLATOR_AVAILABLE:
        try:
            return GoogleTranslator(source="en", target=lang).translate(answer)
        except Exception as e:
            print("Offline-answer translation failed, returning English:", e)
    return answer

@app.route("/api/chat",methods=["POST"])
def api_chat():
    data=request.get_json(silent=True) or {}
    message=str(data.get("message","")).strip()
    if not message:
        return jsonify({"ok":False,"error":"Message is required."}),400
    try:
        # Let the farmer type their question in ANY language, regardless of
        # which language is selected in the site's dropdown, and reply back
        # in that same detected language.
        site_lang=session.get("lang","en")
        lang=detect_message_language(message, fallback=site_lang)
        low=message.lower()
        precaution_words=("precaution","precautions","treatment","remedy","remedies","prevent","prevention","care","क्या सावधानी","उपचार","काळजी","उपाय")
        last_cls=session.get("last_class")
        if last_cls and any(w in low for w in precaution_words):
            info=disease_info(last_cls,lang)
            lines=[info.get("name",last_cls), "", "Home / Basic Care:"] + [f"• {x}" for x in info.get("home",[])]
            lines += ["", "Natural Management:"] + [f"• {x}" for x in info.get("natural",[])]
            lines += ["", "Field Management:"] + [f"• {x}" for x in info.get("field",[])]
            lines += ["", "Chemical Control:"] + [f"• {x}" for x in info.get("chemical",[])]
            lines += ["", "Prevention:"] + [f"• {x}" for x in info.get("prevention",[])]
            return jsonify({"ok":True,"answer":"\n".join(lines),"source":"local_guidance"})
        if not GEMINI_ENABLED:
            return jsonify({"ok":True,"answer":offline_farmer_answer(message,lang),"source":"offline_assistant","detected_lang":lang})
        answer=gemini_chat(message,lang,data.get("history") or [])
        return jsonify({"ok":True,"answer":answer,"detected_lang":lang,"context":{"plant":session.get("last_plant"),"disease":session.get("last_class"),"confidence":session.get("last_confidence")}})
    except Exception as e:
        # Keep the existing assistant behavior, but show a friendly message
        # when Gemini free-tier quota is exhausted (HTTP 429).
        err_text = str(e)
        print("Gemini chat error (falling back to offline assistant):", err_text)
        # Never break the farmer UI on quota/network/server failures.
        return jsonify({"ok":True,"answer":offline_farmer_answer(message,lang),"source":"offline_assistant","fallback_reason":"ai_unavailable"})

@app.route("/api/plantnet-test")
def plantnet_test():
    """Safe server-side diagnostic: reports configuration and quota without exposing the API key."""
    if not PLANTNET_ENABLED:
        return jsonify({"ok":False,"configured":False,"error":"PLANTNET_API_KEY is missing from .env"}),400
    try:
        r=requests.get("https://my-api.plantnet.org/v2/quota/daily",params={"api-key":PLANTNET_API_KEY},timeout=15)
        if r.status_code in (401,403):
            return jsonify({"ok":False,"configured":True,"error":"PlantNet API key is invalid or not authorized."}),401
        r.raise_for_status()
        return jsonify({"ok":True,"configured":True,"quota":r.json()})
    except Exception as e:
        return jsonify({"ok":False,"configured":True,"error":"Could not reach PlantNet quota service."}),502

@app.route("/api/health")
def api_health():
    return jsonify({"ok":True,"service":"Smart Crop AI","plantnet":PLANTNET_ENABLED,"gemini":GEMINI_ENABLED,
                    "local_model":os.path.exists(MODEL_PATH),"local_classes":len(labels),
                    "plantnet_project":PLANTNET_PROJECT,"model_version":"PlantDiseaseModel-v1"})

@app.route("/api/weather")
def weather():
    import urllib.request, urllib.parse
    city=request.args.get("city","").strip()
    lat=request.args.get("lat"); lon=request.args.get("lon")
    try:
        if lat and lon:
            qlat,qlon=float(lat),float(lon)
            place="Your location"
        elif city:
            q="https://geocoding-api.open-meteo.com/v1/search?"+urllib.parse.urlencode({"name":city,"count":1,"language":"en","format":"json"})
            data=json.loads(urllib.request.urlopen(q,timeout=8).read().decode())
            if not data.get("results"): return jsonify({"ok":False,"error":"City not found"}),404
            r=data["results"][0]; qlat,qlon=r["latitude"],r["longitude"]; place=r.get("name",city)
        else: return jsonify({"ok":False,"error":"Location or city required"}),400
        q="https://api.open-meteo.com/v1/forecast?"+urllib.parse.urlencode({"latitude":qlat,"longitude":qlon,"current":"temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m","timezone":"auto"})
        data=json.loads(urllib.request.urlopen(q,timeout=8).read().decode())
        return jsonify({"ok":True,"place":place,"current":data.get("current",{})})
    except Exception as e: return jsonify({"ok":False,"error":str(e)}),500

def allowed_file(filename): return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

if __name__=="__main__":
    init_db()
    print("Smart Crop AI | model:",os.path.exists(MODEL_PATH),"labels:",len(labels),"Gemini:",GEMINI_ENABLED,"model:",GEMINI_MODEL)
    app.run(debug=True,host="0.0.0.0",port=5000)
