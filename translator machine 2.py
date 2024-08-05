import os
from docx import Document
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import spacy
from langdetect import detect
from googletrans import Translator
import speech_recognition as sr
from gtts import gTTS
import pygame
from io import BytesIO
import pyttsx3

# Tải mô hình tiếng Anh của spaCy
nlp = spacy.load("en_core_web_sm")

# Khởi tạo đối tượng WordNetLemmatizer và lấy danh sách từ dừng cho tiếng Anh
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Khởi tạo đối tượng Translator từ googletrans
translator = Translator()

# Hàm để lấy dạng từ (POS) chính xác cho lemmatization
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Hàm nhận diện thực thể
def recognize_entities(text):
    """Nhận diện thực thể trong văn bản bằng spaCy"""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Hàm tóm tắt văn bản
def summarize_text(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, 2)  # Số câu tóm tắt
    summary_text = ' '.join([str(sent) for sent in summary])
    return summary_text

# Hàm dịch văn bản
def translate_text(text, src_lang, dest_lang='vi'):
    translated = translator.translate(text, src=src_lang, dest=dest_lang)
    return translated.text

# Hàm nhận dạng giọng nói
def get_input_from_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Hãy nói gì đó:")
        recognizer.adjust_for_ambient_noise(source)  # Điều chỉnh nhiễu âm thanh
        audio = recognizer.listen(source)

    try:
        sentence = recognizer.recognize_google(audio, language='vi')  # Nhận dạng tiếng Việt
        return sentence
    except sr.UnknownValueError:
        print("Không thể nhận diện giọng nói.")
        return ""
    except sr.RequestError:
        print("Yêu cầu không được xử lý, vui lòng kiểm tra kết nối mạng.")
        return ""

# Chuyển đổi tên ngôn ngữ sang mã ngôn ngữ
def get_language_code(language_name):
    languages = {
        'afrikaans': 'af', 'albanian': 'sq', 'amharic': 'am', 'arập': 'ar', 'armenian': 'hy',
        'azerbaijani': 'az', 'basque': 'eu', 'belarusian': 'be', 'bengali': 'bn', 'bosnian': 'bs',
        'bulgary': 'bg', 'catalan': 'ca', 'cebuano': 'ceb', 'chichewa': 'ny', 'tiếng Trung (giản thể)': 'zh-cn',
        'tiếng trung (truyền thống)': 'zh-tw', 'corsican': 'co', 'croatia': 'hr', 'cộng hòa séc': 'cs', 'danish': 'da',
        'dutch': 'nl', 'tiếng anh': 'en', 'esperanto': 'eo', 'estonian': 'et', 'filipino': 'tl',
        'finnish': 'fi', 'tiếng pháp': 'fr', 'frisian': 'fy', 'galician': 'gl', 'georgian': 'ka',
        'tiếng đức': 'de', 'tiếng Hy Lạp': 'el', 'gujarati': 'gu', 'haitian creole': 'ht', 'hausa': 'ha',
        'hawaiian': 'haw', 'hebrew': 'he', 'hinđu': 'hi', 'hmong': 'hmn', 'hungary': 'hu',
        'icelandic': 'is', 'igbo': 'ig', 'indonesia': 'id', 'irish': 'ga', 'italian': 'it',
        'tiếng Nhật': 'ja', 'javanese': 'jw', 'kannada': 'kn', 'kazakh': 'kk', 'khmer': 'km',
        'korean': 'ko', 'kurdish (kurmanji)': 'ku', 'kyrgyz': 'ky', 'lao': 'lo', 'latin': 'la',
        'latvian': 'lv', 'lithuanian': 'lt', 'luxembourgish': 'lb', 'macedonian': 'mk', 'malagasy': 'mg',
        'malay': 'ms', 'malayalam': 'ml', 'maltese': 'mt', 'maori': 'mi', 'marathi': 'mr',
        'mongolian': 'mn', 'myanmar (burmese)': 'my', 'nepali': 'ne', 'norwegian': 'no', 'odia': 'or',
        'pashto': 'ps', 'persian': 'fa', 'polish': 'pl', 'bồ đào nha': 'pt', 'punjabi': 'pa',
        'romanian': 'ro', 'nga': 'ru', 'samoan': 'sm', 'scots gaelic': 'gd', 'serbian': 'sr',
        'sesotho': 'st', 'shona': 'sn', 'sindhi': 'sd', 'sinhala': 'si', 'slovak': 'sk',
        'slovenian': 'sl', 'somali': 'so', 'tiếng tây ban nha': 'es', 'sundanese': 'su', 'swahili': 'sw',
        'swedish': 'sv', 'tajik': 'tg', 'tamil': 'ta', 'telugu': 'te', 'thai': 'th',
        'turkish': 'tr', 'ukrainian': 'uk', 'urdu': 'ur', 'uyghur': 'ug', 'uzbek': 'uz',
        'tiếng việt': 'vi', 'welsh': 'cy', 'xhosa': 'xh', 'yiddish': 'yi', 'yoruba': 'yo',
        'zulu': 'zu'
    }
    return languages.get(language_name.lower(), 'vi')

# Hàm phát âm thanh từ văn bản
def speak_with_pyttsx3(text, lang='en'):
    engine = pyttsx3.init()
    engine.setProperty('volume', 1.5)  # Điều chỉnh âm lượng lớn hơn
    voices = engine.getProperty('voices')
    
    # Chọn giọng đọc phù hợp
    for voice in voices:
        if lang in voice.languages:
            engine.setProperty('voice', voice.id)
            break
    
    engine.say(text)
    engine.runAndWait()

def speak_with_gtts(text, lang='vi'):
    # Tạo âm thanh từ văn bản
    tts = gTTS(text=text, lang=lang, slow=False)
    
    # Tạo một luồng Bytes từ âm thanh
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)

    # Khởi tạo pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load(audio_bytes)
    pygame.mixer.music.play()

    # Đợi cho đến khi âm thanh phát xong
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Hàm đọc văn bản từ file docx
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    print("văn bản từ file: ", full_text)
    return '\n'.join(full_text)

# Chương trình chính
def main():
    while True:
        choice = input("Bạn muốn nhập câu bằng cách nào? (nhập 'key' (bàn phím), 'mic' (giọng nói), hoặc 'file' (file .docx)): ").strip().lower()

        if choice == 'key':
            sentence = input("Nhập câu cần xử lý: ")
            break  # Thoát khỏi vòng lặp sau khi xử lý thành công
        elif choice == 'mic':
            sentence = get_input_from_speech()
            if sentence:
                break  # Thoát khỏi vòng lặp sau khi xử lý thành công
        elif choice == 'file':
            file_path = input("Nhập đường dẫn tới file .docx: ")
            if os.path.exists(file_path):
                sentence = extract_text_from_docx(file_path)
                break  # Thoát khỏi vòng lặp sau khi xử lý thành công
            else:
                print("File không tồn tại. Vui lòng kiểm tra lại đường dẫn.")
        else:
            print("Lựa chọn không hợp lệ. Vui lòng nhập 'key', 'mic', hoặc 'file'.")

    # Phát hiện ngôn ngữ của câu
    input_language = detect(sentence)
    print(f"Ngôn ngữ đầu vào: {input_language}")

    if input_language == 'vi':
        dest_lang_name = input("Bạn muốn dịch sang ngôn ngữ nào? (ví dụ: 'English' cho tiếng Anh, 'French' cho tiếng Pháp): ").strip().lower()
        dest_lang = get_language_code(dest_lang_name)
        translated_sentence = translate_text(sentence, src_lang='vi', dest_lang=dest_lang)
        print(f"translated: {translated_sentence}")
    else:
        # Dịch câu sang tiếng Việt
        dest_lang = 'vi'
        translated_sentence = translate_text(sentence, src_lang=input_language, dest_lang=dest_lang)
        print(f"Câu đã dịch sang tiếng Việt: {translated_sentence}")

    print("Bạn có muốn tôi thực hiện tác vụ khác không?\n"
          "Vui lòng chọn một hoặc nhiều tác vụ, cách nhau bởi dấu phẩy:\n"
          "'NER': nhận diện thực thể,\n"
          "'sum': tóm tắt văn bản đã dịch,\n"
          "'read': đọc văn bản đã dịch\n"
          "'all': thực hiện tất cả tác vụ\n"
          "'none': kết thúc chương trình")
    
    valid_tasks = {'ner', 'sum', 'read', 'all', 'none'}
    while True:
        tasks = input("Nhập lựa chọn của bạn: ").strip().lower()

        # Xử lý các lựa chọn
        task_list = [task.strip() for task in tasks.split(',')]
        # Kiểm tra nếu tất cả các tác vụ trong task_list đều hợp lệ
        if all(task in valid_tasks for task in task_list):
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng chỉ nhập 'NER', 'sum', 'read', 'all', hoặc 'none'.")

    if 'all' in task_list:
        # Nếu chọn 'all', thực hiện tất cả các tác vụ
        print("Thực hiện tất cả các tác vụ.")
        entities = recognize_entities(sentence)
        print("Tên địa danh và thực thể nhận diện được:")
        for entity in entities:
            print(f"{entity[0]}: {entity[1]}")
        # tóm tắt văn bản
        summary_text = summarize_text(sentence)
        print("Tóm tắt văn bản đã dịch:")
        print(summary_text)
        # đọc văn bản
        print("Thực hiện đọc văn bản.")
        if dest_lang == 'vi':
            speak_with_gtts(translated_sentence, lang='vi')
        else:
            speak_with_pyttsx3(translated_sentence, lang=dest_lang)

    else:
        if 'ner' in task_list:
            print("Thực hiện tác vụ nhận diện thực thể.")
            # Thực hiện tác vụ NER
            entities = recognize_entities(sentence)
            print("Tên địa danh và thực thể nhận diện được:")
            for entity in entities:
                print(f"{entity[0]}: {entity[1]}")

        if 'sum' in task_list:
            print("Thực hiện tóm tắt văn bản:")
            # Thực hiện tác vụ tóm tắt văn bản
            summary = input("Bạn muốn tóm tắt văn bản nào?\n"
                            "vui lòng chọn('main' hoặc 'translated'): ")
            if summary == 'main':
                summary_text = summarize_text(sentence)
                print(" tóm tắt Văn bản gốc: ")
                print(summary_text)
            if summary == 'translated':
                summary_text = summarize_text(translated_sentence)
                print(" tóm tắt Văn bản đã dịch:")
                print(summary_text)

        if 'read' in task_list:
            print("Thực hiện đọc văn bản.")
            if dest_lang == 'vi':
                speak_with_gtts(translated_sentence, lang='vi')
            else:
                speak_with_pyttsx3(translated_sentence, lang=dest_lang)

        if 'none' in task_list:
            print("Kết thúc chương trình. Xin cảm ơn!")
            return  # Kết thúc chương trình

if __name__ == "__main__":
    main()
