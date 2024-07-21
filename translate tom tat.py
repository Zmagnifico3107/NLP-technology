import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import spacy
from pyvi import ViTokenizer, ViPosTagger
from langdetect import detect
from googletrans import Translator

# Tải các tài nguyên cần thiết từ nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('stopwords')

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
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Hàm nhận diện thực thể
def recognize_entities(text):
    """Nhận diện thực thể trong văn bản bằng spaCy"""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Nhập câu cần xử lý
sentence = str(input("Nhập câu cần xử lý: "))

# Nhận diện thực thể
entities = recognize_entities(sentence)
print("Tên địa danh và thực thể nhận diện được:")
for entity in entities:
    print(f"{entity[0]}: {entity[1]}")

# Xử lý văn bản
words = word_tokenize(sentence)
filtered_words = [word for word in words if word.lower() not in stop_words]
lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in filtered_words]
processed_sentence = ' '.join(lemmatized_words)

print("Câu đã xử lý:")
print(processed_sentence)

# Tóm tắt văn bản đã xử lý
parser = PlaintextParser.from_string(processed_sentence, Tokenizer("english"))
summarizer = TextRankSummarizer()
summary = summarizer(parser.document, 2)  # Số câu tóm tắt

print("Văn bản tóm tắt:")
summary_text = ' '.join([str(sent) for sent in summary])
print(summary_text)

# Phát hiện ngôn ngữ của văn bản tóm tắt
language = detect(summary_text)
print(f"Ngôn ngữ phát hiện: {language}")

# Nếu ngôn ngữ không phải là tiếng Việt, dịch văn bản tóm tắt sang tiếng Việt
if language != 'vi':
    translated = translator.translate(summary_text, src=language, dest='vi')
    translated_text = translated.text
    print(f"Văn bản tóm tắt đã dịch sang tiếng Việt: {translated_text}")
