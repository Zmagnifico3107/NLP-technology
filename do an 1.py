import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Tải các tài nguyên cần thiết từ NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Khởi tạo đối tượng WordNetLemmatizer và lấy danh sách từ dừng cho tiếng Anh
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Hàm để lấy dạng từ (POS) chính xác cho lemmatization
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Nhập câu cần xử lý
sentence = str(input("Nhập câu cần xử lý: "))

# Tokenize câu thành các từ
words = word_tokenize(sentence)

# Loại bỏ các từ dừng
filtered_words = [word for word in words if word.lower() not in stop_words]

# Lemmatization các từ còn lại
lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in filtered_words]

# Tạo lại câu từ các từ đã lọc và lemmatization
processed_sentence = ' '.join(lemmatized_words)

# Sử dụng CountVectorizer để mã hóa từ thành các số
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([processed_sentence])

# Hiển thị từ điển các từ và ma trận mã hóa
print("Các từ sau khi mã hóa thành các số:")
print(vectorizer.vocabulary_)
print("Ma trận mã hóa:")
print(X.toarray())

# Tokenize câu thành các câu con (không bắt buộc, chỉ để hiển thị nếu cần)
sentences = sent_tokenize(sentence)
print("Các câu con trong đoạn văn:")
for sent in sentences:
    print(sent)
