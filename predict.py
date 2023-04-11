from transformers import BertTokenizer
from transformers import TFBertModel
import re
from operator import itemgetter
import nltk
import numpy as np
import json
from huggingface_hub import from_pretrained_keras
import tensorflow as tf
import json
from sentence_transformers import SentenceTransformer, util
from cog import BasePredictor, Input
from typing import Any

nltk.download("stopwords")

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        self.title_model = from_pretrained_keras("focia/text-grade", custom_objects={"TFBertModel": self.bert_model})
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.lst_stopwords = nltk.corpus.stopwords.words("english")

    def bert_encode(self, input_text, max_len):
        input_ids = []
        attension_masks = []
        for text in input_text:
            output_dict = self.tokenizer.encode_plus(
                text, 
                add_special_tokens = True,
                truncation=True,
                max_length = max_len,
                pad_to_max_length = True,
                return_attention_mask = True
            )
            input_ids.append(output_dict['input_ids'])
            attension_masks.append(output_dict['attention_mask'])
        return np.array(input_ids), np.array(attension_masks)

    def utils_preprocess_text(self, text):
        text = str(text).lower()
        text = text.strip().replace("-"," ")
        text = re.sub(r'[^\w\s]', '', text)
        lst_text = text.split()
        if self.lst_stopwords is not None:
            lst_text = [word for word in lst_text if word not in self.lst_stopwords]
        text = " ".join(lst_text)
        return text

    def get_similarity(self, concept, description):
        sentences = [concept, description]
        embeddings = self.sentence_model.encode(sentences)
        result = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return float(result)

    def predict(
        self, 
        concept: str = Input(description="Concept to compare the title generations with"),
        title: str = Input(description="Title to grade"),
        ) -> Any:
        title_cleaned = self.utils_preprocess_text(title)

        input_id, attention_mask = self.bert_encode([title_cleaned], 60)
        sim_score = self.get_similarity(concept, title)
        if sim_score >= 0.6:
            grade = self.title_model.predict((input_id,attention_mask))[0][0]*10*369
            response_dict = {'text': title,'score': float(grade), 'context': 1, 'contextScore': sim_score}

        else:
            grade = self.title_model.predict((input_id,attention_mask))[0][0]*10*369*sim_score
            response_dict = {'text': title,'score': float(grade), 'context': 0, 'contextScore': sim_score}

        return response_dict