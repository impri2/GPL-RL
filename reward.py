import torch
from torch.nn import Module,Sigmoid
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,AutoModel
from sentence_transformers import SentenceTransformer,util,models,CrossEncoder

class Reward(Module):
    def __init__(self,dataset:str,device:str):
        super().__init__()
        #self.monoT5 = AutoModelForSeq2SeqLM.from_pretrained("castorini/monot5-base-msmarco-10k").to(device)
       # self.retriever = SentenceTransformer('GPL/scifact-tsdae-msmarco-distilbert-margin-mse').to(device)
        embedding = models.Transformer(f'GPL/{dataset}-tsdae-msmarco-distilbert-margin-mse',max_seq_length=512)
        pooling = models.Pooling(embedding.get_word_embedding_dimension())
        self.retriever = SentenceTransformer(modules=[embedding,pooling],device=device)
        self.T5Tokenizer = AutoTokenizer.from_pretrained("castorini/monot5-base-msmarco-10k")
        self.crossEncoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2',max_length=512,device=device,num_labels=1,default_activation_function=Sigmoid())
        self.device=device
    def cross_score(self,documents,queries):
        scores = self.crossEncoder.predict(list(zip(queries,documents)),convert_to_tensor=True)
        return scores
    def monoT5_score(self,documents,queries):
        t5_sentences =[f"Document: {document} Query: {query} Relevant: " for document,query in zip(documents,queries)]
        ids = self.T5Tokenizer(t5_sentences,padding=True,truncation=True,max_length=512,return_tensors="pt").input_ids.to(self.device)
        outputs = self.T5Tokenizer.batch_decode(self.monoT5.generate(ids,max_length=1),skip_special_tokens=True)
        return torch.tensor(list(map(lambda x: 1 if x == 'true' else 0,outputs))).to(self.device)
    def retriever_score(self,documents,queries):
        document_embeddings = self.retriever.encode(documents,convert_to_tensor=True).to(self.device)
        query_embeddings = self.retriever.encode(queries,convert_to_tensor=True).to(self.device)
        score = util.pairwise_cos_sim(document_embeddings,query_embeddings)
        return 1 - score
    def forward(self,documents,queries):
        with torch.no_grad():
            result = ((self.retriever_score(documents,queries) + self.cross_score(documents,queries))/2).tolist()
            return [torch.tensor(score).to(self.device) for score in result]

if __name__ == "__main__":
   ''' sentences = ["This is an example sentence", "Each sentence is converted"]
   model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v3')
   embeddings = model.encode(sentences)
   print(embeddings) '''
   '''tokenizer = AutoTokenizer.from_pretrained("castorini/monot5-base-msmarco-10k")
   monoT5 =  AutoModelForSeq2SeqLM.from_pretrained("castorini/monot5-base-msmarco-10k")
   sentences = ["Document: python is a programming lanaguage. Query: What is python? Relevant: ","Document: python is a programming lanaguage. Query: What is ruby? Relevant: "]
   ids = tokenizer(sentences,padding=True,truncation=True,max_length=512,return_tensors="pt").input_ids
   outputs = monoT5.generate(ids,max_length=1)
   print(outputs)
   print(tokenizer.batch_decode(outputs, skip_special_tokens=True))'''
   documents = ['Python is a language','Java is a language']
   queries = ['What is Python','What is C++']

   reward = Reward('scifact','cuda:3')
   with torch.no_grad():
     print(reward(documents,queries))

       
  



