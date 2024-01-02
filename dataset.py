from beir import util
from beir.datasets.data_loader import GenericDataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
import pathlib, os

class BeirCorpusDataset(Dataset):
    def load_beir_dataset(self,dataset,split):
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        return {'corpus':corpus,'queries':queries,'qrels':qrels}
    def load_beir_corpus(self,dataset):
        corpus:dict = self.load_beir_dataset(dataset,'train')['corpus']
        res = []
        for document in corpus.values():
            res.append(document['title'] + ' ' + document['text'])   
        return res
    def __init__(self,dataset):
        self.dataset = self.load_beir_corpus(dataset)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,i):
        return self.dataset[i]
    
