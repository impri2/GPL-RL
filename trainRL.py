
from libs.trl.trl import  AutoModelForSeq2SeqLMWithValueHead,PPOConfig,PPOTrainer
from transformers import AutoTokenizer
from tqdm import tqdm
import torch

from dataset import BeirCorpusDataset
from reward import Reward
dataset = 'scifact'
device = 'cuda:3'
config = PPOConfig(model_name="BeIR/query-gen-msmarco-t5-base-v1",learning_rate=2e-6,batch_size=256,mini_batch_size=32,gradient_accumulation_steps=8,log_with='tensorboard',ratio_threshold= 30.0, project_kwargs={"logging_dir": '/root/GPL-RL'})
tokenizer = AutoTokenizer.from_pretrained('BeIR/query-gen-msmarco-t5-base-v1')
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained('BeIR/query-gen-msmarco-t5-base-v1')
ppo_trainer = PPOTrainer(model=model,config=config,dataset=BeirCorpusDataset(dataset),tokenizer=tokenizer)
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
 #  "pad_token_id": tokenizer.eos_token_id
}
rewardModel = Reward(dataset,device)
for epoch,batch in tqdm(enumerate(ppo_trainer.dataloader)):
    document_tensors = [tokenizer.encode(document,return_tensors="pt",truncation=True,padding='max_length',max_length=512).squeeze().to(device) for document in batch]
    response_tensors,ref_tensors = ppo_trainer.generate(document_tensors,**generation_kwargs,generate_ref_response=True)
    response = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    print(response)
    ref_response = [tokenizer.decode(r.squeeze()) for r in ref_tensors]
    rewards = rewardModel(batch,response)
    ref_rewards = rewardModel(batch,ref_response)
    stats = ppo_trainer.step(document_tensors,response_tensors,rewards)
    mean_rewards = 0
    mean_ref_rewards = 0
    for reward in rewards:
        mean_rewards += reward.item()
    mean_rewards /= len(rewards)
    for reward in ref_rewards:
        mean_ref_rewards += reward
    mean_ref_rewards /= len(ref_rewards)
    print("rewards: ",mean_rewards,"ref rewards: ",mean_ref_rewards)
    if epoch % 10 == 0:
        ppo_trainer.save_pretrained(f"/root/GPL-RL/models/tsdae/scifact-ce/finetuned{epoch}")
ppo_trainer.save_pretrained("/root/GPL-RL/models/tsdae/scifact-ce/finetuned_final")