import torch
from tools import Chatbotds,ChatDataset,getDataLoader,loss_function,get_optimizer,cuda_or_cpu,draw_loss
from model import Encoder,Decorder,Seq2Seq
import config
from tqdm import tqdm
import math

def train(model,loss,optimizer,epochs,dataitem,words):
    model=model.to(device=torch.device(cuda_or_cpu()))
    loss_all=list()
    min_loss=math.inf
    for epoch in range(epochs):
        loss_sum = 0
        for people,robot in tqdm(dataitem):
            people=people.to(device=torch.device(cuda_or_cpu()))
            robot=robot.to(device=torch.device(cuda_or_cpu()))
            outptut=model(people,robot) #TODO output(100,6,777)
            label=robot[:,:-1]
            l=loss(outptut.reshape(-1,len(words)),label.reshape(-1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            loss_sum+=l.item()
        with torch.no_grad():
            loss_all.append(loss_sum)
            print(f"{epoch+1} loss is {loss_sum}")
            if loss_sum<min_loss:
                torch.save(model.state_dict(),"./best_model.pth")
    return loss_all

if __name__=="__main__":
    chatbots=Chatbotds(dataPath="./data/conversations.corpus.json",max_length=6,set_max_length=True)
    contents=chatbots.load_data()
    sentence_list=chatbots.tokenize(contents)
    people_sentences=chatbots.revise_sentence_list(chatbots.split_chat(contents,people=True),keep=True)
    robot_sentences=chatbots.revise_sentence_list(chatbots.split_chat(contents,people=False),keep=True)
    words=chatbots.revise_sentence_list(sentence_list,keep=False)
    w2i=chatbots.word_to_id(words)
    i2w=chatbots.id_to_word(w2i)
    people=chatbots.word_to_tensor(people_sentences,w2i)
    robot=chatbots.word_to_tensor(robot_sentences,w2i)
    chat_dataset=ChatDataset(people,robot)
    chat_data_loader=getDataLoader(dataset=chat_dataset,shuffle=True,batch_size=10)
    encoder=Encoder(words=words)
    decoder=Decorder(words=words)
    seq2seq=Seq2Seq(encoder,decoder)
    loss=loss_function()
    optimizer=get_optimizer(model=seq2seq,lr=0.001)
    loss_all=train(model=seq2seq,loss=loss,optimizer=optimizer,epochs=config.epochs,dataitem=chat_data_loader,words=words)
    draw_loss(config.epochs,loss_all)