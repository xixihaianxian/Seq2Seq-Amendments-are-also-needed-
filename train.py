import torch
from tools import Chatbotds,ChatDataset,getDataLoader,loss_function,get_optimizer,cuda_or_cpu
from model import Encoder,Decorder,Seq2Seq

def train(model,loss,optimizer,epochs,dataitem):
    model=model.to(device=torch.devoce(cuda_or_cpu()))
    for epoch in epochs:
        for people,robot in dataitem:
            people=people.to(device=torch.device(cuda_or_cpu()))
            robot=robot.to(device=torch.device(cuda_or_cpu()))


if __name__=="__main__":
    chatbots=Chatbotds(dataPath="./data/conversations.corpus.json",max_length=6,set_max_length=True)
    contents=chatbots.load_data()
    sentence_list=chatbots.tokenize(contents)
    people_sentences=chatbots.revise_sentence_list(chatbots.split_chat(contents,people=True),keep=True)
    robot_sentences=chatbots.revise_sentence_list(chatbots.split_chat(contents,people=False),keep=True)
    words=chatbots.revise_sentence_list(sentence_list,keep=False)
    w2i=chatbots.word_to_id(words)
    i2w=chatbots.id_to_word(w2i)
    print(i2w)
    people=chatbots.word_to_tensor(people_sentences,w2i)
    robot=chatbots.word_to_tensor(robot_sentences,w2i)
    chat_dataset=ChatDataset(people,robot)
    chat_data_loader=getDataLoader(dataset=chat_dataset,shuffle=True,batch_size=10)
    encoder=Encoder(words=words)
    decoder=Decorder(words=words)
    seq2seq=Seq2Seq(encoder,decoder)
    loss=loss_function()
    optimizer=get_optimizer(model=seq2seq,lr=0.001)
