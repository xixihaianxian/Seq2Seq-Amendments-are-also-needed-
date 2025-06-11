import torch
from model import Encoder,Decorder,Seq2Seq
from tools import Chatbotds
import jieba

chatbots = Chatbotds(dataPath="./data/conversations.corpus.json", max_length=6, set_max_length=True)
contents = chatbots.load_data()
sentence_list = chatbots.tokenize(contents)
people_sentences = chatbots.revise_sentence_list(chatbots.split_chat(contents, people=True), keep=True)
robot_sentences = chatbots.revise_sentence_list(chatbots.split_chat(contents, people=False), keep=True)
words = chatbots.revise_sentence_list(sentence_list, keep=False)
word2id = chatbots.word_to_id(words)
id2word = chatbots.id_to_word(word2id)

encoder = Encoder(words=words)
decoder = Decorder(words=words)
seq2seq = Seq2Seq(encoder, decoder)

params=torch.load("./best_model.pth",weights_only=True)

seq2seq.load_state_dict(state_dict=params)

seq2seq.eval()

issue="你好"

issue=chatbots.cutWord(issue)
issue=[chatbots.cleanSentence(issue)]
issue=chatbots.revise_sentence_list(issue)[0]
issue=torch.tensor([[word2id.get(word) for word in issue]],dtype=torch.long)
dummy_target=torch.randint_like(issue,high=len(words)-1)
robot_prodict=seq2seq(issue,dummy_target)
for id in torch.argmax(robot_prodict,dim=-1).squeeze().tolist():
    print(id2word.get(id))