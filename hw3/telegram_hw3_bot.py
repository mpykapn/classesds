from telegram.ext import Application, CommandHandler, MessageHandler, filters
import torch
import torch.nn as nn
import spacy
import numpy as np
import logging

TOKEN = "jfjjfjfjjhgj:hhhhrtrrt123890-"



class NeuralNetWithBatchNorm(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetWithBatchNorm, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.l3(out)
        return out


# chatbot
class Hw3Chatbot():
    def __init__(self, filename = 'chatbot-v3-nndata.pth'):
        self.filename = filename
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return

    def load_model(self):
        data = torch.load(self.filename, map_location=self.device)

        self.input_size = data["input_size"]
        self.hidden_size = data["hidden_size"]
        self.output_size = data["output_size"]
        self.all_words = data["all_words"]
        self.tags = data["tags"]
        self.model_state = data["model_state"]
        self.y = data["y"]
        self.xy = data["xy"]

        self.train_losses = data["train_losses"]
        self.train_accuracy = data["train_accuracy"]
        self.train_precision = data["train_precision"]

        self.model = NeuralNetWithBatchNorm(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.model.load_state_dict(self.model_state)
        self.model.eval()

        self.nlp_chat = spacy.load('ru_core_news_sm')

        return

    def tokenize_chat(self, sentence):
        tokens_chat = []
        doc_chat = self.nlp_chat(sentence.lower())
        token_chat = [token.lemma_ for token in doc_chat if token.is_alpha and not token.is_stop]
        tokens_chat.extend(token_chat)
        return tokens_chat

    def bag_of_words_2(self, tokenized_sentence, words):
        sentence_words = [word for word in tokenized_sentence]
        bag = np.zeros(len(words), dtype=np.float32)
        for idx, w in enumerate(words):
            if w in sentence_words:
                bag[idx] += 1
        return bag

    def answer(self, sentence):
        sentence_tkn = self.tokenize_chat(sentence)
        X = self.bag_of_words_2(sentence_tkn, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).float().to(self.device)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]

        xy_finded = None
        for wxy in self.xy:
            if wxy[1] == str(tag):
                xy_finded = wxy
                break

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        response = "Я не совсем понял..."

        if prob.item() > 0.0:
            for answ in self.y:
                if tag == answ[0]:
                    response = answ[1]
                    break

        return response, sentence_tkn, xy_finded, prob.item()


async def handle_message(update, context):
    user_text = update.message.text
    logging.info(f"input message: {str(user_text)}")

    response, sentence_tkn, xy_finded, prob_item = chatbot.answer(user_text)

    logging.info(f"sentence token: {str(sentence_tkn)}")
    logging.info(f"finded xy: {str(xy_finded)}")
    logging.info(f"prob: {str(prob_item)}")
    logging.info(f"answer: {str(response)}")

    if chatbot_debug:
        await update.message.reply_text(f'sentence token: {sentence_tkn}')
        await update.message.reply_text(f'finded xy: {xy_finded}')
        await update.message.reply_text(f'prob: {prob_item}')
    await update.message.reply_text(response)

async def start(update, context):
    logging.info(f"command start")
    chatbot.load_model()
    await update.message.reply_text("доступные команды /debugon /debugoff /logoff /loginfo /logdebug")

async def debugon(update, context):
    logging.info(f"command debugon")
    global chatbot_debug
    chatbot_debug = True
    await update.message.reply_text("debug включен")

async def debugoff(update, context):
    logging.info(f"command debugoff")
    global chatbot_debug
    chatbot_debug = False
    await update.message.reply_text("debug выключен")

async def logoff(update, context):
    logging.info(f"command logoff")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename='conversation.log', level=logging.CRITICAL,
                        format='%(asctime)s - %(message)s',
                        encoding='utf-8')
    await update.message.reply_text("log выключен")

async def logdebug(update, context):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename='conversation.log', level=logging.DEBUG,
                        format='%(asctime)s - %(message)s',
                        encoding='utf-8')
    logging.info(f"command logdebug")
    await update.message.reply_text("log DEBUG")

async def loginfo(update, context):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename='conversation.log', level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        encoding='utf-8')
    logging.info(f"command loginfo")
    await update.message.reply_text("log INFO")

chatbot = Hw3Chatbot()
chatbot_debug = False

def main():
    logging.basicConfig(filename='conversation.log', level=logging.CRITICAL,
                        format='%(asctime)s - %(message)s',
                        encoding='utf-8')

    chatbot.load_model()

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("debugon", debugon))
    application.add_handler(CommandHandler("debugoff", debugoff))

    application.add_handler(CommandHandler("logoff", logoff))
    application.add_handler(CommandHandler("loginfo", loginfo))
    application.add_handler(CommandHandler("logdebug", logdebug))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()


if __name__ == "__main__":
    main()
