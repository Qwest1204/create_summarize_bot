import telebot
from summarization import summarize
from config import TOKEN

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(content_types=['text'])
def get_text_message(message):
    if message.text == "/help":
        bot.send_message(message.chat.id, "send me a text")

    else:
        bot.send_message(message.chat.id, summarize(message.text))

bot.polling(non_stop=True, interval=0)