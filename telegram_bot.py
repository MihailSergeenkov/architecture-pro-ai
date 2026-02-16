#!/usr/bin/env python3
import os

import telebot

from rag import RAGPipeline

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise SystemExit("Set TELEGRAM_BOT_TOKEN env var")

bot = telebot.TeleBot(TELEGRAM_TOKEN)
rag = RAGPipeline()


@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    bot.reply_to(
        message,
        (
            "Привет! Я RAG-бот.\n\n"
            "Задай вопрос по базе знаний, "
            "а я попробую найти ответ и объяснить ход рассуждений."
        ),
    )


@bot.message_handler(func=lambda m: True)
def handle_query(message):
    user_query = message.text.strip()
    if not user_query:
        return

    chat_id = message.chat.id
    bot.send_chat_action(chat_id, "typing")

    try:
        answer = rag.answer(user_query)
    except Exception as e:
        print(f"[ERROR] RAG failed: {e}")
        answer = (
            "Итог: Я не знаю. Что-то пошло не так при обработке запроса."
        )

    # Если ответ слишком длинный — режем на части
    MAX_LEN = 3800
    if len(answer) <= MAX_LEN:
        bot.reply_to(message, answer)
    else:
        for i in range(0, len(answer), MAX_LEN):
            bot.send_message(chat_id, answer[i : i + MAX_LEN])


def main():
    print("Starting Telegram RAG bot...")
    bot.infinity_polling()


if __name__ == "__main__":
    main()
