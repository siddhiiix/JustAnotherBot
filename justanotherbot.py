import logging
import discord

from discord.ext import commands

from config import DISCORD_TOKEN, DISCORD_MESSAGE_LIMIT
from ai.ollama_client import generate_response
from ai.prompts import SYSTEM_PROMPT
from ai.conversation import get_conversation, save_message

# -------------------- LOGGING --------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("JustAnotherBot")

# -------------------- DISCORD SETUP --------------------

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)


# -------------------- EVENTS --------------------

@bot.event
async def on_ready():
    logger.info("Logged in as %s (ID: %s)", bot.user, bot.user.id)

    await bot.change_presence(
        activity=discord.Game(name="judging your life choices")
    )


@bot.event
async def on_message(message):
    # Ignore messages from bots and DMs.
    if message.author.bot or not message.guild:
        return

    # Allow prefix commands to work.
    await bot.process_commands(message)

    # Respond only when mentioned or when replying to this bot.
    is_mentioned = bot.user in message.mentions

    is_reply = (
        message.reference is not None
        and message.reference.resolved is not None
        and getattr(message.reference.resolved, "author", None) == bot.user
    )

    if not is_mentioned and not is_reply:
        return

    # Remove the bot mention from the message.
    user_message = message.content

    if bot.user:
        user_message = user_message.replace(
            f"<@{bot.user.id}>", ""
        ).replace(
            f"<@!{bot.user.id}>", ""
        )

    user_message = user_message.strip()

    if not user_message:
        user_message = "Say something entertaining."

    # Keep conversations separate by server, channel, and user.
    conversation_key = (
        message.guild.id,
        message.channel.id,
        message.author.id,
    )

    save_message(conversation_key, "user", user_message)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *get_conversation(conversation_key),
    ]

    try:
        async with message.channel.typing():
            reply = await generate_response(messages)

        if not reply:
            reply = "My brain returned an empty file. Try again."

        save_message(conversation_key, "assistant", reply)

        # Discord messages have a character limit.
        for start in range(0, len(reply), DISCORD_MESSAGE_LIMIT):
            await message.reply(
                reply[start:start + DISCORD_MESSAGE_LIMIT],
                mention_author=False,
            )

    except Exception:
        logger.exception("Failed to respond to a message.")

        await message.reply(
            "My brain just crashed. Give me a second and try again.",
            mention_author=False,
        )


# -------------------- START BOT --------------------

if not DISCORD_TOKEN:
    raise RuntimeError(
        "DISCORD_TOKEN is missing. Check that your .env file exists "
        "and contains DISCORD_TOKEN=your_token"
    )

bot.run(DISCORD_TOKEN)