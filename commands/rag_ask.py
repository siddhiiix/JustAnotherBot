import logging

import discord
from discord import app_commands
from discord.ext import commands

from config import DISCORD_MESSAGE_LIMIT
from ai.rag.rag_pipeline import ask_documents

logger = logging.getLogger("JustAnotherBot.RAG")


class RAGAskCommands(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="ask",
        description="Ask a question using your indexed documents.",
    )
    @app_commands.describe(question="What would you like to know from the documents?")
    async def ask(self, interaction: discord.Interaction, question: str):
        await interaction.response.defer(thinking=True)

        try:
            result = await ask_documents(question)
            answer = (result.get("answer") or "I couldn't produce an answer.").strip()
            sources = result.get("sources") or []

            if sources:
                source_lines = ["**Retrieved sources:**"]
                source_lines.extend(
                    f"[{source.get('citation', index)}] `{source.get('source', 'Unknown')}`"
                    for index, source in enumerate(sources, start=1)
                )
                answer = f"{answer}\n\n" + "\n".join(source_lines)

            # Leave a little room below Discord's 2000-character content limit.
            limit = min(DISCORD_MESSAGE_LIMIT, 1900)
            if len(answer) <= limit:
                await interaction.followup.send(answer)
            else:
                chunks = [answer[start:start + limit] for start in range(0, len(answer), limit)]
                await interaction.followup.send(chunks[0])
                for chunk in chunks[1:]:
                    await interaction.followup.send(chunk)
        except Exception:
            logger.exception("The /ask command failed.")
            message = "I ran into an error while searching the documents. Check the bot logs and try again."
            if interaction.response.is_done():
                await interaction.followup.send(message, ephemeral=True)
            else:
                await interaction.response.send_message(message, ephemeral=True)


async def setup(bot: commands.Bot):
    await bot.add_cog(RAGAskCommands(bot))
