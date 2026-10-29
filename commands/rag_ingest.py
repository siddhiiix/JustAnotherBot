import logging

import discord
from discord import app_commands
from discord.ext import commands

from ai.rag.ingest import ingest_all_documents
from ai.rag.vector_store import get_collection_count

logger = logging.getLogger("JustAnotherBot.RAG")


class RAGIngestCommands(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="ingest",
        description="Index PDF and TXT files from the documents folder.",
    )
    @app_commands.default_permissions(manage_guild=True)
    async def ingest(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)

        try:
            result = await ingest_all_documents()
            total_chunks = get_collection_count()

            if result["documents"] == 0 and result["failed"] == 0:
                message = (
                    "I couldn't find any supported PDF or TXT documents in "
                    "the `documents` folder. Add files there and try again."
                )
            else:
                message = (
                    "**Document ingestion complete!**\n"
                    f"Documents indexed this run: **{result['documents']}**\n"
                    f"Chunks added/updated this run: **{result['chunks']}**\n"
                    f"Documents that failed: **{result['failed']}**\n"
                    f"Total chunks in the vector database: **{total_chunks}**"
                )
                if result["failed"]:
                    message += "\nSome files failed; check the bot console/logs for details."

            await interaction.followup.send(message, ephemeral=True)
        except Exception:
            logger.exception("The /ingest command failed.")
            await interaction.followup.send(
                "Ingestion failed. Check the bot console/logs for details.",
                ephemeral=True,
            )


async def setup(bot: commands.Bot):
    await bot.add_cog(RAGIngestCommands(bot))
