import logging
from collections import Counter

import discord
from discord import app_commands
from discord.ext import commands

from ai.rag.vector_store import collection

logger = logging.getLogger("JustAnotherBot.RAG")


class DocumentCommands(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="documents",
        description="List the documents currently indexed for RAG.",
    )
    async def documents(self, interaction: discord.Interaction):
        try:
            count = collection.count()
            if count == 0:
                await interaction.response.send_message(
                    "No documents are indexed yet. Add PDF/TXT files to the `documents` folder, then ask a server administrator to run `/ingest`.",
                    ephemeral=True,
                )
                return

            result = collection.get(include=["metadatas"])
            metadata = result.get("metadatas") or []
            chunk_counts = Counter(
                item.get("source", "Unknown")
                for item in metadata
                if item
            )

            embed = discord.Embed(
                title="📚 Indexed Documents",
                description=f"**{len(chunk_counts)}** document(s) · **{count}** indexed chunk(s)",
                color=discord.Color.dark_gold(),
            )
            for source, chunks in sorted(chunk_counts.items(), key=lambda item: item[0].lower()):
                embed.add_field(name=source[:256], value=f"Chunks: **{chunks}**", inline=False)
            embed.set_footer(text="Ask questions with /ask · Administrators can refresh the index with /ingest")
            await interaction.response.send_message(embed=embed, ephemeral=True)
        except Exception:
            logger.exception("The /documents command failed.")
            if interaction.response.is_done():
                await interaction.followup.send(
                    "I couldn't retrieve the indexed document list. Check the bot logs.",
                    ephemeral=True,
                )
            else:
                await interaction.response.send_message(
                    "I couldn't retrieve the indexed document list. Check the bot logs.",
                    ephemeral=True,
                )


async def setup(bot: commands.Bot):
    await bot.add_cog(DocumentCommands(bot))
