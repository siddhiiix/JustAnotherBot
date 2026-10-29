import discord
from discord import app_commands
from discord.ext import commands

from ai.long_term_memory import get_memories


class MemoriesCommands(commands.Cog):
    """Commands for viewing saved long-term memories."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="memories",
        description="View the memories I have saved about you.",
    )
    async def memories(self, interaction: discord.Interaction):
        """Display the invoking user's saved memories."""

        saved_memories = get_memories(interaction.user.id)

        if not saved_memories:
            await interaction.response.send_message(
                "My notebook appears to be empty, my dear fellow. "
                "Use `/remember` to entrust me with a fact worth keeping.",
                ephemeral=True,
            )
            return

        embed = discord.Embed(
            title="📚 My Personal Notebook",
            description=(
                "Here are the recollections I have preserved for you. "
                "Pray, inspect them at your leisure."
            ),
            color=discord.Color.blurple(),
        )

        # Discord embed descriptions and fields have length limits.
        # Keep the response within those limits.
        lines = []

        for memory in saved_memories:
            content = memory["content"]
            memory_id = memory["id"]

            lines.append(
                f"**ID: {memory_id}**\n"
                f"{content}\n"
            )

        current_field = ""

        for line in lines:
            if len(current_field) + len(line) > 1000:
                embed.add_field(
                    name="Saved Memories",
                    value=current_field,
                    inline=False,
                )
                current_field = ""

            current_field += line

        if current_field:
            embed.add_field(
                name="Saved Memories" if not embed.fields else "Continued",
                value=current_field,
                inline=False,
            )

        embed.set_footer(
            text=(
                f"{len(saved_memories)} memor"
                f"{'y' if len(saved_memories) == 1 else 'ies'} preserved."
            )
        )

        await interaction.response.send_message(
            embed=embed,
            ephemeral=True,
        )


async def setup(bot: commands.Bot):
    """Register the memories command."""

    await bot.add_cog(MemoriesCommands(bot))