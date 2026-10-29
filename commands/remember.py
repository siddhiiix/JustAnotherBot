import discord
from discord import app_commands
from discord.ext import commands

from ai.long_term_memory import add_memory


class RememberCommands(commands.Cog):
    """Commands for saving long-term memories."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="remember",
        description="Ask me to remember something about you.",
    )
    @app_commands.describe(
        memory="The information you would like me to remember."
    )
    async def remember(
        self,
        interaction: discord.Interaction,
        memory: str,
    ):
        """Save a memory for the user invoking the command."""

        memory = memory.strip()

        if not memory:
            await interaction.response.send_message(
                "Pray, give me something to remember!",
                ephemeral=True,
            )
            return

        try:
            memory_id = add_memory(
                user_id=interaction.user.id,
                content=memory,
            )

        except ValueError as error:
            await interaction.response.send_message(
                f"Alas! {error}",
                ephemeral=True,
            )
            return

        await interaction.response.send_message(
            f"Splendid! I have committed that fact to memory, "
            f"my dear fellow.\n\n"
            f"**Memory ID:** `{memory_id}`\n"
            f"**Remembered:** {memory}",
            ephemeral=True,
        )


async def setup(bot: commands.Bot):
    """Register the remember command."""

    await bot.add_cog(RememberCommands(bot))