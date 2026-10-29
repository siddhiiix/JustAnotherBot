import discord
from discord import app_commands
from discord.ext import commands

from ai.long_term_memory import delete_memory


class ForgetCommands(commands.Cog):
    """Commands for deleting saved long-term memories."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="forget",
        description="Forget a specific saved memory.",
    )
    @app_commands.describe(
        memory_id="The ID of the memory you wish me to forget."
    )
    async def forget(
        self,
        interaction: discord.Interaction,
        memory_id: app_commands.Range[int, 1],
    ):
        """Delete a memory belonging to the invoking user."""

        deleted = delete_memory(
            user_id=interaction.user.id,
            memory_id=memory_id,
        )

        if deleted:
            message = (
                "Very well. That particular recollection has been "
                "removed from my notebook. We shall speak of it no more."
            )
        else:
            message = (
                "I could not find that memory in your personal notebook. "
                "Pray, check its ID using `/memories`."
            )

        await interaction.response.send_message(
            message,
            ephemeral=True,
        )


async def setup(bot: commands.Bot):
    """Register the forget command."""

    await bot.add_cog(ForgetCommands(bot))