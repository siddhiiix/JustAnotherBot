import discord
from discord import app_commands
from discord.ext import commands

from ai.conversation import clear_conversation


class ResetCommands(commands.Cog):
    """Commands for managing conversation history."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="reset",
        description="Clear your conversation history with me in this channel.",
    )
    @app_commands.guild_only()
    async def reset(self, interaction: discord.Interaction):
        """Clear only the invoking user's history in this channel."""

        # A guild and channel are required for this command.
        if interaction.guild_id is None or interaction.channel_id is None:
            await interaction.response.send_message(
                "I beg your pardon, but this command must be used "
                "within a server channel.",
                ephemeral=True,
            )
            return

        # Match the same key used by the existing conversation system.
        conversation_key = (
            interaction.guild_id,
            interaction.channel_id,
            interaction.user.id,
        )

        # Delete only this user's messages in this channel.
        deleted_count = clear_conversation(conversation_key)

        if deleted_count:
            message = (
                "Very well! Your conversation history in this channel "
                "has been cleared. We may begin anew, my dear fellow."
            )
        else:
            message = (
                "There is no conversation history to clear in this "
                "channel. A clean slate already awaits you!"
            )

        await interaction.response.send_message(
            message,
            ephemeral=True,
        )


async def setup(bot: commands.Bot):
    """Register this command extension with the bot."""

    await bot.add_cog(ResetCommands(bot))