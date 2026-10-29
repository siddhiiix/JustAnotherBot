import discord
from discord import app_commands
from discord.ext import commands

from ai.long_term_memory import (
    is_auto_memory_enabled,
    set_auto_memory_enabled,
)


class AutomaticMemoryCommands(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="memory_auto",
        description="Enable or disable automatic memory extraction.",
    )
    @app_commands.describe(
        enabled="Whether the bot may automatically save eligible memories."
    )
    async def memory_auto(
        self,
        interaction: discord.Interaction,
        enabled: bool,
    ):
        set_auto_memory_enabled(interaction.user.id, enabled)

        if enabled:
            title = "🧠 Automatic Memory Enabled"
            description = (
                "I'll look for a small number of useful, stable facts "
                "in your messages and may save eligible memories "
                "automatically.\n\n"
                "Sensitive information is filtered conservatively, "
                "but no automatic filter is perfect. Use `/memories` "
                "to review saved entries and `/forget` to delete them. "
                "Use `/memory_auto enabled: False` to turn this off."
            )
            color = discord.Color.green()
        else:
            title = "🔒 Automatic Memory Disabled"
            description = (
                "I won't initiate new automatic memory extraction. "
                "Your existing saved memories remain available until "
                "you remove them with `/forget`."
            )
            color = discord.Color.dark_grey()

        embed = discord.Embed(
            title=title,
            description=description,
            color=color,
        )

        await interaction.response.send_message(
            embed=embed,
            ephemeral=True,
        )


async def setup(bot: commands.Bot):
    await bot.add_cog(AutomaticMemoryCommands(bot))