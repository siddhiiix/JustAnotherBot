import discord
from discord import app_commands
from discord.ext import commands

from ai.chaos_settings import is_chaos_enabled, set_chaos_enabled


class ChaosCommands(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="chaos",
        description="Toggle your personal Victorian Chaos Mode.",
    )
    @app_commands.describe(
        enabled="Turn Chaos Mode on or off."
    )
    async def chaos(
        self,
        interaction: discord.Interaction,
        enabled: bool,
    ):
        set_chaos_enabled(interaction.user.id, enabled)

        if enabled:
            title = "🎩 CHAOS MODE: ACTIVATED"
            description = (
                "Good heavens! My composure has been utterly compromised!\n\n"
                "Expect theatrical outbursts, absurd observations, "
                "dramatic declarations, and a gentleman operating at "
                "a frankly unreasonable level of eccentricity."
            )
            color = discord.Color.red()
        else:
            title = "🫖 CHAOS MODE: DEACTIVATED"
            description = (
                "Order has been restored to the household. "
                "I shall endeavour to conduct myself with proper "
                "Victorian dignity once more."
            )
            color = discord.Color.dark_gold()

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
    await bot.add_cog(ChaosCommands(bot))