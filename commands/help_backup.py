import discord
from discord import app_commands
from discord.ext import commands


class HelpCommands(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="help",
        description="View JustAnotherBot's commands and features.",
    )
    async def help_command(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="🎩 JustAnotherBot — Command Guide",
            description=(
                "Your local AI assistant. Here are the available commands."
            ),
            color=discord.Color.dark_gold(),
        )

        embed.add_field(
            name="🧠 Memory",
            value=(
                "`/remember` — Save a memory manually.\n"
                "`/memories` — View your saved memories.\n"
                "`/forget` — Delete a saved memory by ID.\n"
                "`/memory_auto` — Enable or disable automatic memory."
            ),
            inline=False,
        )

        embed.add_field(
            name="💬 Conversation",
            value=(
                "`/reset` — Clear your history and summary in this channel.\n"
                "`/personality` — Choose your preferred personality.\n"
                "`/chaos` — Toggle your personal Chaos Mode.\n"
                "`/help` — Display this command guide."
            ),
            inline=False,
        )

        embed.add_field(
            name="How to Chat",
            value=(
                "Mention me in a server message or reply to one of my "
                "messages. Recent messages and saved context can help "
                "me follow the conversation."
            ),
            inline=False,
        )

        embed.set_footer(
            text="Automatic memory is opt-in. Use /memory_auto to manage it."
        )

        await interaction.response.send_message(
            embed=embed,
            ephemeral=True,
        )


async def setup(bot: commands.Bot):
    await bot.add_cog(HelpCommands(bot))