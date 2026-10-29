import discord
from discord import app_commands
from discord.ext import commands

from ai.personality_settings import (
    PERSONALITIES,
    get_personality,
    set_personality,
)


class PersonalityCommands(commands.Cog):
    """Commands for changing JustAnotherBot's personality."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="personality",
        description="Change my conversational temperament.",
    )
    @app_commands.describe(
        mode="Choose the personality you would like me to adopt."
    )
    @app_commands.choices(
        mode=[
            app_commands.Choice(
                name="Victorian Gentleman 🎩",
                value="victorian",
            ),
            app_commands.Choice(
                name="Chaotic Gentleman 🎭",
                value="chaotic",
            ),
            app_commands.Choice(
                name="Wholesome Gentleman 🌷",
                value="wholesome",
            ),
            app_commands.Choice(
                name="Roaster Gentleman 🔥",
                value="roaster",
            ),
            app_commands.Choice(
                name="Nerd Gentleman 🧠",
                value="nerd",
            ),
        ]
    )
    async def personality(
        self,
        interaction: discord.Interaction,
        mode: app_commands.Choice[str],
    ):
        """Save the selected personality for the invoking user."""

        user_id = interaction.user.id

        # Save the preference permanently in SQLite.
        set_personality(user_id, mode.value)

        # Retrieve the saved preference to confirm it was stored.
        selected_mode = get_personality(user_id)

        descriptions = {
            "victorian": (
                "I shall return to my customary, distinguished "
                "Victorian demeanour. Order has been restored."
            ),
            "chaotic": (
                "Splendid! A touch of delightful disorder shall "
                "now accompany my impeccable manners. "
                "Let the absurdity commence!"
            ),
            "wholesome": (
                "A most excellent choice. I shall endeavour to "
                "offer greater warmth, encouragement, and "
                "kindness in our conversations."
            ),
            "roaster": (
                "Very well! Kindly prepare yourself for a few "
                "elegantly delivered observations about your "
                "questionable decisions, my dear fellow."
            ),
            "nerd": (
                "Fascinating! I shall approach our discussions "
                "with the enthusiasm of a Victorian scholar "
                "who has just discovered a revolutionary theorem."
            ),
        }

        embed = discord.Embed(
            title="🎩 Personality Updated",
            description=descriptions[selected_mode],
            color=discord.Color.blurple(),
        )

        embed.add_field(
            name="Selected Mode",
            value=mode.name,
            inline=False,
        )

        embed.set_footer(
            text="A gentleman's manners are eternal; only his temperament changes."
        )

        await interaction.response.send_message(
            embed=embed,
            ephemeral=True,
        )


async def setup(bot: commands.Bot):
    """Register this command extension with the bot."""

    await bot.add_cog(PersonalityCommands(bot))